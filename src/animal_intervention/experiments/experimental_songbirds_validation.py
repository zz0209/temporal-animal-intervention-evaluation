from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from animal_intervention.data.contract import CanonicalDataset
from animal_intervention.estimands.intervention_value import AnchorWindow, node_support
from animal_intervention.transmission.contract import ExposureStream
from animal_intervention.transmission.mappers import GroupMixingMapper

from .wytham_validation import run as run_group_event_validation


EXPECTED_PHASE_EVENTS = {"pre": 10_954, "during": 52_483}
EXPECTED_PHASE_MEMBERSHIPS = {"pre": 50_201, "during": 187_232}
EXPECTED_PHASE_INDIVIDUALS = {"pre": 240, "during": 339}
PHASE_ORDER = ("pre", "during")


def _group_attributes(dataset: CanonicalDataset) -> pd.DataFrame:
    attributes = dataset.group_events["attributes_json"].map(json.loads).apply(pd.Series)
    frame = dataset.group_events[["group_event_id"]].copy()
    frame["experiment_phase"] = attributes["experiment_phase"].astype(str)
    frame["allows_odds"] = attributes["allows_odds"].map(
        lambda value: value
        if isinstance(value, bool)
        else str(value).strip().lower() == "true"
    )
    return frame


def _observed_group_stream(
    dataset: CanonicalDataset, config: dict[str, Any]
) -> ExposureStream:
    mode = str(config["data"].get("group_mixing_mode", "frequency_dependent"))
    full = GroupMixingMapper(mode=mode).compile(dataset)
    observed_nodes = tuple(
        sorted(full.group_memberships["node_id"].astype(str).unique())
    )
    stream = ExposureStream(
        dataset_id=full.dataset_id,
        population_nodes=observed_nodes,
        group_exposures=full.group_exposures.copy(),
        group_memberships=full.group_memberships.copy(),
        metadata={
            **full.metadata,
            "population_definition": "all_birds_observed_in_published_foraging_groups",
        },
    )
    stream.validate()
    return stream


def _phase_bounds(groups: pd.DataFrame, phase: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    selected = groups.loc[groups["experiment_phase"].eq(phase)]
    start = pd.Timestamp(selected["start_time"].min()).floor("D")
    end = pd.Timestamp(selected["end_time"].max()).ceil("D")
    return start, end


def _prepare_windows(
    stream: ExposureStream,
    windows: dict[str, Any],
    max_anchors: int,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    groups = stream.group_exposures.copy()
    groups["start_time"] = pd.to_datetime(groups["start_time"])
    groups["end_time"] = pd.to_datetime(groups["end_time"])
    groups["experiment_phase"] = (
        groups["group_event_id"].astype(str).str.split(":").str[1]
    )
    memberships = stream.group_memberships.copy()
    memberships["node_id"] = memberships["node_id"].astype(str)
    history_days = int(windows["history_days"])
    horizon_days = int(windows["horizon_days"])
    step_days = int(windows["step_days"])
    history_segments = int(windows["history_segments"])
    minimum_segments = int(windows["min_history_segments_per_node"])
    minimum_events = int(windows["min_history_group_events_per_node"])
    if history_days % history_segments:
        raise ValueError("history_days must be divisible by history_segments")
    segment_days = history_days // history_segments

    prepared: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    for phase in PHASE_ORDER:
        phase_start, phase_end = _phase_bounds(groups, phase)
        local_anchor_index = 0
        anchor_time = phase_start + pd.Timedelta(days=history_days)
        while anchor_time + pd.Timedelta(days=horizon_days) <= phase_end:
            history_start = anchor_time - pd.Timedelta(days=history_days)
            horizon_end = anchor_time + pd.Timedelta(days=horizon_days)
            history_groups = groups.loc[
                groups["experiment_phase"].eq(phase)
                & groups["start_time"].ge(history_start)
                & groups["start_time"].lt(anchor_time)
            ].copy()
            forecast_groups = groups.loc[
                groups["experiment_phase"].eq(phase)
                & groups["start_time"].ge(anchor_time)
                & groups["start_time"].lt(horizon_end)
            ].copy()
            if history_groups.empty or forecast_groups.empty:
                raise ValueError(f"{phase} window at {anchor_time} has empty support")

            history_group_ids = set(history_groups["group_event_id"].astype(str))
            forecast_group_ids = set(forecast_groups["group_event_id"].astype(str))
            history_memberships = memberships.loc[
                memberships["group_event_id"].astype(str).isin(history_group_ids)
            ].merge(
                history_groups[["group_event_id", "start_time"]],
                on="group_event_id",
                how="left",
                validate="many_to_one",
            )
            history_memberships["history_segment"] = (
                (history_memberships["start_time"] - history_start)
                .dt.total_seconds()
                .floordiv(segment_days * 86_400)
                .astype(int)
            )
            history_event_count = history_memberships.groupby(
                "node_id", observed=True
            ).size()
            history_segment_count = history_memberships.groupby(
                "node_id", observed=True
            )["history_segment"].nunique()
            eligible = sorted(
                str(node)
                for node in history_event_count.index
                if history_event_count.loc[node] >= minimum_events
                and history_segment_count.loc[node] >= minimum_segments
            )
            if len(eligible) < 2:
                raise ValueError(f"{phase} window at {anchor_time} has <2 candidates")

            eligible_set = set(eligible)
            future_memberships = memberships.loc[
                memberships["group_event_id"].astype(str).isin(forecast_group_ids)
                & memberships["node_id"].isin(eligible_set)
            ].copy()
            retained_sizes = future_memberships.groupby(
                "group_event_id", observed=True
            ).size()
            transmission_group_ids = set(
                retained_sizes.loc[retained_sizes.ge(2)].index.astype(str)
            )
            future_memberships = future_memberships.loc[
                future_memberships["group_event_id"].astype(str).isin(
                    transmission_group_ids
                )
            ].copy()
            future_groups = forecast_groups.loc[
                forecast_groups["group_event_id"].astype(str).isin(
                    transmission_group_ids
                )
            ].drop(columns=["experiment_phase"])
            if future_groups.empty:
                raise ValueError(f"{phase} window at {anchor_time} has no group dyads")

            network_id = "wytham_songbird_manipulation_population"
            future = ExposureStream(
                dataset_id=stream.dataset_id,
                population_nodes=tuple(eligible),
                group_exposures=future_groups,
                group_memberships=future_memberships,
                metadata={
                    **stream.metadata,
                    "network_id": network_id,
                    "experiment_phase": phase,
                    "population_definition": "present_in_both_history_segments",
                },
            )
            future.validate()
            local_anchor_index += 1
            anchor_id = f"{phase}::anchor_{local_anchor_index:03d}"
            anchor = AnchorWindow(
                anchor_id=anchor_id,
                history_start=history_start,
                anchor_time=anchor_time,
                horizon_end=horizon_end,
            )
            future_support = node_support(future)
            future_active = {node for node, count in future_support.items() if count > 0}
            history_spans = [
                (
                    history_start + pd.Timedelta(days=index * segment_days),
                    history_start + pd.Timedelta(days=(index + 1) * segment_days),
                )
                for index in range(history_segments)
            ]
            prepared.append(
                {
                    "anchor": anchor,
                    "future": future,
                    "eligible": eligible,
                    "history_support": history_event_count,
                    "history_period_support": history_segment_count,
                    "future_support": future_support,
                    "population_size": len(eligible),
                    "network_id": network_id,
                    "observation_unit_id": phase,
                    "history_weekend_starts": [start for start, _ in history_spans],
                    "history_weekend_spans": history_spans,
                    "forecast_weekend_start": anchor_time,
                    "future_active_count": len(future_active),
                }
            )
            metadata_rows.append(
                {
                    "dataset_id": stream.dataset_id,
                    "network_id": network_id,
                    "observation_phase": phase,
                    "anchor_id": anchor_id,
                    "history_start": history_start,
                    "anchor_time": anchor_time,
                    "horizon_end": horizon_end,
                    "history_segments": history_segments,
                    "eligible_population": len(eligible),
                    "future_active_population": len(future_active),
                    "future_active_fraction": len(future_active) / len(eligible),
                    "transmission_group_events": len(future_groups),
                }
            )
            if len(prepared) >= max_anchors:
                return prepared, pd.DataFrame(metadata_rows)
            anchor_time += pd.Timedelta(days=step_days)
    return prepared, pd.DataFrame(metadata_rows)


def _tag_parity_map(dataset: CanonicalDataset) -> dict[str, str]:
    return {
        str(row.node_id): str(json.loads(row.attributes_json)["tag_parity"])
        for row in dataset.individuals.itertuples(index=False)
    }


def _linked_wytham_interval_overlap(dataset: CanonicalDataset) -> dict[str, int]:
    linked_path = Path(__file__).resolve().parents[3] / "data/wytham_great_tits_divorce/processed"
    if not linked_path.exists():
        return {"exact_interval_matches": 0, "matched_songbird_events": 0}
    linked = CanonicalDataset.read(linked_path)
    matches = dataset.group_events[["group_event_id", "start_time", "end_time"]].merge(
        linked.group_events[["group_event_id", "start_time", "end_time"]],
        on=["start_time", "end_time"],
        suffixes=("_songbird", "_wytham"),
    )
    return {
        "exact_interval_matches": int(len(matches)),
        "matched_songbird_events": int(matches["group_event_id_songbird"].nunique()),
    }


def _data_quality_audit(
    dataset: CanonicalDataset,
    stream: ExposureStream,
    prepared: list[dict[str, Any]],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    raw_attributes = _group_attributes(dataset)
    raw_groups = dataset.group_events.merge(
        raw_attributes, on="group_event_id", validate="one_to_one"
    )
    raw_memberships = dataset.group_memberships.merge(
        raw_attributes[["group_event_id", "experiment_phase"]],
        on="group_event_id",
        validate="many_to_one",
    )
    groups = stream.group_exposures.merge(
        raw_attributes,
        on="group_event_id",
        validate="one_to_one",
    )
    groups["start_time"] = pd.to_datetime(groups["start_time"])
    groups["end_time"] = pd.to_datetime(groups["end_time"])
    memberships = stream.group_memberships.copy()
    memberships["node_id"] = memberships["node_id"].astype(str)
    sizes = memberships.groupby("group_event_id", observed=True).size().rename(
        "host_group_size"
    )
    groups = groups.merge(sizes, on="group_event_id", validate="one_to_one")
    groups["date"] = groups["start_time"].dt.floor("D")
    daily = (
        groups.groupby(["experiment_phase", "date"], observed=True)
        .agg(group_events=("group_event_id", "size"))
        .reset_index()
        .rename(columns={"experiment_phase": "season"})
    )
    daily_active = (
        memberships.merge(
            groups[["group_event_id", "experiment_phase", "date"]],
            on="group_event_id",
            validate="many_to_one",
        )
        .groupby(["experiment_phase", "date"], observed=True)["node_id"]
        .nunique()
        .rename("active_host_individuals")
        .reset_index()
        .rename(columns={"experiment_phase": "season"})
    )
    daily = daily.merge(daily_active, on=["season", "date"], validate="one_to_one")

    parity = _tag_parity_map(dataset)
    during = memberships.merge(
        groups.loc[
            groups["experiment_phase"].eq("during"),
            ["group_event_id", "allows_odds"],
        ],
        on="group_event_id",
        validate="many_to_one",
    )
    during["tag_is_odd"] = during["node_id"].map(parity).eq("odd")
    matched = during["tag_is_odd"].eq(during["allows_odds"])
    raw_phase_events = raw_groups.groupby("experiment_phase", observed=True).size()
    raw_phase_memberships = raw_memberships.groupby(
        "experiment_phase", observed=True
    ).size()
    raw_phase_individuals = raw_memberships.groupby(
        "experiment_phase", observed=True
    )["node_id"].nunique()
    durations = (groups["end_time"] - groups["start_time"]).dt.total_seconds()
    transmission_durations = durations.loc[groups["host_group_size"].ge(2)]
    linked_overlap = _linked_wytham_interval_overlap(dataset)
    phase_anchor_counts = pd.Series(
        [item["observation_unit_id"] for item in prepared]
    ).value_counts()
    checks = {
        "published_phase_event_counts": raw_phase_events.to_dict()
        == EXPECTED_PHASE_EVENTS,
        "published_phase_membership_counts": raw_phase_memberships.to_dict()
        == EXPECTED_PHASE_MEMBERSHIPS,
        "published_phase_individual_counts": raw_phase_individuals.to_dict()
        == EXPECTED_PHASE_INDIVIDUALS,
        "five_roster_species": dataset.individuals["species"].nunique() == 5,
        "all_observed_nodes_have_roster_metadata": set(memberships["node_id"]).issubset(
            set(dataset.individuals["node_id"].astype(str))
        ),
        "all_tags_have_parity": all(value in {"odd", "even"} for value in parity.values()),
        "manipulation_match_direction": float(matched.mean()) > 0.5,
        "nonpositive_intervals_reconciled": int(
            pd.to_numeric(raw_groups["duration_seconds"], errors="coerce").le(0).sum()
        )
        == int(stream.metadata["excluded_nonpositive_or_missing_intervals"]),
        "unique_group_memberships": not memberships.duplicated(
            ["group_event_id", "node_id"]
        ).any(),
        "all_windows_stay_within_phase": all(
            item["anchor"].anchor_id.split("::", 1)[0]
            == item["observation_unit_id"]
            for item in prepared
        ),
        "closed_future_population": all(
            item["future"].nodes() == set(item["eligible"]) for item in prepared
        ),
        "eligibility_requires_both_history_segments": all(
            item["history_period_support"].reindex(item["eligible"]).ge(2).all()
            for item in prepared
        ),
        "expected_phase_anchor_counts": (
            int(phase_anchor_counts.sum()) == len(prepared)
            and int(phase_anchor_counts.get("pre", 0)) <= 3
            and int(phase_anchor_counts.get("during", 0)) <= 10
            and set(phase_anchor_counts.index).issubset(set(PHASE_ORDER))
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    audit = {
        "status": "passed" if all(checks.values()) else "needs_revision",
        "checks": checks,
        "registered_individuals": int(len(dataset.individuals)),
        "observed_group_stream_individuals": int(memberships["node_id"].nunique()),
        "species_counts": {
            str(key): int(value)
            for key, value in dataset.individuals["species"].value_counts().items()
        },
        "raw_group_events_by_phase": {
            str(key): int(value) for key, value in raw_phase_events.items()
        },
        "raw_memberships_by_phase": {
            str(key): int(value) for key, value in raw_phase_memberships.items()
        },
        "raw_individuals_by_phase": {
            str(key): int(value) for key, value in raw_phase_individuals.items()
        },
        "excluded_nonpositive_group_events": int(
            stream.metadata["excluded_nonpositive_or_missing_intervals"]
        ),
        "mapped_group_events": int(len(groups)),
        "mapped_group_memberships": int(len(memberships)),
        "singleton_group_events": int(groups["host_group_size"].eq(1).sum()),
        "transmission_group_events": int(groups["host_group_size"].ge(2).sum()),
        "during_feeder_tag_match_fraction": float(matched.mean()),
        "host_group_size": {
            "median": float(groups["host_group_size"].median()),
            "p90": float(groups["host_group_size"].quantile(0.90)),
            "p99": float(groups["host_group_size"].quantile(0.99)),
            "maximum": int(groups["host_group_size"].max()),
        },
        "group_duration_seconds": {
            "median": float(durations.median()),
            "p90": float(durations.quantile(0.90)),
            "p99": float(durations.quantile(0.99)),
            "maximum": float(durations.max()),
        },
        "transmission_event_duration_tail": {
            "events_over_one_hour": int(transmission_durations.gt(3_600).sum()),
            "duration_share_over_one_hour": float(
                transmission_durations.loc[transmission_durations.gt(3_600)].sum()
                / transmission_durations.sum()
            ),
        },
        "analysis_windows": int(len(prepared)),
        "phase_anchor_counts": {
            str(key): int(value) for key, value in phase_anchor_counts.items()
        },
        "eligible_population_range": [
            min(len(item["eligible"]) for item in prepared),
            max(len(item["eligible"]) for item in prepared),
        ],
        "future_active_fraction_range": [
            min(item["future_active_count"] / len(item["eligible"]) for item in prepared),
            max(item["future_active_count"] / len(item["eligible"]) for item in prepared),
        ],
        "linked_wytham_observation_overlap": linked_overlap,
        "primary_mapper": stream.metadata["mapper"],
        "group_mixing_mode": stream.metadata["mode"],
        "beta_unit": stream.metadata["beta_unit"],
        "edge_semantics": "author-inferred mixed-species co-flocking association",
        "analytical_risk": (
            "The 2013-14 interval partially overlaps the Wytham divorce deposit. "
            "Treat both datasets as one linked observation family in cross-dataset splits."
        ),
    }
    return audit, daily, groups[["group_event_id", "host_group_size"]].copy()


def _evaluation_units(anchor_metadata: pd.DataFrame) -> pd.DataFrame:
    units = anchor_metadata[["dataset_id", "network_id"]].drop_duplicates().copy()
    units["evaluation_unit_id"] = units["dataset_id"] + "::" + units["network_id"]
    units["independent_unit_type"] = "longitudinal_experimental_population"
    units["evaluation_dependency_id"] = (
        "wytham_woods_2013_14_feeder_observation_family"
    )
    units["linked_dataset_id"] = "wytham_great_tits_divorce"
    units["split_constraint"] = (
        "keep this dataset and Wytham winter_2013_14 in the same outer fold"
    )
    return units.reset_index(drop=True)


def run(config_path: Path, profile: str) -> tuple[Path, Path]:
    return run_group_event_validation(
        config_path,
        profile,
        stream_builder=_observed_group_stream,
        window_builder=_prepare_windows,
        audit_builder=_data_quality_audit,
        evaluation_units_builder=_evaluation_units,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run experimental-songbird data, calibration, and label validation."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()
    results_dir, report_dir = run(args.config, args.profile)
    print(f"Results: {results_dir}")
    print(f"Report: {report_dir}")


if __name__ == "__main__":
    main()
