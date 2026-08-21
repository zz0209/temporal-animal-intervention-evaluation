from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from animal_intervention.data.contract import CanonicalDataset
from animal_intervention.estimands.intervention_value import AnchorWindow, node_support
from animal_intervention.transmission.contract import ExposureStream
from animal_intervention.transmission.mappers import GroupMixingMapper

from .wytham_validation import run as run_group_event_validation


EXPECTED_SAMPLING_PERIODS = {
    "summer": 14,
    "autumn": 3,
    "winter": 3,
    "spring": 3,
}


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
            "population_definition": "all_observed_great_tits_in_published_gmm_events",
        },
    )
    stream.validate()
    return stream


def _sampling_table(stream: ExposureStream) -> pd.DataFrame:
    groups = stream.group_exposures.copy()
    groups["start_time"] = pd.to_datetime(groups["start_time"])
    groups["end_time"] = pd.to_datetime(groups["end_time"])
    groups["season"] = groups["group_event_id"].astype(str).str.split(":").str[1]
    groups["sampling_period_start"] = (
        groups["start_time"].dt.to_period("W-SUN").dt.start_time
    )
    return groups


def _prepare_windows(
    stream: ExposureStream,
    windows: dict[str, Any],
    max_anchors: int,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    groups = _sampling_table(stream)
    memberships = stream.group_memberships.copy()
    memberships["node_id"] = memberships["node_id"].astype(str)
    history_periods = int(windows["history_sampling_periods"])
    minimum_history_periods = int(windows["min_history_sampling_periods_per_node"])
    minimum_history_events = int(windows["min_history_group_events_per_node"])
    forecast_indices = {
        str(season): [int(value) for value in values]
        for season, values in windows["forecast_sampling_period_indices"].items()
    }
    prepared: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    for season in ["summer", "autumn", "winter", "spring"]:
        season_groups = groups.loc[groups["season"].eq(season)]
        sampling_periods = sorted(season_groups["sampling_period_start"].unique())
        local_anchor_index = 0
        for forecast_index in forecast_indices.get(season, []):
            if forecast_index < history_periods or forecast_index >= len(sampling_periods):
                raise ValueError(
                    f"{season} cannot support forecast sampling-period index {forecast_index}"
                )
            history_starts = sampling_periods[
                forecast_index - history_periods : forecast_index
            ]
            forecast_start = sampling_periods[forecast_index]
            history_groups = season_groups.loc[
                season_groups["sampling_period_start"].isin(history_starts)
            ]
            forecast_groups = season_groups.loc[
                season_groups["sampling_period_start"].eq(forecast_start)
            ]
            history_group_ids = set(history_groups["group_event_id"].astype(str))
            forecast_group_ids = set(forecast_groups["group_event_id"].astype(str))
            history_memberships = memberships.loc[
                memberships["group_event_id"].astype(str).isin(history_group_ids)
            ].merge(
                history_groups[["group_event_id", "sampling_period_start"]],
                on="group_event_id",
                how="left",
                validate="many_to_one",
            )
            history_event_count = history_memberships.groupby(
                "node_id", observed=True
            ).size()
            history_period_count = history_memberships.groupby(
                "node_id", observed=True
            )["sampling_period_start"].nunique()
            eligible = sorted(
                str(node)
                for node in history_event_count.index
                if history_event_count.loc[node] >= minimum_history_events
                and history_period_count.loc[node] >= minimum_history_periods
            )
            if len(eligible) < 2:
                raise ValueError(f"{season} forecast index {forecast_index} has <2 candidates")
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
            ].drop(columns=["season", "sampling_period_start"])
            if future_groups.empty:
                raise ValueError(f"{season} forecast index {forecast_index} has no dyads")
            network_id = "radolfzell_longitudinal_population"
            future = ExposureStream(
                dataset_id=stream.dataset_id,
                population_nodes=tuple(eligible),
                group_exposures=future_groups,
                group_memberships=future_memberships,
                metadata={
                    **stream.metadata,
                    "network_id": network_id,
                    "observation_season": season,
                    "population_definition": "present_in_both_history_sampling_periods",
                },
            )
            future.validate()
            history_start = pd.Timestamp(history_groups["start_time"].min())
            anchor_time = pd.Timestamp(forecast_groups["start_time"].min())
            horizon_end = pd.Timestamp(forecast_groups["end_time"].max())
            local_anchor_index += 1
            anchor_id = f"{season}::anchor_{local_anchor_index:03d}"
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
                    pd.Timestamp(
                        history_groups.loc[
                            history_groups["sampling_period_start"].eq(period),
                            "start_time",
                        ].min()
                    ),
                    pd.Timestamp(
                        history_groups.loc[
                            history_groups["sampling_period_start"].eq(period),
                            "end_time",
                        ].max()
                    ),
                )
                for period in history_starts
            ]
            prepared.append(
                {
                    "anchor": anchor,
                    "future": future,
                    "eligible": eligible,
                    "history_support": history_event_count,
                    "history_period_support": history_period_count,
                    "future_support": future_support,
                    "population_size": len(eligible),
                    "network_id": network_id,
                    "observation_unit_id": season,
                    "history_weekend_starts": list(map(pd.Timestamp, history_starts)),
                    "history_weekend_spans": history_spans,
                    "forecast_weekend_start": pd.Timestamp(forecast_start),
                    "future_active_count": len(future_active),
                }
            )
            metadata_rows.append(
                {
                    "dataset_id": stream.dataset_id,
                    "network_id": network_id,
                    "observation_season": season,
                    "anchor_id": anchor_id,
                    "history_start": history_start,
                    "anchor_time": anchor_time,
                    "horizon_end": horizon_end,
                    "history_sampling_periods": history_periods,
                    "forecast_sampling_period_start": pd.Timestamp(forecast_start),
                    "eligible_population": len(eligible),
                    "future_active_population": len(future_active),
                    "future_active_fraction": len(future_active) / len(eligible),
                    "transmission_group_events": len(future_groups),
                }
            )
            if len(prepared) >= max_anchors:
                return prepared, pd.DataFrame(metadata_rows)
    return prepared, pd.DataFrame(metadata_rows)


def _data_quality_audit(
    dataset: CanonicalDataset,
    stream: ExposureStream,
    prepared: list[dict[str, Any]],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    groups = _sampling_table(stream)
    memberships = stream.group_memberships.copy()
    memberships["node_id"] = memberships["node_id"].astype(str)
    sizes = memberships.groupby("group_event_id", observed=True).size().rename(
        "host_group_size"
    )
    groups = groups.merge(sizes, on="group_event_id", how="left", validate="one_to_one")
    groups["date"] = groups["start_time"].dt.floor("D")
    daily = (
        groups.groupby(["season", "date"], observed=True)
        .agg(group_events=("group_event_id", "size"))
        .reset_index()
    )
    daily_active = (
        memberships.merge(
            groups[["group_event_id", "season", "date"]],
            on="group_event_id",
            how="left",
            validate="many_to_one",
        )
        .groupby(["season", "date"], observed=True)["node_id"]
        .nunique()
        .rename("active_host_individuals")
        .reset_index()
    )
    daily = daily.merge(daily_active, on=["season", "date"], validate="one_to_one")
    sampling_periods = groups.groupby("season", observed=True)[
        "sampling_period_start"
    ].nunique()
    duration = (groups["end_time"] - groups["start_time"]).dt.total_seconds()
    transmission_durations = duration.loc[
        sizes.reindex(groups["group_event_id"]).ge(2).to_numpy()
    ]
    all_group_durations = (
        pd.to_datetime(dataset.group_events["end_time"])
        - pd.to_datetime(dataset.group_events["start_time"])
    ).dt.total_seconds()
    source_roster_nodes = set(
        dataset.individuals.loc[
            dataset.individuals["species"].eq("GRETI"), "node_id"
        ].astype(str)
    )
    observed_nodes = set(memberships["node_id"].astype(str))
    checks = {
        "four_observation_seasons": set(sampling_periods.index)
        == set(EXPECTED_SAMPLING_PERIODS),
        "expected_sampling_periods_by_season": all(
            int(sampling_periods.get(season, -1)) == expected
            for season, expected in EXPECTED_SAMPLING_PERIODS.items()
        ),
        "positive_mapped_group_intervals": duration.gt(0).all(),
        "nonpositive_intervals_reconciled": int(all_group_durations.le(0).sum())
        == int(stream.metadata["excluded_nonpositive_or_missing_intervals"]),
        "unique_group_memberships": not memberships.duplicated(
            ["group_event_id", "node_id"]
        ).any(),
        "observed_population_reconciles": len(stream.population_nodes)
        == memberships["node_id"].nunique(),
        "all_windows_stay_within_one_season": all(
            item["anchor"].anchor_id.split("::", 1)[0]
            == item["observation_unit_id"]
            for item in prepared
        ),
        "closed_future_population": all(
            item["future"].nodes() == set(item["eligible"]) for item in prepared
        ),
        "eligibility_requires_both_history_periods": all(
            item["history_period_support"].reindex(item["eligible"]).ge(2).all()
            for item in prepared
        ),
        "all_forecast_groups_have_two_candidates": all(
            item["future"].group_memberships.groupby("group_event_id").size().ge(2).all()
            for item in prepared
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    p99_duration = float(transmission_durations.quantile(0.99))
    audit = {
        "status": "passed" if all(checks.values()) else "needs_revision",
        "checks": checks,
        "all_registered_individuals": int(len(dataset.individuals)),
        "host_species_roster_individuals": int(len(source_roster_nodes)),
        "observed_host_species_individuals": int(memberships["node_id"].nunique()),
        "observed_individuals_with_roster_metadata": int(
            len(observed_nodes & source_roster_nodes)
        ),
        "observed_individuals_without_roster_metadata": int(
            len(observed_nodes - source_roster_nodes)
        ),
        "all_group_events": int(len(dataset.group_events)),
        "excluded_nonpositive_all_species_group_events": int(
            stream.metadata["excluded_nonpositive_or_missing_intervals"]
        ),
        "mapped_group_events": int(len(groups)),
        "group_memberships": int(len(memberships)),
        "singleton_group_events": int(sizes.eq(1).sum()),
        "transmission_group_events": int(sizes.ge(2).sum()),
        "recording_dates": int(daily["date"].nunique()),
        "recording_periods_by_season": {
            key: int(value) for key, value in sampling_periods.items()
        },
        "host_group_size": {
            "median": float(sizes.median()),
            "p90": float(sizes.quantile(0.90)),
            "p99": float(sizes.quantile(0.99)),
            "maximum": int(sizes.max()),
        },
        "group_duration_seconds": {
            "median": float(duration.median()),
            "p90": float(duration.quantile(0.90)),
            "p99": float(duration.quantile(0.99)),
            "maximum": float(duration.max()),
        },
        "transmission_event_duration_tail": {
            "p99_seconds": p99_duration,
            "events_over_one_hour": int(transmission_durations.gt(3600).sum()),
            "duration_share_over_one_hour": float(
                transmission_durations.loc[transmission_durations.gt(3600)].sum()
                / transmission_durations.sum()
            ),
        },
        "analysis_windows": len(prepared),
        "eligible_population_range": [
            min(len(item["eligible"]) for item in prepared),
            max(len(item["eligible"]) for item in prepared),
        ],
        "future_active_fraction_range": [
            min(item["future_active_count"] / len(item["eligible"]) for item in prepared),
            max(item["future_active_count"] / len(item["eligible"]) for item in prepared),
        ],
        "primary_mapper": stream.metadata["mapper"],
        "group_mixing_mode": stream.metadata["mode"],
        "beta_unit": stream.metadata["beta_unit"],
        "edge_semantics": "author-inferred RFID co-flocking association, not physical contact",
        "analytical_risk": (
            "Long inferred group spans contribute more exposure under duration mapping; "
            "the primary analysis retains the published intervals and calibrates beta, "
            "with duration sensitivity reserved as a secondary arm."
        ),
    }
    return audit, daily, sizes.reset_index()


def _evaluation_units(anchor_metadata: pd.DataFrame) -> pd.DataFrame:
    units = anchor_metadata[["dataset_id", "network_id"]].drop_duplicates().copy()
    units["evaluation_unit_id"] = units["dataset_id"] + "::" + units["network_id"]
    units["independent_unit_type"] = "longitudinal_population"
    units["split_constraint"] = (
        "use prospective season splits; group repeated bird identities explicitly"
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
        description="Run Radolfzell great-tit data, calibration, and label validation."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()
    results_dir, report_dir = run(args.config, args.profile)
    print(f"Results: {results_dir}")
    print(f"Report: {report_dir}")


if __name__ == "__main__":
    main()
