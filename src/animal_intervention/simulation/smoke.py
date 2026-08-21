from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm

from animal_intervention.data.adapters import ADAPTERS
from animal_intervention.data.contract import CanonicalDataset
from animal_intervention.simulation.sir import SIRParameters, TemporalSIREngine
from animal_intervention.transmission.contract import ExposureStream
from animal_intervention.transmission.mappers import compile_primary_exposure


def _trim_stream(stream: ExposureStream, max_events: int) -> ExposureStream:
    dyadic = stream.dyadic_exposures.sort_values("start_time").head(max_events).copy()
    groups = stream.group_exposures.sort_values("start_time").head(max_events).copy()
    memberships = stream.group_memberships.loc[
        stream.group_memberships["group_event_id"].isin(groups["group_event_id"])
    ].copy()
    return ExposureStream(
        dataset_id=stream.dataset_id,
        dyadic_exposures=dyadic,
        group_exposures=groups,
        group_memberships=memberships,
        metadata={**stream.metadata, "smoke_max_events": max_events},
    )


def _initial_node(stream: ExposureStream) -> str:
    if not stream.group_memberships.empty:
        sizes = stream.group_memberships.groupby("group_event_id", observed=True).size()
        group_id = str(sizes.idxmax())
        return str(
            stream.group_memberships.loc[
                stream.group_memberships["group_event_id"].astype(str).eq(group_id),
                "node_id",
            ].iloc[0]
        )
    endpoints = pd.concat(
        [stream.dyadic_exposures["source_id"], stream.dyadic_exposures["target_id"]]
    ).astype(str)
    return str(endpoints.value_counts().index[0])


def run_smoke(
    *,
    data_root: Path,
    output_dir: Path,
    seeds: list[int],
    max_events: int,
    progress: bool,
) -> pd.DataFrame:
    """Run non-inferential end-to-end checks across every canonical modality."""
    rows: list[dict] = []
    engine = TemporalSIREngine()
    for dataset_id in tqdm(ADAPTERS, desc="primary mapper/SIR smoke", disable=not progress):
        dataset = CanonicalDataset.read(data_root / dataset_id / "interim" / "smoke")
        try:
            stream = _trim_stream(compile_primary_exposure(dataset), max_events)
        except ValueError as error:
            expected = not dataset.metadata.has_temporal_order
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "status": "expected_incompatible" if expected else "failed",
                    "reason": str(error),
                    "mapper": None,
                    "seed": None,
                    "initial_node": None,
                    "exposure_events": 0,
                    "exposure_nodes": 0,
                    "final_size": None,
                    "peak_infectious": None,
                }
            )
            if not expected:
                raise
            continue
        mapper_name = str(stream.metadata["mapper"])
        beta = 2.0 if mapper_name == "AggregatedAssociationMapper" else 0.01
        initial = _initial_node(stream)
        for seed in tqdm(
            seeds,
            desc=dataset_id,
            disable=not progress,
            leave=False,
            unit="seed",
        ):
            result = engine.simulate(
                stream,
                SIRParameters(beta=beta, recovery_rate=0.0),
                initial_infected=[initial],
                seed=seed,
            )
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "status": "ok",
                    "reason": "non-inferential interface smoke test",
                    "mapper": mapper_name,
                    "seed": seed,
                    "initial_node": initial,
                    "exposure_events": len(stream.dyadic_exposures)
                    + len(stream.group_exposures),
                    "exposure_nodes": len(stream.nodes()),
                    "final_size": result.final_size,
                    "peak_infectious": result.peak_infectious,
                }
            )
    result_frame = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_frame.to_csv(output_dir / "primary_mapper_sir_smoke.csv", index=False)
    (output_dir / "README.md").write_text(
        "# Primary mapper and temporal SIR smoke test\n\n"
        "These runs verify interface compatibility and temporal execution only. "
        "Their arbitrary smoke parameters are not epidemiological estimates and must not be cited as results.\n",
        encoding="utf-8",
    )
    _save_figure(result_frame, output_dir / "primary_mapper_sir_smoke.png")
    return result_frame


def _save_figure(frame: pd.DataFrame, path: Path) -> None:
    successful = frame.loc[frame["status"].eq("ok")].copy()
    sns.set_theme(style="whitegrid", context="notebook")
    figure, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    sns.stripplot(
        data=successful,
        y="dataset_id",
        x="final_size",
        jitter=0.12,
        size=7,
        ax=axes[0],
        color="#2878B5",
    )
    axes[0].set_title("Smoke-test epidemic final size across seeds")
    axes[0].set_xlabel("ever infected (functional check only)")
    mapper_counts = successful.drop_duplicates("dataset_id").sort_values("exposure_events")
    axes[1].barh(
        mapper_counts["dataset_id"], mapper_counts["exposure_events"], color="#39A96B"
    )
    axes[1].set_xscale("log")
    axes[1].set_title("Exposure events passed to the simulator")
    axes[1].set_xlabel("events (log scale)")
    figure.suptitle("Canonical adaptor → transmission mapper → temporal SIR audit")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Temporal Animal Intervention Evaluation mapper/SIR smoke tests")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/smoke_simulation"))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--max-events", type=int, default=5_000)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    frame = run_smoke(
        data_root=args.data_root,
        output_dir=args.output_dir,
        seeds=list(range(args.seeds)),
        max_events=args.max_events,
        progress=not args.no_progress,
    )
    print(json.dumps(frame.to_dict("records"), indent=2, default=str))
    return 1 if frame["status"].eq("failed").any() else 0


if __name__ == "__main__":
    raise SystemExit(main())
