from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from .contract import CanonicalDataset, TABLE_COLUMNS


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    severity: str
    code: str
    message: str
    count: int = 0
    fraction: float | None = None


@dataclass(slots=True)
class ValidationReport:
    dataset_id: str
    issues: list[ValidationIssue]
    metrics: dict[str, Any]

    @property
    def has_errors(self) -> bool:
        return any(issue.severity in {"critical", "high"} for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "has_errors": self.has_errors,
            "metrics": self.metrics,
            "issues": [asdict(issue) for issue in self.issues],
        }


def _fraction(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def validate_dataset(dataset: CanonicalDataset) -> ValidationReport:
    issues: list[ValidationIssue] = []
    metrics: dict[str, Any] = dataset.summary()

    for table_name, required_columns in TABLE_COLUMNS.items():
        frame = getattr(dataset, table_name)
        missing = sorted(set(required_columns) - set(frame.columns))
        if missing:
            issues.append(
                ValidationIssue(
                    "critical",
                    "missing_columns",
                    f"{table_name} is missing required columns: {missing}",
                    len(missing),
                )
            )

    individuals = dataset.individuals
    duplicate_nodes = int(individuals["node_id"].astype("string").duplicated().sum())
    missing_nodes = int(individuals["node_id"].isna().sum())
    metrics["duplicate_node_ids"] = duplicate_nodes
    if duplicate_nodes:
        issues.append(
            ValidationIssue(
                "high",
                "duplicate_node_ids",
                "individuals must contain one row per node_id",
                duplicate_nodes,
                _fraction(duplicate_nodes, len(individuals)),
            )
        )
    if missing_nodes:
        issues.append(
            ValidationIssue("critical", "missing_node_ids", "node_id is required", missing_nodes)
        )

    known_nodes = set(individuals["node_id"].dropna().astype(str))
    dyadic = dataset.dyadic_events
    if len(dyadic):
        duplicate_events = int(dyadic["event_id"].astype("string").duplicated().sum())
        self_loops = int((dyadic["source_id"].astype(str) == dyadic["target_id"].astype(str)).sum())
        missing_endpoints = int(dyadic[["source_id", "target_id"]].isna().any(axis=1).sum())
        orphan_nodes = (
            set(dyadic["source_id"].dropna().astype(str))
            | set(dyadic["target_id"].dropna().astype(str))
        ) - known_nodes
        metrics.update(
            {
                "duplicate_dyadic_event_ids": duplicate_events,
                "dyadic_self_loops": self_loops,
                "dyadic_orphan_nodes": len(orphan_nodes),
            }
        )
        flagged_source_duplicates = int(
            dyadic["quality_flag"]
            .fillna("")
            .astype(str)
            .str.contains("exact_duplicate_source_record", regex=False)
            .sum()
        )
        metrics["flagged_exact_source_duplicates"] = flagged_source_duplicates
        if flagged_source_duplicates:
            issues.append(
                ValidationIssue(
                    "medium",
                    "exact_duplicate_source_record",
                    "exact duplicate raw rows are retained for provenance and excluded from exposure mapping",
                    flagged_source_duplicates,
                    _fraction(flagged_source_duplicates, len(dyadic)),
                )
            )
        if duplicate_events:
            issues.append(
                ValidationIssue("high", "duplicate_event_ids", "event_id must be unique", duplicate_events)
            )
        if self_loops:
            issues.append(
                ValidationIssue("high", "self_loops", "animal interaction self-loops are invalid", self_loops)
            )
        if missing_endpoints:
            issues.append(
                ValidationIssue("critical", "missing_endpoints", "dyadic endpoints are required", missing_endpoints)
            )
        if orphan_nodes:
            issues.append(
                ValidationIssue(
                    "high", "orphan_dyadic_nodes", "event endpoints missing from individuals", len(orphan_nodes)
                )
            )
        starts = pd.to_datetime(dyadic["start_time"], errors="coerce")
        ends = pd.to_datetime(dyadic["end_time"], errors="coerce")
        durations = pd.to_numeric(dyadic["duration_seconds"], errors="coerce")
        negative_duration = int((durations < 0).sum())
        zero_duration = int((durations == 0).sum())
        coordinate_durations = (ends - starts).dt.total_seconds()
        duration_mismatch = int(
            (
                durations.notna()
                & coordinate_durations.notna()
                & (durations - coordinate_durations).abs().gt(1.0)
            ).sum()
        )
        metrics.update(
            {
                "dyadic_zero_duration_rows": zero_duration,
                "dyadic_duration_coordinate_mismatch": duration_mismatch,
            }
        )
        if negative_duration:
            issues.append(
                ValidationIssue("critical", "negative_duration", "duration_seconds cannot be negative", negative_duration)
            )
        if zero_duration:
            issues.append(
                ValidationIssue(
                    "medium",
                    "zero_duration",
                    "zero-duration observations are retained but excluded from exposure mapping",
                    zero_duration,
                    _fraction(zero_duration, len(dyadic)),
                )
            )
        if duration_mismatch:
            issues.append(
                ValidationIssue(
                    "medium",
                    "duration_coordinate_mismatch",
                    "reported duration differs from start/end interval by more than one second",
                    duration_mismatch,
                    _fraction(duration_mismatch, len(dyadic)),
                )
            )
        reversed_time = int(((starts.notna() & ends.notna()) & (ends < starts)).sum())
        missing_temporal = int((starts.isna() | ends.isna()).sum())
        metrics["dyadic_missing_temporal_rows"] = missing_temporal
        if reversed_time:
            issues.append(
                ValidationIssue("critical", "reversed_time", "end_time precedes start_time", reversed_time)
            )
        if missing_temporal:
            severity = "high" if dataset.metadata.has_temporal_order else "medium"
            issues.append(
                ValidationIssue(
                    severity,
                    "missing_temporal_coordinates",
                    "rows without time cannot drive temporal simulation",
                    missing_temporal,
                    _fraction(missing_temporal, len(dyadic)),
                )
            )
        non_finite_values = int(
            (~np.isfinite(pd.to_numeric(dyadic["measurement_value"], errors="coerce"))).sum()
        )
        if non_finite_values:
            issues.append(
                ValidationIssue(
                    "medium",
                    "non_finite_measurement",
                    "measurement_value contains missing or non-finite values",
                    non_finite_values,
                    _fraction(non_finite_values, len(dyadic)),
                )
            )

    groups = dataset.group_events
    memberships = dataset.group_memberships
    if len(groups) or len(memberships):
        duplicate_groups = int(groups["group_event_id"].astype("string").duplicated().sum())
        duplicate_memberships = int(
            memberships[["group_event_id", "node_id"]].astype("string").duplicated().sum()
        )
        orphan_group_ids = set(memberships["group_event_id"].dropna().astype(str)) - set(
            groups["group_event_id"].dropna().astype(str)
        )
        orphan_members = set(memberships["node_id"].dropna().astype(str)) - known_nodes
        metrics.update(
            {
                "duplicate_group_event_ids": duplicate_groups,
                "duplicate_group_memberships": duplicate_memberships,
                "orphan_group_ids": len(orphan_group_ids),
                "orphan_group_members": len(orphan_members),
            }
        )
        if len(memberships):
            group_sizes = memberships.groupby("group_event_id", observed=True).size()
            metrics.update(
                {
                    "group_size_min": int(group_sizes.min()),
                    "group_size_median": float(group_sizes.median()),
                    "group_size_max": int(group_sizes.max()),
                    "singleton_group_fraction": float(group_sizes.eq(1).mean()),
                }
            )
        if duplicate_groups:
            issues.append(
                ValidationIssue("high", "duplicate_group_ids", "group_event_id must be unique", duplicate_groups)
            )
        if duplicate_memberships:
            issues.append(
                ValidationIssue(
                    "high",
                    "duplicate_group_membership",
                    "one node may appear only once per group event",
                    duplicate_memberships,
                )
            )
        if orphan_group_ids:
            issues.append(
                ValidationIssue(
                    "critical", "orphan_group_ids", "memberships reference missing groups", len(orphan_group_ids)
                )
            )
        if orphan_members:
            issues.append(
                ValidationIssue(
                    "high", "orphan_group_members", "memberships reference missing nodes", len(orphan_members)
                )
            )
        if len(groups):
            starts = pd.to_datetime(groups["start_time"], errors="coerce")
            ends = pd.to_datetime(groups["end_time"], errors="coerce")
            missing_group_time = int((starts.isna() | ends.isna()).sum())
            reversed_group_time = int(((starts.notna() & ends.notna()) & (ends < starts)).sum())
            group_durations = pd.to_numeric(groups["duration_seconds"], errors="coerce")
            negative_group_duration = int((group_durations < 0).sum())
            zero_group_duration = int((group_durations == 0).sum())
            metrics["group_missing_temporal_rows"] = missing_group_time
            if missing_group_time:
                issues.append(
                    ValidationIssue(
                        "high", "missing_group_time", "group events require start and end times", missing_group_time
                    )
                )
            if reversed_group_time:
                issues.append(
                    ValidationIssue(
                        "critical", "reversed_group_time", "group end precedes start", reversed_group_time
                    )
                )
            if negative_group_duration:
                issues.append(
                    ValidationIssue(
                        "critical",
                        "negative_group_duration",
                        "group duration_seconds cannot be negative",
                        negative_group_duration,
                    )
                )
            if zero_group_duration:
                issues.append(
                    ValidationIssue(
                        "medium",
                        "zero_group_duration",
                        "zero-duration groups are retained but excluded from exposure mapping",
                        zero_group_duration,
                        _fraction(zero_group_duration, len(groups)),
                    )
                )

    return ValidationReport(dataset.metadata.dataset_id, issues, metrics)


def require_temporal_capability(dataset: CanonicalDataset) -> None:
    if not dataset.metadata.has_temporal_order:
        raise ValueError(
            f"{dataset.metadata.dataset_id} has no recoverable temporal order and cannot drive temporal simulation"
        )
    temporal_rows = int(
        dataset.dyadic_events[["start_time", "end_time"]].notna().all(axis=1).sum()
        + dataset.group_events[["start_time", "end_time"]].notna().all(axis=1).sum()
    )
    if temporal_rows == 0:
        raise ValueError(f"{dataset.metadata.dataset_id} contains no temporally located events")
