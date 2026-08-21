from __future__ import annotations

from collections.abc import Iterable
import hashlib

import pandas as pd


def stable_hash_order(values: Iterable[str], *key_parts: object) -> list[str]:
    """Return a deterministic pseudorandom ordering without global RNG state."""

    prefix = "\x1f".join(str(part) for part in key_parts)

    def digest(value: str) -> bytes:
        return hashlib.sha256(f"{prefix}\x1f{value}".encode("utf-8")).digest()

    return sorted((str(value) for value in values), key=lambda value: (digest(value), value))


def pairwise_rank_stability(
    frame: pd.DataFrame,
    *,
    context_columns: list[str],
    item_column: str,
    value_column: str,
    top_k: int,
) -> pd.DataFrame:
    """Compare every context pair on their shared candidate population."""

    if top_k < 1:
        raise ValueError("top_k must be positive")
    contexts = list(frame.groupby(context_columns, observed=True, sort=True))
    rows: list[dict[str, object]] = []
    for left_index, (left_key, left) in enumerate(contexts):
        left_key = left_key if isinstance(left_key, tuple) else (left_key,)
        left_values = left.set_index(item_column)[value_column].astype(float)
        for right_key, right in contexts[left_index + 1 :]:
            right_key = right_key if isinstance(right_key, tuple) else (right_key,)
            right_values = right.set_index(item_column)[value_column].astype(float)
            shared = left_values.index.intersection(right_values.index)
            if len(shared) < 2:
                continue
            left_shared = left_values.loc[shared]
            right_shared = right_values.loc[shared]
            effective_k = min(top_k, len(shared))
            left_top = set(left_shared.nlargest(effective_k).index)
            right_top = set(right_shared.nlargest(effective_k).index)
            intersection = len(left_top & right_top)
            union = len(left_top | right_top)
            left_oracle_value = float(left_shared.loc[list(left_top)].mean())
            right_oracle_value = float(right_shared.loc[list(right_top)].mean())
            right_selected_by_left = float(right_shared.loc[list(left_top)].mean())
            left_selected_by_right = float(left_shared.loc[list(right_top)].mean())

            def retention(transferred: float, oracle: float) -> float:
                return transferred / oracle if oracle > 0 else float("nan")

            left_to_right_retention = retention(
                right_selected_by_left, right_oracle_value
            )
            right_to_left_retention = retention(
                left_selected_by_right, left_oracle_value
            )
            left_ranks = left_shared.rank(method="average")
            right_ranks = right_shared.rank(method="average")
            spearman = (
                float("nan")
                if left_ranks.nunique() < 2 or right_ranks.nunique() < 2
                else float(left_ranks.corr(right_ranks, method="pearson"))
            )
            row: dict[str, object] = {
                "shared_candidates": len(shared),
                "spearman": spearman,
                "top_k": effective_k,
                "top_k_overlap_count": intersection,
                "top_k_overlap_fraction": intersection / effective_k,
                "top_k_jaccard": intersection / union,
                "left_to_right_value_retention": left_to_right_retention,
                "right_to_left_value_retention": right_to_left_retention,
                "mean_top_k_value_retention": (
                    left_to_right_retention + right_to_left_retention
                )
                / 2,
                "mean_absolute_top_k_regret": (
                    (right_oracle_value - right_selected_by_left)
                    + (left_oracle_value - left_selected_by_right)
                )
                / 2,
            }
            for name, value in zip(context_columns, left_key):
                row[f"left_{name}"] = value
            for name, value in zip(context_columns, right_key):
                row[f"right_{name}"] = value
            rows.append(row)
    return pd.DataFrame(rows)
