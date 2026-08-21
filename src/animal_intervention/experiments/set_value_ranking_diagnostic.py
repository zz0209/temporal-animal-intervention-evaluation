from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import yaml

from animal_intervention.models import DeepSetValueModel
from .set_value_model import (
    DISPLAY_NAMES,
    SET_KEY,
    _anchor_validation_mask,
    _cpu_device_consistency,
    _member_tensors,
    _prepare,
    _sha256,
    _standardize,
)


def _git_value(arguments: list[str]) -> str | None:
    result = subprocess.run(
        ["git", *arguments], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or None


def _pair_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    family_contexts = frame.groupby("system_family", observed=True)["context_id"].nunique()
    for context_id, group in frame.groupby("context_id", observed=True, sort=True):
        indices = group.index.to_numpy(int)
        values = group["mean_set_value"].to_numpy(float)
        errors = group["set_value_se"].to_numpy(float)
        left, right = np.triu_indices(len(group), 1)
        differences = values[left] - values[right]
        nonzero = ~np.isclose(differences, 0.0, atol=1e-15)
        left, right, differences = left[nonzero], right[nonzero], differences[nonzero]
        if not len(left):
            continue
        combined_error = np.sqrt(errors[left] ** 2 + errors[right] ** 2)
        confidence = np.abs(differences) / np.sqrt(differences ** 2 + combined_error ** 2)
        confidence_sum = float(confidence.sum())
        if confidence_sum <= 0:
            continue
        family = str(group["system_family"].iloc[0])
        weights = confidence / confidence_sum / float(family_contexts.loc[family])
        for first, second, difference, confidence_value, weight in zip(
            left, right, differences, confidence, weights
        ):
            rows.append(
                {
                    "context_id": context_id,
                    "system_family": family,
                    "left_index": int(indices[first]),
                    "right_index": int(indices[second]),
                    "target": float(difference > 0),
                    "confidence": float(confidence_value),
                    "weight": float(weight),
                }
            )
    pairs = pd.DataFrame(rows)
    if pairs.empty:
        return pairs
    pairs["weight"] = pairs["weight"] / pairs["weight"].mean()
    return pairs


def _chronological_test_mask(frame: pd.DataFrame, fraction: float) -> np.ndarray:
    frame = frame.copy()
    frame["anchor_time"] = pd.to_datetime(frame["anchor_time"], format="mixed")
    anchor_table = (
        frame[["dataset_id", "network_id", "anchor_id", "anchor_time"]]
        .drop_duplicates()
        .sort_values(["anchor_time", "dataset_id", "network_id", "anchor_id"])
    )
    if len(anchor_table) < 3:
        return np.zeros(len(frame), dtype=bool)
    count = max(1, int(math.ceil(len(anchor_table) * fraction)))
    count = min(count, len(anchor_table) - 2)
    test_units = set(
        anchor_table.tail(count)[["dataset_id", "network_id", "anchor_id"]]
        .astype(str)
        .agg("||".join, axis=1)
    )
    units = frame[["dataset_id", "network_id", "anchor_id"]].astype(str).agg("||".join, axis=1)
    return units.isin(test_units).to_numpy()


def _stratified_context_limit(
    labels: pd.DataFrame, members: pd.DataFrame, maximum: int | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if maximum is None:
        return labels, members
    selected: list[str] = []
    for _, family in labels.groupby("system_family", observed=True, sort=True):
        context_table = family[
            ["context_id", "dataset_id", "network_id", "anchor_id", "anchor_time"]
        ].drop_duplicates()
        anchor_columns = ["dataset_id", "network_id", "anchor_id", "anchor_time"]
        groups = [
            group.sort_values("context_id")["context_id"].tolist()
            for _, group in context_table.groupby(anchor_columns, observed=True, sort=True)
        ]
        family_selected: list[str] = []
        offset = 0
        while len(family_selected) < maximum:
            added = False
            for group in groups:
                if offset < len(group) and len(family_selected) < maximum:
                    family_selected.append(group[offset])
                    added = True
            if not added:
                break
            offset += 1
        selected.extend(family_selected)
    keep = set(selected)
    limited_labels = labels.loc[labels["context_id"].isin(keep)].copy().reset_index(drop=True)
    limited_members = members.merge(
        limited_labels[SET_KEY].drop_duplicates(), on=SET_KEY, how="inner", validate="many_to_one"
    )
    return limited_labels, limited_members


def _train_pairwise_model(
    labels: pd.DataFrame,
    member_values: np.ndarray,
    member_mask: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    context_features: list[str],
    member_feature_count: int,
    settings: dict[str, Any],
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any], dict[str, torch.Tensor], pd.DataFrame]:
    train = labels.loc[train_mask].copy()
    test = labels.loc[test_mask].copy()
    validation_local = _anchor_validation_mask(
        train,
        float(settings["validation_anchor_fraction"]),
        int(settings["minimum_validation_anchors_per_family"]),
        seed,
    )
    validation_indices = set(train.index[validation_local])
    fit_indices = set(train.index[~validation_local])
    fit_pairs = _pair_table(labels.loc[sorted(fit_indices)])
    validation_pairs = _pair_table(labels.loc[sorted(validation_indices)])
    if fit_pairs.empty or validation_pairs.empty:
        raise ValueError("pairwise fitting or validation partition has no ordered pairs")
    fit_context = labels.loc[sorted(fit_indices), context_features].to_numpy(np.float32)
    validation_context = labels.loc[sorted(validation_indices), context_features].to_numpy(np.float32)
    test_context = test[context_features].to_numpy(np.float32)
    fit_context, validation_context, test_context = _standardize(
        fit_context, validation_context, test_context
    )
    context_values = np.zeros((len(labels), len(context_features)), dtype=np.float32)
    context_values[sorted(fit_indices)] = fit_context
    context_values[sorted(validation_indices)] = validation_context
    context_values[test.index] = test_context
    pair_dataset = TensorDataset(
        torch.from_numpy(fit_pairs["left_index"].to_numpy(np.int64)),
        torch.from_numpy(fit_pairs["right_index"].to_numpy(np.int64)),
        torch.from_numpy(fit_pairs["target"].to_numpy(np.float32)),
        torch.from_numpy(fit_pairs["weight"].to_numpy(np.float32)),
    )
    loader = DataLoader(
        pair_dataset,
        batch_size=int(settings["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    members_device = torch.from_numpy(member_values).to(device)
    masks_device = torch.from_numpy(member_mask).to(device)
    contexts_device = torch.from_numpy(context_values).to(device)
    model = DeepSetValueModel(
        member_feature_count,
        len(context_features),
        int(settings["hidden_features"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    loss_function = nn.BCEWithLogitsLoss(reduction="none")
    validation_left = torch.from_numpy(validation_pairs["left_index"].to_numpy(np.int64)).to(device)
    validation_right = torch.from_numpy(validation_pairs["right_index"].to_numpy(np.int64)).to(device)
    validation_target = torch.from_numpy(validation_pairs["target"].to_numpy(np.float32)).to(device)
    validation_weight = torch.from_numpy(validation_pairs["weight"].to_numpy(np.float32)).to(device)
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    for epoch in range(int(settings["maximum_epochs"])):
        model.train()
        numerator = denominator = 0.0
        for left, right, target, weight in loader:
            left, right = left.to(device), right.to(device)
            target, weight = target.to(device), weight.to(device)
            optimizer.zero_grad(set_to_none=True)
            left_score = model(members_device[left], masks_device[left], contexts_device[left])
            right_score = model(members_device[right], masks_device[right], contexts_device[right])
            losses = loss_function(left_score - right_score, target)
            loss = (losses * weight).sum() / weight.sum()
            loss.backward()
            optimizer.step()
            numerator += float((losses.detach() * weight).sum().cpu())
            denominator += float(weight.sum().cpu())
        model.eval()
        with torch.no_grad():
            left_score = model(
                members_device[validation_left], masks_device[validation_left], contexts_device[validation_left]
            )
            right_score = model(
                members_device[validation_right], masks_device[validation_right], contexts_device[validation_right]
            )
            losses = loss_function(left_score - right_score, validation_target)
            validation_loss = float((losses * validation_weight).sum().div(validation_weight.sum()).cpu())
        history.append(
            {"epoch": epoch + 1, "train_pairwise_loss": numerator / denominator, "validation_pairwise_loss": validation_loss}
        )
        if validation_loss < best_loss - 1e-8:
            best_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(settings["early_stopping_patience"]):
                break
    if best_state is None:
        raise RuntimeError("pairwise training produced no model state")
    model.load_state_dict(best_state)
    model.eval()
    test_indices = test.index.to_numpy(np.int64)
    predictions = []
    with torch.no_grad():
        batch_size = int(settings["batch_size"])
        for start in range(0, len(test_indices), batch_size):
            indices = torch.from_numpy(test_indices[start : start + batch_size]).to(device)
            predictions.append(
                model(members_device[indices], masks_device[indices], contexts_device[indices]).cpu().numpy()
            )
    metadata = {
        "fit_sets": len(fit_indices),
        "validation_sets": len(validation_indices),
        "test_sets": len(test_indices),
        "fit_pairs": len(fit_pairs),
        "validation_pairs": len(validation_pairs),
        "best_validation_pairwise_loss": best_loss,
        "epochs_completed": len(history),
        "best_epoch": int(np.argmin([row["validation_pairwise_loss"] for row in history]) + 1),
    }
    pair_summary = pd.DataFrame(
        [
            {
                "partition": "fit",
                "pairs": len(fit_pairs),
                "median_confidence": float(fit_pairs["confidence"].median()),
                "mean_confidence": float(fit_pairs["confidence"].mean()),
            },
            {
                "partition": "validation",
                "pairs": len(validation_pairs),
                "median_confidence": float(validation_pairs["confidence"].median()),
                "mean_confidence": float(validation_pairs["confidence"].mean()),
            },
        ]
    )
    return np.concatenate(predictions), pd.DataFrame(history), metadata, best_state, pair_summary


def _decision_table(predictions: pd.DataFrame, prediction_column: str) -> pd.DataFrame:
    rows = []
    for context_id, group in predictions.groupby("context_id", observed=True, sort=True):
        chosen = group.sort_values([prediction_column, "set_signature"], ascending=[False, True]).iloc[0]
        oracle = group.sort_values(["mean_set_value", "set_signature"], ascending=[False, True]).iloc[0]
        fusion = group.loc[
            group["source_methods"].astype(str).str.split("|").map(
                lambda items: "stable_plus_tracing_base" in items
            )
        ].sort_values("set_signature").iloc[0]
        score_min = float(group[prediction_column].min())
        score_max = float(group[prediction_column].max())
        score_range = score_max - score_min
        normalized_margin = (
            float(chosen[prediction_column] - fusion[prediction_column]) / score_range
            if score_range > 0
            else 0.0
        )
        rows.append(
            {
                "context_id": context_id,
                "system_family": str(group["system_family"].iloc[0]),
                "dataset_id": str(group["dataset_id"].iloc[0]),
                "network_id": str(group["network_id"].iloc[0]),
                "anchor_id": str(group["anchor_id"].iloc[0]),
                "reproducible": bool(group["reproducible"].iloc[0]),
                "ranking_value": float(chosen["mean_set_value"]),
                "reference_value": float(fusion["mean_set_value"]),
                "sampled_oracle_value": float(oracle["mean_set_value"]),
                "gain": float(chosen["mean_set_value"] - fusion["mean_set_value"]),
                "ranking_regret": float(oracle["mean_set_value"] - chosen["mean_set_value"]),
                "reference_regret": float(oracle["mean_set_value"] - fusion["mean_set_value"]),
                "ranking_score": float(chosen[prediction_column]),
                "reference_score": float(fusion[prediction_column]),
                "score_min": score_min,
                "score_max": score_max,
                "normalized_margin": normalized_margin,
                "selected_set_signature": str(chosen["set_signature"]),
                "reference_set_signature": str(fusion["set_signature"]),
            }
        )
    return pd.DataFrame(rows)


def _paired_summary(
    decisions: pd.DataFrame, repetitions: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    family = decisions.groupby("system_family", observed=True).agg(
        contexts=("context_id", "size"),
        mean_gain=("gain", "mean"),
        mean_ranking_value=("ranking_value", "mean"),
        mean_reference_value=("reference_value", "mean"),
        mean_ranking_regret=("ranking_regret", "mean"),
        mean_reference_regret=("reference_regret", "mean"),
    ).reset_index()
    anchor = decisions.groupby(
        ["system_family", "dataset_id", "network_id", "anchor_id"],
        observed=True,
    )["gain"].mean().reset_index()
    rng = np.random.default_rng(seed)
    families = sorted(anchor["system_family"].unique())
    arrays = {
        family_name: group["gain"].to_numpy(float)
        for family_name, group in anchor.groupby("system_family", observed=True, sort=True)
    }
    samples = np.empty(repetitions, dtype=float)
    for repetition in range(repetitions):
        values = []
        for family_name in rng.choice(families, size=len(families), replace=True):
            family_values = arrays[str(family_name)]
            sampled = rng.integers(0, len(family_values), size=len(family_values))
            values.append(float(family_values[sampled].mean()))
        samples[repetition] = float(np.mean(values))
    overall = pd.DataFrame(
        [
            {
                "families": len(families),
                "contexts": len(decisions),
                "family_equal_mean_gain": float(family["mean_gain"].mean()),
                "gain_ci_low": float(np.quantile(samples, 0.025)),
                "gain_ci_high": float(np.quantile(samples, 0.975)),
                "bootstrap_probability_positive": float((samples > 0).mean()),
                "families_with_positive_gain": int((family["mean_gain"] > 0).sum()),
                "family_equal_mean_ranking_regret": float(family["mean_ranking_regret"].mean()),
                "family_equal_mean_reference_regret": float(family["mean_reference_regret"].mean()),
            }
        ]
    )
    return family, overall


def _plot(
    loso_family: pd.DataFrame,
    loso_overall: pd.DataFrame,
    within_overall: pd.DataFrame,
    loso_decisions: pd.DataFrame,
    mse_summary: pd.DataFrame,
    histories: pd.DataFrame,
    report_dir: Path,
) -> None:
    comparison = mse_summary.loc[mse_summary["subset"].eq("all")].copy()
    comparison = comparison.loc[comparison["method"].isin(["stable_watchlist", "contact_to_detected", "ridge", "deep_sets"])]
    comparison = comparison[["method", "family_equal_gain_over_stable_plus_tracing", "gain_ci_low", "gain_ci_high"]]
    ranking = pd.DataFrame(
        [{
            "method": "ranking_deep_sets",
            "family_equal_gain_over_stable_plus_tracing": float(loso_overall["family_equal_mean_gain"].iloc[0]),
            "gain_ci_low": float(loso_overall["gain_ci_low"].iloc[0]),
            "gain_ci_high": float(loso_overall["gain_ci_high"].iloc[0]),
        }]
    )
    comparison = pd.concat([comparison, ranking], ignore_index=True)
    order = ["stable_watchlist", "contact_to_detected", "ridge", "deep_sets", "ranking_deep_sets"]
    comparison = comparison.set_index("method").loc[order]
    values = 100 * comparison["family_equal_gain_over_stable_plus_tracing"].to_numpy(float)
    low = values - 100 * comparison["gain_ci_low"].to_numpy(float)
    high = 100 * comparison["gain_ci_high"].to_numpy(float) - values
    fig, axis = plt.subplots(figsize=(10.5, 6.2))
    y = np.arange(len(comparison))
    colors = ["#9E9E9E", "#F58518", "#72B7B2", "#7A9CC6", "#4C78A8"]
    for index, (value, lower, upper, color) in enumerate(zip(values, low, high, colors)):
        axis.errorbar(value, index, xerr=np.array([[lower], [upper]]), fmt="o", color=color, ecolor=color, capsize=4)
    axis.axvline(0, color="#444444", linestyle="--", linewidth=1)
    axis.set_yticks(y, [item.replace("_", " ") for item in order])
    axis.set_xlabel("Gain over stable + tracing (attack-rate percentage points)")
    fig.suptitle("Unseen-system comparison of set-selection models", fontsize=18, fontweight="bold", y=0.98)
    fig.text(0.5, 0.915, "Family-equal means with hierarchical family-to-anchor 95% intervals", ha="center", color="#555555")
    axis.grid(axis="x", alpha=0.18)
    fig.subplots_adjust(left=0.24, right=0.98, top=0.84, bottom=0.14)
    fig.savefig(report_dir / "loso_model_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    frame = loso_family[["system_family", "mean_gain"]].copy()
    fig, axis = plt.subplots(figsize=(10.5, 6.2))
    names = [DISPLAY_NAMES.get(item, item.replace("_", " ")) for item in frame["system_family"]]
    axis.barh(names, 100 * frame["mean_gain"], color="#4C78A8")
    axis.axvline(0, color="#444444", linestyle="--", linewidth=1)
    axis.set_xlabel("Ranking Deep Sets gain over stable + tracing\n(attack-rate percentage points)")
    fig.suptitle("Ranking-objective transfer by held-out animal system", fontsize=18, fontweight="bold", y=0.98)
    fig.text(0.5, 0.915, "Positive values mean the learned ranking selected a more effective candidate set", ha="center", color="#555555")
    axis.grid(axis="x", alpha=0.18)
    fig.subplots_adjust(left=0.28, right=0.98, top=0.84, bottom=0.16)
    fig.savefig(report_dir / "ranking_transfer_by_family.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.5, 5.8))
    labels = ["Within system\n(later anchors)", "Unseen system\n(LOSO)"]
    estimates = [float(within_overall["family_equal_mean_gain"].iloc[0]), float(loso_overall["family_equal_mean_gain"].iloc[0])]
    lows = [float(within_overall["gain_ci_low"].iloc[0]), float(loso_overall["gain_ci_low"].iloc[0])]
    highs = [float(within_overall["gain_ci_high"].iloc[0]), float(loso_overall["gain_ci_high"].iloc[0])]
    x = np.arange(2)
    axis.errorbar(100 * np.array(estimates), x, xerr=np.vstack([100 * (np.array(estimates) - np.array(lows)), 100 * (np.array(highs) - np.array(estimates))]), fmt="o", color="#4C78A8", ecolor="#9ECAE1", capsize=5)
    axis.axvline(0, color="#444444", linestyle="--", linewidth=1)
    axis.set_yticks(x, labels)
    axis.set_xlabel("Ranking Deep Sets gain over stable + tracing\n(attack-rate percentage points)")
    fig.suptitle("Failure attribution: temporal prediction versus transfer", fontsize=17, fontweight="bold", y=0.98)
    fig.text(0.5, 0.91, "Within-system tests use chronologically later complete anchors", ha="center", color="#555555")
    axis.grid(axis="x", alpha=0.18)
    fig.subplots_adjust(left=0.26, right=0.98, top=0.80, bottom=0.17)
    fig.savefig(report_dir / "failure_attribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 6))
    for label, group in histories.groupby("fold_label", observed=True, sort=True):
        axis.plot(group["epoch"], group["validation_pairwise_loss"], alpha=0.8, label=label)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Weighted validation pairwise loss")
    axis.set_yscale("log")
    fig.suptitle("Pairwise-ranking training convergence", fontsize=18, fontweight="bold", y=0.98)
    fig.text(0.5, 0.915, "Validation uses complete anchors from training systems only", ha="center", color="#555555")
    axis.grid(alpha=0.18)
    axis.legend(frameon=False, fontsize=8, ncol=2)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.82, bottom=0.14)
    fig.savefig(report_dir / "ranking_training_convergence.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    outcome_rows = []
    tolerance = 1e-12
    for family_name, group in loso_decisions.groupby("system_family", observed=True, sort=True):
        same = group["selected_set_signature"].eq(group["reference_set_signature"])
        categories = {
            "Same candidate set": same,
            "Different set, same outcome": ~same & group["gain"].abs().le(tolerance),
            "Ranking model better": group["gain"].gt(tolerance),
            "Ranking model worse": group["gain"].lt(-tolerance),
        }
        for category, mask in categories.items():
            outcome_rows.append(
                {
                    "system_family": family_name,
                    "category": category,
                    "fraction": float(mask.mean()),
                }
            )
    outcomes = pd.DataFrame(outcome_rows)
    categories = [
        "Same candidate set",
        "Different set, same outcome",
        "Ranking model better",
        "Ranking model worse",
    ]
    colors = ["#666666", "#D9D9D9", "#4C78A8", "#F58518"]
    family_order = list(loso_family["system_family"])
    fig, axis = plt.subplots(figsize=(11, 6.2))
    left = np.zeros(len(family_order))
    for category, color in zip(categories, colors):
        values = (
            outcomes.loc[outcomes["category"].eq(category)]
            .set_index("system_family")
            .reindex(family_order)["fraction"]
            .to_numpy(float)
        )
        axis.barh(
            [DISPLAY_NAMES.get(item, item.replace("_", " ")) for item in family_order],
            values,
            left=left,
            color=color,
            label=category,
        )
        left += values
    axis.set_xlim(0, 1)
    axis.set_xlabel("Fraction of decision contexts")
    fig.suptitle("Outcome of changing the stable + tracing decision", fontsize=18, fontweight="bold", y=0.98)
    fig.text(0.5, 0.915, "The model often changes the set without changing outcome; harmful changes outweigh helpful changes in mean value", ha="center", color="#555555")
    axis.grid(axis="x", alpha=0.18)
    axis.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.30))
    fig.subplots_adjust(left=0.27, right=0.98, top=0.82, bottom=0.25)
    fig.savefig(report_dir / "decision_outcome_composition.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(config_path: Path, profile_name: str) -> tuple[Path, Path]:
    started = time.perf_counter()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment_id = str(config["experiment"]["id"])
    results_dir = Path(config["outputs"]["results_root"]) / experiment_id / profile_name
    report_dir = Path(config["outputs"]["report_root"]) / experiment_id / profile_name
    model_dir = results_dir / "models"
    for directory in (results_dir, report_dir, model_dir):
        directory.mkdir(parents=True, exist_ok=True)
    data_paths = {key: Path(value) for key, value in config["data"].items()}
    precision_audit = json.loads(data_paths["precision_audit"].read_text(encoding="utf-8"))
    if precision_audit.get("status") != "pass" or precision_audit.get("scientific_gate", {}).get("status") != "pass":
        raise ValueError("EXP-013 precision gate must pass before ranking diagnostics")
    labels = pd.read_csv(data_paths["labels"], dtype={"initial_infected": str})
    members = pd.read_csv(data_paths["members"], dtype={"initial_infected": str, "candidate_id": str})
    reliability = pd.read_csv(data_paths["reliability"], dtype={"initial_infected": str})
    profile = dict(config["profiles"][profile_name])
    maximum_contexts = profile.get("maximum_contexts_per_family")
    prepare_profile = dict(profile)
    prepare_profile["maximum_contexts_per_family"] = None
    labels, members = _prepare(labels, members, reliability, prepare_profile)
    labels, members = _stratified_context_limit(labels, members, maximum_contexts)
    labels = labels.reset_index(drop=True)
    member_features = list(config["features"]["member"])
    context_features = list(config["features"]["context"])
    member_values, member_mask = _member_tensors(labels, members, member_features)
    settings = dict(config["model"])
    settings.update({key: value for key, value in profile.items() if key in settings})
    repetitions = int(profile.get("bootstrap_repetitions", config["evaluation"]["bootstrap_repetitions"]))
    seed = int(config["evaluation"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    deterministic = bool(config["execution"]["deterministic"])
    torch.use_deterministic_algorithms(deterministic)
    preferred = str(config["execution"]["preferred_device"])
    device = torch.device("cuda" if preferred == "cuda" and torch.cuda.is_available() else "cpu")
    consistency_ok, consistency_difference = _cpu_device_consistency(
        len(member_features), len(context_features), int(settings["hidden_features"]), seed, device
    )
    loso_predictions = []
    within_predictions = []
    histories = []
    metadata_rows = []
    pair_summaries = []
    families = sorted(labels["system_family"].unique())
    for fold_index, held_out in enumerate(tqdm(families, desc="LOSO ranking folds", unit="fold")):
        train_mask = labels["system_family"].ne(held_out).to_numpy()
        test_mask = labels["system_family"].eq(held_out).to_numpy()
        prediction, history, metadata, state, pair_summary = _train_pairwise_model(
            labels, member_values, member_mask, train_mask, test_mask, context_features,
            len(member_features), settings, seed + fold_index, device,
        )
        test = labels.loc[test_mask].copy()
        test["ranking_prediction"] = prediction
        loso_predictions.append(test)
        fold_label = f"LOSO: {DISPLAY_NAMES.get(held_out, held_out)}"
        history["fold_label"] = fold_label
        histories.append(history)
        metadata.update({"evaluation": "loso", "held_out_family": held_out})
        metadata_rows.append(metadata)
        pair_summary["fold_label"] = fold_label
        pair_summaries.append(pair_summary)
        torch.save({"state_dict": state, "metadata": metadata}, model_dir / f"loso_{held_out}.pt")
    assessable_within = []
    for fold_index, family in enumerate(tqdm(families, desc="Within-system ranking folds", unit="fold")):
        family_mask = labels["system_family"].eq(family).to_numpy()
        family_frame = labels.loc[family_mask]
        local_test = _chronological_test_mask(
            family_frame, float(config["evaluation"]["within_system_test_anchor_fraction"])
        )
        if not local_test.any():
            continue
        family_indices = family_frame.index.to_numpy()
        test_indices = set(family_indices[local_test])
        train_indices = set(family_indices[~local_test])
        train_mask = labels.index.isin(train_indices)
        test_mask = labels.index.isin(test_indices)
        prediction, history, metadata, state, pair_summary = _train_pairwise_model(
            labels, member_values, member_mask, train_mask, test_mask, context_features,
            len(member_features), settings, seed + 100 + fold_index, device,
        )
        test = labels.loc[test_mask].copy()
        test["ranking_prediction"] = prediction
        within_predictions.append(test)
        assessable_within.append(family)
        fold_label = f"Within: {DISPLAY_NAMES.get(family, family)}"
        history["fold_label"] = fold_label
        histories.append(history)
        metadata.update({"evaluation": "within_system", "held_out_family": family})
        metadata_rows.append(metadata)
        pair_summary["fold_label"] = fold_label
        pair_summaries.append(pair_summary)
        torch.save({"state_dict": state, "metadata": metadata}, model_dir / f"within_{family}.pt")
    loso_prediction_frame = pd.concat(loso_predictions, ignore_index=True)
    within_prediction_frame = pd.concat(within_predictions, ignore_index=True)
    history_frame = pd.concat(histories, ignore_index=True)
    pair_summary_frame = pd.concat(pair_summaries, ignore_index=True)
    loso_decisions = _decision_table(loso_prediction_frame, "ranking_prediction")
    within_decisions = _decision_table(within_prediction_frame, "ranking_prediction")
    loso_family, loso_overall = _paired_summary(loso_decisions, repetitions, seed)
    within_family, within_overall = _paired_summary(within_decisions, repetitions, seed + 1)
    loso_pass = bool(
        loso_overall["family_equal_mean_gain"].iloc[0] > 0
        and loso_overall["families_with_positive_gain"].iloc[0] >= int(config["evaluation"]["minimum_loso_positive_families"])
        and loso_overall["bootstrap_probability_positive"].iloc[0] >= float(config["evaluation"]["minimum_loso_bootstrap_probability_positive"])
    )
    within_pass = bool(
        within_overall["family_equal_mean_gain"].iloc[0] > 0
        and within_overall["families_with_positive_gain"].iloc[0] >= int(config["evaluation"]["minimum_within_positive_families"])
        and within_overall["bootstrap_probability_positive"].iloc[0] >= float(config["evaluation"]["minimum_within_bootstrap_probability_positive"])
    )
    interpretation = (
        "ranking_objective_resolves_transfer" if loso_pass
        else "cross_system_transfer_ceiling" if within_pass
        else "current_set_feature_planner_not_supported"
    )
    loso_prediction_frame.to_csv(results_dir / "loso_ranking_predictions.csv.gz", index=False, compression="gzip")
    within_prediction_frame.to_csv(results_dir / "within_system_ranking_predictions.csv.gz", index=False, compression="gzip")
    loso_decisions.to_csv(results_dir / "loso_ranking_decisions.csv.gz", index=False, compression="gzip")
    within_decisions.to_csv(results_dir / "within_system_ranking_decisions.csv.gz", index=False, compression="gzip")
    loso_family.to_csv(results_dir / "loso_family_summary.csv", index=False)
    loso_overall.to_csv(results_dir / "loso_overall_summary.csv", index=False)
    within_family.to_csv(results_dir / "within_system_family_summary.csv", index=False)
    within_overall.to_csv(results_dir / "within_system_overall_summary.csv", index=False)
    history_frame.to_csv(results_dir / "training_history.csv", index=False)
    pair_summary_frame.to_csv(results_dir / "pair_supervision_summary.csv", index=False)
    (results_dir / "fold_metadata.json").write_text(json.dumps(metadata_rows, indent=2), encoding="utf-8")
    checks = {
        "precision_gate_passed": True,
        "four_independent_loso_families": len(families) == 4,
        "chronological_within_system_families": len(assessable_within) == 3,
        "one_loso_prediction_per_set": not loso_prediction_frame.duplicated(SET_KEY).any(),
        "finite_predictions": bool(np.isfinite(loso_prediction_frame["ranking_prediction"]).all() and np.isfinite(within_prediction_frame["ranking_prediction"]).all()),
        "cpu_device_forward_consistency": consistency_ok,
        "nonconstant_loso_scores": bool(loso_prediction_frame.groupby("context_id")["ranking_prediction"].nunique().gt(1).all()),
        "nonnegative_regrets": bool((loso_decisions[["ranking_regret", "reference_regret"]] >= -1e-12).all().all()),
        "positive_pair_weights": bool(pair_summary_frame["pairs"].gt(0).all()),
    }
    audit = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "scientific_gate": {
            "status": "pass" if loso_pass else "fail",
            "loso_pass": loso_pass,
            "within_system_pass": within_pass,
            "interpretation": interpretation,
            "loso": loso_overall.iloc[0].to_dict(),
            "within_system": within_overall.iloc[0].to_dict(),
        },
    }
    if audit["status"] != "pass":
        raise ValueError(f"ranking diagnostic artifact audit failed: {audit}")
    (results_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (results_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    mse_summary = pd.read_csv(data_paths["mse_overall_summary"])
    _plot(
        loso_family,
        loso_overall,
        within_overall,
        loso_decisions,
        mse_summary,
        history_frame,
        report_dir,
    )
    manifest = {
        "experiment_id": experiment_id,
        "profile": profile_name,
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "git_commit": _git_value(["rev-parse", "HEAD"]),
        "git_worktree_dirty": bool(_git_value(["status", "--porcelain"])),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor(),
        "numeric_precision": "float32",
        "deterministic_algorithms": deterministic,
        "cpu_device_max_abs_difference": consistency_difference,
        "seed": seed,
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "input_hashes": {key: _sha256(path) for key, path in data_paths.items()},
        "artifact_audit": audit["status"],
        "scientific_gate": audit["scientific_gate"]["status"],
    }
    (results_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (report_dir / "README.md").write_text(
        "# Ranking-objective failure attribution\n\n"
        f"Profile: **{profile_name}**. Artifact audit: **{audit['status']}**. LOSO scientific gate: **{audit['scientific_gate']['status']}**. "
        f"Interpretation: **{interpretation}**.\n\n"
        "This experiment changes only the supervised objective. It does not change epidemic labels, candidate sets, disease parameters, input features, or the stable-plus-tracing reference. "
        "Within-system tests use chronologically later complete anchors; Oxford is excluded from that diagnostic because only one matching anchor exists.\n",
        encoding="utf-8",
    )
    return results_dir, report_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ranking-objective failure attribution")
    parser.add_argument("--config", type=Path, default=Path("configs/EXP-20260816-015_ranking_diagnostic.yaml"))
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    args = parser.parse_args()
    results, reports = run(args.config, args.profile)
    print(f"Results: {results}")
    print(f"Reports: {reports}")


if __name__ == "__main__":
    main()
