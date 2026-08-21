from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
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
from .set_value_precision import OBSERVATION_KEYS


SET_KEY = OBSERVATION_KEYS + ["set_signature"]
DISPLAY_NAMES = {
    "guinea_baboons_sociopatterns": "Guinea baboons",
    "linked_wytham_songbird_family": "Linked Wytham/songbirds",
    "oxford_wildbird_network": "Oxford wild birds",
    "radolfzell_great_tits_ontogeny": "Radolfzell great tits",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_value(arguments: list[str]) -> str | None:
    result = subprocess.run(
        ["git", *arguments], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or None


def _context_id(frame: pd.DataFrame) -> pd.Series:
    return frame[OBSERVATION_KEYS].astype(str).agg("||".join, axis=1)


def _prepare(
    labels: pd.DataFrame,
    members: pd.DataFrame,
    reliability: pd.DataFrame,
    profile: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = labels.copy()
    labels["context_id"] = _context_id(labels)
    reliability = reliability.copy()
    reliability["context_id"] = _context_id(reliability)
    labels = labels.merge(
        reliability[["context_id", "reproducible", "full_mean_spread"]],
        on="context_id",
        how="left",
        validate="many_to_one",
    )
    if labels[["reproducible", "full_mean_spread"]].isna().any().any():
        raise ValueError("some set labels lack reliability metadata")
    maximum = profile.get("maximum_contexts_per_family")
    if maximum is not None:
        selected = []
        for _, group in labels.groupby("system_family", observed=True, sort=True):
            ids = sorted(group["context_id"].unique())[: int(maximum)]
            selected.extend(ids)
        labels = labels.loc[labels["context_id"].isin(selected)].copy()
    labels["centered_set_value"] = labels["mean_set_value"] - labels.groupby(
        "context_id", observed=True
    )["mean_set_value"].transform("mean")
    labels["detected_case_fraction"] = (
        labels["detected_case_count"] / labels["population_size"].clip(lower=1)
    )
    labels["log_population_size"] = np.log1p(labels["population_size"])
    labels["log_budget"] = np.log1p(labels["budget"])
    keep = labels[SET_KEY].drop_duplicates()
    members = members.merge(keep, on=SET_KEY, how="inner", validate="many_to_one")
    return labels.sort_values(SET_KEY).reset_index(drop=True), members


def _balanced_weights(frame: pd.DataFrame) -> np.ndarray:
    family_contexts = frame.groupby("system_family", observed=True)["context_id"].nunique()
    context_sets = frame.groupby("context_id", observed=True)["set_signature"].size()
    weights = np.array(
        [
            1.0 / (family_contexts.loc[family] * context_sets.loc[context])
            for family, context in zip(frame["system_family"], frame["context_id"])
        ],
        dtype=np.float64,
    )
    return weights / weights.mean()


def _anchor_validation_mask(
    train: pd.DataFrame, fraction: float, minimum: int, seed: int
) -> np.ndarray:
    validation: set[str] = set()
    for family, group in train.groupby("system_family", observed=True, sort=True):
        anchors = sorted(
            group[["dataset_id", "network_id", "anchor_id"]]
            .drop_duplicates()
            .astype(str)
            .agg("||".join, axis=1)
        )
        ranked = sorted(
            anchors,
            key=lambda item: hashlib.sha256(f"{seed}|{family}|{item}".encode()).hexdigest(),
        )
        count = min(len(ranked) - 1, max(minimum, int(round(len(ranked) * fraction))))
        if count > 0:
            validation.update(ranked[:count])
    anchor_ids = train[["dataset_id", "network_id", "anchor_id"]].astype(str).agg("||".join, axis=1)
    if not validation:
        ranked = sorted(
            anchor_ids.unique(),
            key=lambda item: hashlib.sha256(f"{seed}|global|{item}".encode()).hexdigest(),
        )
        if len(ranked) < 2:
            raise ValueError("at least two complete anchors are required for fitting and validation")
        validation.add(ranked[0])
    return anchor_ids.isin(validation).to_numpy()


def _standardize(
    train: np.ndarray, *others: np.ndarray
) -> tuple[np.ndarray, ...]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale == 0] = 1.0
    return tuple((array - mean) / scale for array in (train, *others))


def _ridge_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    penalty: float,
) -> np.ndarray:
    x_train = train[features].to_numpy(np.float64)
    x_test = test[features].to_numpy(np.float64)
    x_train, x_test = _standardize(x_train, x_test)
    x_train = np.column_stack([np.ones(len(x_train)), x_train])
    x_test = np.column_stack([np.ones(len(x_test)), x_test])
    weights = _balanced_weights(train)
    regularizer = np.eye(x_train.shape[1]) * penalty
    regularizer[0, 0] = 0.0
    coefficients = np.linalg.solve(
        x_train.T @ (weights[:, None] * x_train) + regularizer,
        x_train.T @ (weights * train["centered_set_value"].to_numpy(float)),
    )
    return x_test @ coefficients


def _member_tensors(
    labels: pd.DataFrame,
    members: pd.DataFrame,
    member_features: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    grouped = {
        tuple(str(item) for item in (key if isinstance(key, tuple) else (key,))): group
        for key, group in members.groupby(SET_KEY, observed=True, sort=False)
    }
    maximum = int(labels["budget"].max())
    values = np.zeros((len(labels), maximum, len(member_features)), dtype=np.float32)
    mask = np.zeros((len(labels), maximum), dtype=bool)
    for row_index, row in enumerate(labels.itertuples(index=False)):
        key = tuple(str(getattr(row, column)) for column in SET_KEY)
        group = grouped.get(key)
        if group is None:
            raise ValueError(f"missing member features for set {key}")
        array = group[member_features].to_numpy(np.float32)
        values[row_index, : len(array)] = array
        mask[row_index, : len(array)] = True
    return values, mask


def _train_deep_sets(
    train: pd.DataFrame,
    test: pd.DataFrame,
    members: pd.DataFrame,
    member_features: list[str],
    context_features: list[str],
    settings: dict[str, Any],
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any], dict[str, torch.Tensor]]:
    validation_mask = _anchor_validation_mask(
        train,
        float(settings["validation_anchor_fraction"]),
        int(settings["minimum_validation_anchors_per_family"]),
        seed,
    )
    fit = train.loc[~validation_mask].reset_index(drop=True)
    validation = train.loc[validation_mask].reset_index(drop=True)
    if fit.empty or validation.empty:
        raise ValueError("anchor-grouped training or validation partition is empty")
    combined = pd.concat([fit, validation, test], ignore_index=True)
    member_values, member_mask = _member_tensors(combined, members, member_features)
    fit_end = len(fit)
    validation_end = fit_end + len(validation)
    context_fit = fit[context_features].to_numpy(np.float32)
    context_validation = validation[context_features].to_numpy(np.float32)
    context_test = test[context_features].to_numpy(np.float32)
    context_fit, context_validation, context_test = _standardize(
        context_fit, context_validation, context_test
    )
    context_values = np.concatenate(
        [context_fit, context_validation, context_test]
    ).astype(np.float32)
    weights = _balanced_weights(fit).astype(np.float32)
    dataset = TensorDataset(
        torch.from_numpy(member_values[:fit_end]),
        torch.from_numpy(member_mask[:fit_end]),
        torch.from_numpy(context_values[:fit_end]),
        torch.from_numpy(fit["centered_set_value"].to_numpy(np.float32)),
        torch.from_numpy(weights),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=int(settings["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    model = DeepSetValueModel(
        len(member_features), len(context_features), int(settings["hidden_features"])
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    loss_function = nn.MSELoss(reduction="none")
    validation_tensors = (
        torch.from_numpy(member_values[fit_end:validation_end]).to(device),
        torch.from_numpy(member_mask[fit_end:validation_end]).to(device),
        torch.from_numpy(context_values[fit_end:validation_end]).to(device),
        torch.from_numpy(validation["centered_set_value"].to_numpy(np.float32)).to(device),
        torch.from_numpy(_balanced_weights(validation).astype(np.float32)).to(device),
    )
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    maximum_epochs = int(settings["maximum_epochs"])
    for epoch in range(maximum_epochs):
        model.train()
        numerator = denominator = 0.0
        for member_batch, mask_batch, context_batch, target_batch, weight_batch in loader:
            member_batch = member_batch.to(device)
            mask_batch = mask_batch.to(device)
            context_batch = context_batch.to(device)
            target_batch = target_batch.to(device)
            weight_batch = weight_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(member_batch, mask_batch, context_batch)
            losses = loss_function(prediction, target_batch)
            loss = (losses * weight_batch).sum() / weight_batch.sum()
            loss.backward()
            optimizer.step()
            numerator += float((losses.detach() * weight_batch).sum().cpu())
            denominator += float(weight_batch.sum().cpu())
        model.eval()
        with torch.no_grad():
            validation_prediction = model(*validation_tensors[:3])
            validation_errors = (validation_prediction - validation_tensors[3]) ** 2
            validation_loss = float(
                ((validation_errors * validation_tensors[4]).sum() / validation_tensors[4].sum()).cpu()
            )
        history.append(
            {"epoch": epoch + 1, "train_mse": numerator / denominator, "validation_mse": validation_loss}
        )
        if validation_loss < best_loss - 1e-10:
            best_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(settings["early_stopping_patience"]):
                break
    if best_state is None:
        raise RuntimeError("training produced no valid model state")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prediction = model(
            torch.from_numpy(member_values[validation_end:]).to(device),
            torch.from_numpy(member_mask[validation_end:]).to(device),
            torch.from_numpy(context_values[validation_end:]).to(device),
        ).cpu().numpy()
    metadata = {
        "fit_sets": len(fit),
        "validation_sets": len(validation),
        "test_sets": len(test),
        "best_validation_mse": best_loss,
        "epochs_completed": len(history),
        "best_epoch": int(np.argmin([row["validation_mse"] for row in history]) + 1),
    }
    return prediction, pd.DataFrame(history), metadata, best_state


def _method_row(group: pd.DataFrame, token: str) -> pd.Series:
    matching = group.loc[
        group["source_methods"].astype(str).str.split("|").map(lambda items: token in items)
    ]
    if matching.empty:
        raise ValueError(f"context lacks required method {token}")
    return matching.sort_values("set_signature").iloc[0]


def _decision_rows(predictions: pd.DataFrame, reference: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for context_id, group in predictions.groupby("context_id", observed=True, sort=True):
        oracle = group.sort_values(
            ["mean_set_value", "set_signature"], ascending=[False, True]
        ).iloc[0]
        random_rows = group.loc[group["source_methods"].str.contains("random_", regex=False)]
        method_rows = {
            "stable_watchlist": _method_row(group, "stable_watchlist_base"),
            "contact_to_detected": _method_row(group, "contact_to_detected_base"),
            "stable_plus_tracing": _method_row(group, reference),
            "ridge": group.sort_values(
                ["ridge_prediction", "set_signature"], ascending=[False, True]
            ).iloc[0],
            "deep_sets": group.sort_values(
                ["deep_sets_prediction", "set_signature"], ascending=[False, True]
            ).iloc[0],
        }
        base = {
            "context_id": context_id,
            "system_family": str(group["system_family"].iloc[0]),
            "dataset_id": str(group["dataset_id"].iloc[0]),
            "network_id": str(group["network_id"].iloc[0]),
            "anchor_id": str(group["anchor_id"].iloc[0]),
            "reproducible": bool(group["reproducible"].iloc[0]),
            "sampled_oracle_value": float(oracle["mean_set_value"]),
            "random_expected_value": float(random_rows["mean_set_value"].mean()),
        }
        for method, row in method_rows.items():
            value = float(row["mean_set_value"])
            base[f"{method}_value"] = value
            base[f"{method}_regret"] = float(oracle["mean_set_value"] - value)
            base[f"{method}_set_signature"] = str(row["set_signature"])
        rows.append(base)
    return pd.DataFrame(rows)


def _summaries(decisions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    methods = ["stable_watchlist", "contact_to_detected", "stable_plus_tracing", "ridge", "deep_sets"]
    family_rows: list[dict[str, Any]] = []
    for family, group in decisions.groupby("system_family", observed=True, sort=True):
        for subset_name, subset in (
            ("all", group), ("precision_qualified", group.loc[group["reproducible"]])
        ):
            if subset.empty:
                continue
            reference = float(subset["stable_plus_tracing_value"].mean())
            for method in methods:
                family_rows.append(
                    {
                        "system_family": family,
                        "subset": subset_name,
                        "method": method,
                        "contexts": len(subset),
                        "mean_value": float(subset[f"{method}_value"].mean()),
                        "mean_regret": float(subset[f"{method}_regret"].mean()),
                        "gain_over_stable_plus_tracing": float(subset[f"{method}_value"].mean() - reference),
                    }
                )
    family = pd.DataFrame(family_rows)
    overall = family.groupby(["subset", "method"], observed=True).agg(
        families=("system_family", "nunique"),
        family_equal_mean_value=("mean_value", "mean"),
        family_equal_mean_regret=("mean_regret", "mean"),
        family_equal_gain_over_stable_plus_tracing=("gain_over_stable_plus_tracing", "mean"),
        families_with_positive_gain=("gain_over_stable_plus_tracing", lambda values: int((values > 0).sum())),
    ).reset_index()
    return family, overall


def _bootstrap_gain_intervals(
    decisions: pd.DataFrame, repetitions: int, seed: int
) -> pd.DataFrame:
    methods = ["stable_watchlist", "contact_to_detected", "stable_plus_tracing", "ridge", "deep_sets"]
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for subset_name, subset in (
        ("all", decisions),
        ("precision_qualified", decisions.loc[decisions["reproducible"]]),
    ):
        families = sorted(subset["system_family"].unique())
        gain_columns = []
        working = subset.copy()
        for method in methods:
            column = f"{method}_gain"
            working[column] = working[f"{method}_value"] - working["stable_plus_tracing_value"]
            gain_columns.append(column)
        anchor_keys = ["system_family", "dataset_id", "anchor_id"]
        if "network_id" in working.columns:
            anchor_keys.insert(2, "network_id")
        anchor_means = working.groupby(
            anchor_keys,
            observed=True,
            sort=True,
        )[gain_columns].mean().reset_index()
        family_arrays = {
            family: group[gain_columns].to_numpy(float)
            for family, group in anchor_means.groupby("system_family", observed=True, sort=True)
        }
        samples = np.empty((repetitions, len(methods)), dtype=float)
        for repetition in range(repetitions):
            family_values = []
            for family in rng.choice(families, size=len(families), replace=True):
                values = family_arrays[str(family)]
                sampled = rng.integers(0, len(values), size=len(values))
                family_values.append(values[sampled].mean(axis=0))
            samples[repetition] = np.mean(family_values, axis=0)
        for method_index, method in enumerate(methods):
            values = samples[:, method_index]
            rows.append(
                {
                    "subset": subset_name,
                    "method": method,
                    "gain_ci_low": float(np.quantile(values, 0.025)),
                    "gain_ci_high": float(np.quantile(values, 0.975)),
                    "bootstrap_probability_positive": float((values > 0).mean()),
                    "bootstrap_repetitions": repetitions,
                }
            )
    return pd.DataFrame(rows)


def _cpu_device_consistency(
    member_features: int,
    context_features: int,
    hidden_features: int,
    seed: int,
    device: torch.device,
) -> tuple[bool, float]:
    if device.type != "cuda":
        return True, 0.0
    generator = torch.Generator().manual_seed(seed)
    cpu_model = DeepSetValueModel(member_features, context_features, hidden_features).eval()
    device_model = DeepSetValueModel(member_features, context_features, hidden_features).to(device).eval()
    device_model.load_state_dict(cpu_model.state_dict())
    members = torch.rand((17, 9, member_features), generator=generator)
    mask = torch.ones((17, 9), dtype=torch.bool)
    mask[:, -2:] = False
    context = torch.rand((17, context_features), generator=generator)
    with torch.no_grad():
        cpu_output = cpu_model(members, mask, context)
        device_output = device_model(
            members.to(device), mask.to(device), context.to(device)
        ).cpu()
    difference = float(torch.max(torch.abs(cpu_output - device_output)))
    return bool(torch.allclose(cpu_output, device_output, rtol=1e-5, atol=1e-6)), difference


def _plot(
    family: pd.DataFrame,
    overall: pd.DataFrame,
    histories: pd.DataFrame,
    report_dir: Path,
) -> None:
    all_rows = family.loc[family["subset"].eq("all")].copy()
    pivot = all_rows.pivot(index="system_family", columns="method", values="gain_over_stable_plus_tracing")
    order = list(pivot.index)
    methods = ["stable_watchlist", "contact_to_detected", "ridge", "deep_sets"]
    colors = ["#9E9E9E", "#F58518", "#72B7B2", "#4C78A8"]
    fig, axis = plt.subplots(figsize=(12, 6.8))
    x = np.arange(len(order))
    width = 0.18
    for offset, method, color in zip(np.arange(len(methods)) - 1.5, methods, colors):
        axis.bar(x + offset * width, 100 * pivot[method], width, label=method.replace("_", " "), color=color)
    axis.axhline(0, color="#444444", linewidth=1, linestyle="--")
    axis.set_xticks(x, [DISPLAY_NAMES.get(item, item.replace("_", " ")) for item in order])
    axis.set_ylabel("Gain over stable + tracing (attack-rate percentage points)")
    fig.suptitle("Strict leave-one-animal-system-out set selection", fontsize=18, fontweight="bold", y=0.98)
    fig.text(0.5, 0.915, "Each animal system is tested only by models trained on the other systems", ha="center", color="#555555")
    axis.grid(axis="y", alpha=0.18)
    axis.legend(frameon=False, ncol=2)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.84, bottom=0.15)
    fig.savefig(report_dir / "loso_family_gain.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 6))
    for family_name, group in histories.groupby("held_out_family", observed=True, sort=True):
        axis.plot(group["epoch"], group["validation_mse"], label=DISPLAY_NAMES.get(family_name, family_name))
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Validation mean squared error")
    axis.set_yscale("log")
    fig.suptitle("Deep Sets training convergence", fontsize=18, fontweight="bold", y=0.98)
    fig.text(0.5, 0.915, "Validation anchors come only from training animal systems", ha="center", color="#555555")
    axis.grid(alpha=0.18)
    axis.legend(frameon=False)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.84, bottom=0.14)
    fig.savefig(report_dir / "training_convergence.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    frame = overall.loc[
        overall["method"].isin(["stable_watchlist", "contact_to_detected", "ridge", "deep_sets"])
    ].copy()
    method_order = ["stable_watchlist", "contact_to_detected", "ridge", "deep_sets"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharey=True)
    for axis, subset_name, subtitle in zip(
        axes,
        ["all", "precision_qualified"],
        ["All decision contexts", "EXP-013 precision-qualified contexts"],
    ):
        panel = frame.loc[frame["subset"].eq(subset_name)].set_index("method").loc[method_order]
        values = 100 * panel["family_equal_gain_over_stable_plus_tracing"].to_numpy(float)
        low = values - 100 * panel["gain_ci_low"].to_numpy(float)
        high = 100 * panel["gain_ci_high"].to_numpy(float) - values
        y = np.arange(len(panel))
        axis.errorbar(values, y, xerr=np.vstack([low, high]), fmt="o", color="#4C78A8", ecolor="#9ECAE1", capsize=4)
        axis.axvline(0, color="#444444", linewidth=1, linestyle="--")
        axis.set_yticks(y, [item.replace("_", " ") for item in method_order])
        axis.set_title(subtitle, fontweight="bold")
        axis.set_xlabel("Gain over stable + tracing\n(attack-rate percentage points)")
        axis.grid(axis="x", alpha=0.18)
    fig.suptitle("Cross-system uncertainty in set-selection gain", fontsize=18, fontweight="bold", y=0.98)
    fig.text(0.5, 0.91, "95% hierarchical bootstrap intervals resample animal-system families, then complete anchors", ha="center", color="#555555")
    fig.subplots_adjust(left=0.18, right=0.98, top=0.80, bottom=0.17, wspace=0.18)
    fig.savefig(report_dir / "overall_gain_uncertainty.png", dpi=180, bbox_inches="tight")
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
    paths = {key: Path(value) for key, value in config["data"].items()}
    precision_audit = json.loads(paths["precision_audit"].read_text(encoding="utf-8"))
    if precision_audit.get("status") != "pass" or precision_audit.get("scientific_gate", {}).get("status") != "pass":
        raise ValueError("EXP-013 precision gate must pass before model training")
    labels = pd.read_csv(paths["labels"], dtype={"initial_infected": str})
    members = pd.read_csv(paths["members"], dtype={"initial_infected": str, "candidate_id": str})
    reliability = pd.read_csv(paths["reliability"], dtype={"initial_infected": str})
    labels, members = _prepare(labels, members, reliability, config["profiles"][profile_name])
    member_features = list(config["features"]["member"])
    context_features = list(config["features"]["context"])
    ridge_features = context_features + list(config["features"]["ridge_set"])
    required = set(member_features)
    if not required.issubset(members.columns):
        raise ValueError(f"missing member features: {sorted(required - set(members.columns))}")
    settings = dict(config["model"])
    settings.update({key: value for key, value in config["profiles"][profile_name].items() if key in settings})
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
        len(member_features),
        len(context_features),
        int(settings["hidden_features"]),
        seed,
        device,
    )
    predictions = []
    histories = []
    fold_metadata = []
    families = sorted(labels["system_family"].unique())
    for fold_index, held_out in enumerate(tqdm(families, desc="LOSO model folds", unit="fold")):
        train = labels.loc[labels["system_family"].ne(held_out)].reset_index(drop=True)
        test = labels.loc[labels["system_family"].eq(held_out)].reset_index(drop=True)
        if held_out in set(train["system_family"]) or set(test["system_family"]) != {held_out}:
            raise AssertionError("held-out family leaked across the outer split")
        test["ridge_prediction"] = _ridge_predict(
            train, test, ridge_features, float(settings["ridge_penalty"])
        )
        deep_prediction, history, metadata, state = _train_deep_sets(
            train,
            test,
            members,
            member_features,
            context_features,
            settings,
            seed + fold_index,
            device,
        )
        test["deep_sets_prediction"] = deep_prediction
        predictions.append(test)
        history["held_out_family"] = held_out
        histories.append(history)
        metadata["held_out_family"] = held_out
        fold_metadata.append(metadata)
        torch.save(
            {"state_dict": state, "member_features": member_features, "context_features": context_features, "metadata": metadata},
            model_dir / f"held_out_{held_out}.pt",
        )
    prediction_frame = pd.concat(predictions, ignore_index=True)
    history_frame = pd.concat(histories, ignore_index=True)
    decisions = _decision_rows(prediction_frame, str(config["evaluation"]["primary_reference"]))
    family_summary, overall_summary = _summaries(decisions)
    intervals = _bootstrap_gain_intervals(
        decisions,
        int(config["evaluation"]["bootstrap_repetitions"]),
        seed,
    )
    overall_summary = overall_summary.merge(
        intervals, on=["subset", "method"], how="left", validate="one_to_one"
    )
    deep_all = overall_summary.loc[
        overall_summary["subset"].eq("all") & overall_summary["method"].eq("deep_sets")
    ].iloc[0]
    scientific_pass = bool(
        deep_all["family_equal_gain_over_stable_plus_tracing"]
        > float(config["evaluation"]["minimum_family_equal_gain"])
        and deep_all["families_with_positive_gain"]
        >= int(config["evaluation"]["minimum_families_with_positive_gain"])
    )
    prediction_frame.to_csv(results_dir / "loso_set_predictions.csv.gz", index=False, compression="gzip")
    decisions.to_csv(results_dir / "loso_context_decisions.csv.gz", index=False, compression="gzip")
    family_summary.to_csv(results_dir / "family_model_summary.csv", index=False)
    overall_summary.to_csv(results_dir / "overall_model_summary.csv", index=False)
    history_frame.to_csv(results_dir / "training_history.csv", index=False)
    checks = {
        "precision_gate_passed": True,
        "four_independent_test_families": len(families) == 4,
        "one_prediction_per_set": not prediction_frame.duplicated(SET_KEY).any(),
        "finite_predictions": bool(np.isfinite(prediction_frame[["ridge_prediction", "deep_sets_prediction"]]).all().all()),
        "no_family_overlap_within_fold": True,
        "cpu_device_forward_consistency": consistency_ok,
        "deep_sets_nonconstant_within_context": bool(
            prediction_frame.groupby("context_id", observed=True)["deep_sets_prediction"].nunique().gt(1).all()
        ),
        "all_contexts_have_reference": bool(decisions["stable_plus_tracing_value"].notna().all()),
        "all_regrets_nonnegative": bool((decisions.filter(regex="_regret$") >= -1e-12).all().all()),
    }
    audit = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "scientific_gate": {
            "status": "pass" if scientific_pass else "fail",
            "family_equal_gain_over_stable_plus_tracing": float(deep_all["family_equal_gain_over_stable_plus_tracing"]),
            "families_with_positive_gain": int(deep_all["families_with_positive_gain"]),
            "minimum_families_with_positive_gain": int(config["evaluation"]["minimum_families_with_positive_gain"]),
        },
        "sets": len(prediction_frame),
        "contexts": len(decisions),
        "families": len(families),
    }
    if audit["status"] != "pass":
        raise ValueError(f"model artifact audit failed: {audit}")
    (results_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (results_dir / "fold_metadata.json").write_text(json.dumps(fold_metadata, indent=2), encoding="utf-8")
    (results_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    _plot(family_summary, overall_summary, history_frame, report_dir)
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
        "input_hashes": {key: _sha256(path) for key, path in paths.items()},
        "artifact_audit": audit["status"],
        "scientific_gate": audit["scientific_gate"]["status"],
    }
    (results_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (report_dir / "README.md").write_text(
        "# Strict leave-one-animal-system-out set-value model\n\n"
        f"Profile: **{profile_name}**. Sets: {len(prediction_frame):,}; contexts: {len(decisions):,}; independent held-out families: {len(families)}. "
        f"Artifact audit: **{audit['status']}**. Pre-specified Deep Sets improvement gate: **{audit['scientific_gate']['status']}**.\n\n"
        "Every test animal system was absent from fitting, feature normalization, validation, and early stopping. "
        "The primary table includes every decision context; `precision_qualified` rows are a pre-declared sensitivity analysis based on EXP-013. "
        "The sampled oracle is an upper bound only within the finite candidate-set playground, not a deployable method or global optimum.\n",
        encoding="utf-8",
    )
    return results_dir, report_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strict LOSO fixed-budget set-value models")
    parser.add_argument("--config", type=Path, default=Path("configs/EXP-20260816-014_set_value_model.yaml"))
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    args = parser.parse_args()
    results, reports = run(args.config, args.profile)
    print(f"Results: {results}")
    print(f"Reports: {reports}")


if __name__ == "__main__":
    main()
