#!/usr/bin/env python
"""
Utility script for Project 03:
- Loads the production LR bundles from Project 02, evaluates them on train/test,
  and emits metrics tables plus confusion-matrix figures.
- Builds comparison visuals for Project 01 results (best test AUC per model).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression  # noqa: F401  (needed for joblib)
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

# Optional imbalanced-learn imports
try:
    from imblearn.combine import SMOTEENN, SMOTETomek
    from imblearn.over_sampling import ADASYN, BorderlineSMOTE, SMOTE

    HAS_IMB = True
except Exception:  # pragma: no cover - runtime guard
    SMOTE = ADASYN = BorderlineSMOTE = SMOTEENN = SMOTETomek = None  # type: ignore
    HAS_IMB = False


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
NOTEBOOK_PATH = SRC_DIR / "02_lr_production_playbook" / "lr_production_playbook.ipynb"
PROJECT01_RESULTS = SRC_DIR / "01_brute-force_model_search" / "experiment_results_full.csv"

OUTPUT_DIR = Path(__file__).parent
ASSET_DIR = OUTPUT_DIR / "assets"
ARTIFACT_DIR = OUTPUT_DIR / "artifacts"
ASSET_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "Attrition"
DROP_COLS = ["Over18", "StandardHours", "EmployeeNumber"]
NOMINAL_FEATURES = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime",
]

SCENARIOS: Dict[str, Dict[str, str]] = {
    "lr_score_chaser": {"title": "Score Chaser (AUC-first)"},
    "lr_validation_guardian": {"title": "Validation Guardian (stability-first)"},
}

FEATURE_PIPELINE_READY = False


def _bootstrap_feature_pipeline() -> None:
    """Executes the FeaturePipeline definition directly from the notebook."""
    global FEATURE_PIPELINE_READY
    if FEATURE_PIPELINE_READY:
        return

    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    helper_code = "".join(notebook["cells"][5]["source"])
    pipeline_code = "".join(notebook["cells"][6]["source"])

    scope: Dict[str, Any] = {
        "Optional": Optional,
        "Dict": Dict,
        "List": List,
        "Tuple": Tuple,
        "Any": Any,
        "pd": pd,
        "np": np,
        "OneHotEncoder": OneHotEncoder,
        "StandardScaler": StandardScaler,
        "PolynomialFeatures": PolynomialFeatures,
        "SelectKBest": SelectKBest,
        "f_classif": f_classif,
        "NOMINAL_FEATURES": NOMINAL_FEATURES,
        "HAS_IMB": HAS_IMB,
        "SMOTE": SMOTE,
        "ADASYN": ADASYN,
        "BorderlineSMOTE": BorderlineSMOTE,
        "SMOTEENN": SMOTEENN,
        "SMOTETomek": SMOTETomek,
    }

    exec(helper_code, scope)
    exec(pipeline_code, scope)

    globals()["build_sampler"] = scope["build_sampler"]
    globals()["apply_sampler"] = scope["apply_sampler"]
    globals()["FeaturePipeline"] = scope["FeaturePipeline"]
    FEATURE_PIPELINE_READY = True


def _load_bundle(scenario: str) -> Dict[str, Any]:
    _bootstrap_feature_pipeline()
    bundle_path = SRC_DIR / "02_lr_production_playbook" / "artifacts" / scenario / "models" / f"{scenario}.pkl"
    return joblib.load(bundle_path)


def _load_datasets() -> Dict[str, Tuple[pd.DataFrame, np.ndarray]]:
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    for col in DROP_COLS:
        if col in train_df.columns:
            train_df = train_df.drop(columns=col)
        if col in test_df.columns:
            test_df = test_df.drop(columns=col)
    y_train = train_df[TARGET_COL].astype(int).to_numpy()
    y_test = test_df[TARGET_COL].astype(int).to_numpy()
    X_train = train_df.drop(columns=[TARGET_COL])
    X_test = test_df.drop(columns=[TARGET_COL])
    return {"train": (X_train, y_train), "test": (X_test, y_test)}


def _ensemble_predict(bundle: Dict[str, Any], X: pd.DataFrame) -> np.ndarray:
    agg = bundle.get("soft_voting", "mean")
    member_probs: List[np.ndarray] = []
    for member in bundle["models"]:
        pipeline = member["pipeline"]
        model = member["model"]
        Xt = pipeline.transform(X)
        member_probs.append(model.predict_proba(Xt)[:, 1])
    stacked = np.vstack(member_probs)
    if agg == "median":
        return np.median(stacked, axis=0)
    return np.mean(stacked, axis=0)


def _evaluate_split(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> Dict[str, Any]:
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds)
    metrics = {
        "threshold": float(threshold),
        "auc": float(roc_auc_score(y_true, probs)),
        "ap": float(average_precision_score(y_true, probs)),
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "support": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
        "confusion_matrix": cm.astype(int).tolist(),
    }
    return metrics


def _plot_confusion(cm: np.ndarray, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Attrition", "Attrition"],
        yticklabels=["No Attrition", "Attrition"],
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _build_project01_chart(out_path: Path) -> None:
    df = pd.read_csv(PROJECT01_RESULTS)
    top = df.sort_values("test_auc", ascending=False).groupby("model", as_index=False).first()
    top = top.sort_values("test_auc")

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#94b3fd" if m != "LR" else "#1f77b4" for m in top["model"]]
    ax.barh(top["model"], top["test_auc"], color=colors)
    for idx, row in top.iterrows():
        ax.text(
            row["test_auc"] + 0.001,
            idx,
            f"{row['test_auc']:.3f}",
            va="center",
            fontsize=9,
        )
    ax.set_xlabel("Best Hold-out AUC")
    ax.set_title("Project 01 · Best Test AUC by Model Family\n(615 brute-force runs)")
    ax.axvline(0.8908, color="#1f77b4", linestyle="--", linewidth=1, label="Selected LR")
    ax.set_xlim(left=min(top["test_auc"]) - 0.01, right=max(top["test_auc"]) + 0.01)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    sns.set_theme(style="whitegrid")
    datasets = _load_datasets()
    results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    flat_rows: List[Dict[str, Any]] = []

    for scenario, meta in SCENARIOS.items():
        bundle = _load_bundle(scenario)
        threshold = float(bundle.get("threshold", 0.5))
        scenario_res: Dict[str, Dict[str, Any]] = {}
        for split_name, (X, y) in datasets.items():
            probs = _ensemble_predict(bundle, X)
            metrics = _evaluate_split(y, probs, threshold)
            scenario_res[split_name] = metrics
            cm = np.array(metrics["confusion_matrix"])
            cm_path = ASSET_DIR / f"{scenario}_{split_name}_confusion.png"
            _plot_confusion(cm, f"{meta['title']} · {split_name.title()} Confusion", cm_path)

            flat_row = {
                "scenario": scenario,
                "scenario_title": meta["title"],
                "split": split_name,
                **{k: v for k, v in metrics.items() if k != "confusion_matrix"},
            }
            flat_rows.append(flat_row)

        results[scenario] = scenario_res

    metrics_json = ARTIFACT_DIR / "lr_bundle_metrics.json"
    metrics_csv = ARTIFACT_DIR / "lr_bundle_metrics.csv"
    metrics_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    pd.DataFrame(flat_rows).to_csv(metrics_csv, index=False)

    chart_path = ASSET_DIR / "project01_best_model_auc.png"
    _build_project01_chart(chart_path)

    print(f"[OK] Metrics saved to {metrics_json.relative_to(PROJECT_ROOT)} and {metrics_csv.relative_to(PROJECT_ROOT)}.")
    print(f"[OK] Figures written under {ASSET_DIR.relative_to(PROJECT_ROOT)}.")


if __name__ == "__main__":
    main()
