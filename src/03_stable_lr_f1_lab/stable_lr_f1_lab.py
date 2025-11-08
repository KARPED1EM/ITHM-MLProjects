from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import (
    ADASYN,
    BorderlineSMOTE,
    RandomOverSampler,
    SMOTE,
)
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
DATA_ARTIFACT_DIR = ARTIFACT_DIR / "data"
FIG_ARTIFACT_DIR = ARTIFACT_DIR / "figures"
MODEL_ARTIFACT_DIR = ARTIFACT_DIR / "models"
LOG_ARTIFACT_DIR = ARTIFACT_DIR / "logs"

TARGET = "Attrition"
ID_COLUMN = "EmployeeNumber"

RANDOM_STATE = 42
CV_FOLDS = 5


def setup_logger() -> Tuple[logging.Logger, Path]:
    LOG_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_ARTIFACT_DIR / f"stable_lr_run_{timestamp}.log"

    logger = logging.getLogger("stable_lr_f1_lab")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s ✨ [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.propagate = False
    return logger, log_path


def log_block(logger: logging.Logger, title: str, lines: Iterable[str]) -> None:
    bullet_lines = "\n".join(f"    • {line}" for line in lines)
    logger.info("%s\n%s", title, bullet_lines)


def load_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    return train_df, test_df


def add_manual_features(df: pd.DataFrame) -> pd.DataFrame:
    """Hook for lightweight, human-curated features."""
    engineered = df.copy()

    eps = 1e-3
    level = (engineered["JobLevel"].clip(lower=1)).astype(float)
    engineered["income_per_level"] = engineered["MonthlyIncome"] / level
    engineered["years_at_company_per_level"] = engineered["YearsAtCompany"] / level
    engineered["years_in_role_ratio"] = (engineered["YearsInCurrentRole"] + eps) / (
        engineered["YearsAtCompany"] + eps
    )
    satisfaction_cols = [
        "EnvironmentSatisfaction",
        "JobSatisfaction",
        "RelationshipSatisfaction",
        "WorkLifeBalance",
    ]
    engineered["satisfaction_mean"] = engineered[satisfaction_cols].mean(axis=1)
    engineered["managerial_tenure_gap"] = (
        engineered["YearsWithCurrManager"] - engineered["YearsSinceLastPromotion"]
    )
    return engineered


def safe_log1p(data: np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(data, a_min=0, a_max=None))


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    feature_df = df.drop(columns=[TARGET])
    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = sorted(set(feature_df.columns) - set(numeric_cols))

    # Remove identifier columns from the numeric bucket to avoid leakage.
    if ID_COLUMN in numeric_cols:
        numeric_cols.remove(ID_COLUMN)

    log_candidates = [
        "MonthlyIncome",
        "DistanceFromHome",
        "TotalWorkingYears",
        "YearsAtCompany",
        "YearsInCurrentRole",
        "YearsSinceLastPromotion",
        "YearsWithCurrManager",
        "NumCompaniesWorked",
    ]
    log_cols = [col for col in log_candidates if col in numeric_cols]
    linear_numeric_cols = [col for col in numeric_cols if col not in log_cols]

    log_pipeline = Pipeline(
        steps=[
            (
                "log1p",
                FunctionTransformer(
                    safe_log1p,
                    feature_names_out="one-to-one",
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )

    linear_numeric_pipeline = Pipeline(steps=[("scaler", StandardScaler())])

    categorical_pipeline = Pipeline(
        steps=[
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("log_num", log_pipeline, log_cols),
            ("num", linear_numeric_pipeline, linear_numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor


def build_pipeline(preprocessor: ColumnTransformer, sampler) -> ImbPipeline:
    steps: List[Tuple[str, object]] = [("preprocessor", preprocessor)]
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(
        (
            "clf",
            LogisticRegression(
                max_iter=5000,
                random_state=RANDOM_STATE,
                n_jobs=None,
            ),
        )
    )
    return ImbPipeline(steps=steps)


def sampler_space() -> Dict[str, Optional[object]]:
    return {
        "none": None,
        "random_over": RandomOverSampler(random_state=RANDOM_STATE),
        "smote": SMOTE(random_state=RANDOM_STATE, k_neighbors=5),
        "borderline_smote": BorderlineSMOTE(random_state=RANDOM_STATE),
        "adasyn": ADASYN(random_state=RANDOM_STATE),
    }


def hyperparameter_grid() -> List[Dict[str, object]]:
    return [
        {
            "clf__solver": ["lbfgs"],
            "clf__penalty": ["l2"],
            "clf__C": [0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0],
            "clf__class_weight": [None, "balanced"],
        },
        {
            "clf__solver": ["liblinear"],
            "clf__penalty": ["l1", "l2"],
            "clf__C": [0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0],
            "clf__class_weight": [None, "balanced"],
        },
        {
            "clf__solver": ["saga"],
            "clf__penalty": ["elasticnet"],
            "clf__l1_ratio": [0.1, 0.2, 0.3, 0.5, 0.7],
            "clf__C": [0.1, 0.3, 0.5, 1.0, 2.0, 3.0],
            "clf__class_weight": [None, "balanced"],
        },
    ]


def detailed_classification_report(
    y_true: pd.Series,
    y_pred: np.ndarray,
    proba: np.ndarray,
    dataset_name: str = "Dataset",
) -> str:
    """Generate a detailed classification report with nice formatting."""
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        average_precision_score,
        confusion_matrix,
    )

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Metrics that require probabilities
    roc_auc = roc_auc_score(y_true, proba)
    avg_precision = average_precision_score(y_true, proba)

    lines = []
    lines.append("")
    lines.append("=" * 70)
    lines.append(f"  DETAILED CLASSIFICATION REPORT - {dataset_name}")
    lines.append("=" * 70)
    lines.append("")

    # Confusion Matrix
    lines.append("Confusion Matrix:")
    lines.append("")
    lines.append("                 Predicted")
    lines.append("                 Stay    Leave")
    lines.append("    Actual  Stay   {:>4}     {:>4}".format(tn, fp))
    lines.append("           Leave   {:>4}     {:>4}".format(fn, tp))
    lines.append("")

    # Class-wise metrics
    lines.append("Class-wise Metrics:")
    lines.append("")
    lines.append("              Precision    Recall    F1-Score    Support")
    lines.append("    " + "-" * 58)

    # Class 0 (Stay)
    prec_0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec_0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_0 = 2 * prec_0 * rec_0 / (prec_0 + rec_0) if (prec_0 + rec_0) > 0 else 0.0
    support_0 = int(tn + fp)
    lines.append("    Stay        {:.4f}      {:.4f}      {:.4f}      {:>4}".format(
        prec_0, rec_0, f1_0, support_0
    ))

    # Class 1 (Leave/Attrition)
    prec_1 = precision
    rec_1 = recall
    f1_1 = f1
    support_1 = int(tp + fn)
    lines.append("    Leave       {:.4f}      {:.4f}      {:.4f}      {:>4}".format(
        prec_1, rec_1, f1_1, support_1
    ))
    lines.append("    " + "-" * 58)

    # Weighted average
    total = support_0 + support_1
    prec_avg = (prec_0 * support_0 + prec_1 * support_1) / total
    rec_avg = (rec_0 * support_0 + rec_1 * support_1) / total
    f1_avg = (f1_0 * support_0 + f1_1 * support_1) / total
    lines.append("    avg/total   {:.4f}      {:.4f}      {:.4f}      {:>4}".format(
        prec_avg, rec_avg, f1_avg, total
    ))
    lines.append("")

    # Overall metrics
    lines.append("Overall Metrics:")
    lines.append("")
    lines.append("    Accuracy:                {:.4f}".format(accuracy))
    lines.append("    Precision (Leave):       {:.4f}".format(precision))
    lines.append("    Recall (Leave):          {:.4f}".format(recall))
    lines.append("    F1-Score (Leave):        {:.4f}".format(f1))
    lines.append("    Specificity (Stay):      {:.4f}".format(specificity))
    lines.append("    ROC-AUC:                 {:.4f}".format(roc_auc))
    lines.append("    Average Precision:       {:.4f}".format(avg_precision))
    lines.append("")
    lines.append("=" * 70)
    lines.append("")

    return "\n".join(lines)


def extract_coefficient_frame(estimator: ImbPipeline) -> pd.DataFrame:
    preprocessor = estimator.named_steps["preprocessor"]
    clf: LogisticRegression = estimator.named_steps["clf"]  # type: ignore[assignment]
    feature_names = preprocessor.get_feature_names_out()
    coefficients = clf.coef_.ravel()
    coef_frame = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
        }
    )
    coef_frame["abs_coefficient"] = coef_frame["coefficient"].abs()
    return coef_frame.sort_values("abs_coefficient", ascending=False).reset_index(
        drop=True
    )


def run_sampler_sweeps(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: ColumnTransformer,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame, str, ImbPipeline, Dict[str, float]]:
    sampler_results: List[Dict[str, object]] = []
    cv_frames: List[pd.DataFrame] = []

    best_name = ""
    best_estimator: Optional[ImbPipeline] = None
    best_score = -np.inf

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "f1": make_scorer(f1_score),
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score),
        "roc_auc": "roc_auc",
        "accuracy": "accuracy",
    }

    for sampler_name, sampler in sampler_space().items():
        log_block(
            logger,
            f"🧪 Starting sampler sweep: {sampler_name}",
            [
                "building pipeline",
                f"cv folds: {CV_FOLDS}",
                "scoring: f1 + precision + recall + roc_auc + accuracy",
            ],
        )
        pipeline = build_pipeline(preprocessor, sampler)
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=hyperparameter_grid(),
            scoring=scoring,
            refit="f1",
            cv=cv,
            n_jobs=-1,
            verbose=0,
        )
        grid.fit(X, y)

        best_idx = grid.best_index_
        params_count = len(grid.cv_results_["params"])

        # Collect detailed statistics for each metric
        sampler_results.append(
            {
                "sampler": sampler_name,
                # F1 statistics
                "cv_f1_mean": grid.cv_results_["mean_test_f1"][best_idx],
                "cv_f1_std": grid.cv_results_["std_test_f1"][best_idx],
                "cv_f1_min": grid.cv_results_["mean_test_f1"][best_idx] - grid.cv_results_["std_test_f1"][best_idx],
                "cv_f1_max": grid.cv_results_["mean_test_f1"][best_idx] + grid.cv_results_["std_test_f1"][best_idx],
                # Precision statistics
                "cv_precision_mean": grid.cv_results_["mean_test_precision"][best_idx],
                "cv_precision_std": grid.cv_results_["std_test_precision"][best_idx],
                "cv_precision_min": grid.cv_results_["mean_test_precision"][best_idx] - grid.cv_results_["std_test_precision"][best_idx],
                "cv_precision_max": grid.cv_results_["mean_test_precision"][best_idx] + grid.cv_results_["std_test_precision"][best_idx],
                # Recall statistics
                "cv_recall_mean": grid.cv_results_["mean_test_recall"][best_idx],
                "cv_recall_std": grid.cv_results_["std_test_recall"][best_idx],
                "cv_recall_min": grid.cv_results_["mean_test_recall"][best_idx] - grid.cv_results_["std_test_recall"][best_idx],
                "cv_recall_max": grid.cv_results_["mean_test_recall"][best_idx] + grid.cv_results_["std_test_recall"][best_idx],
                # ROC-AUC statistics
                "cv_roc_auc_mean": grid.cv_results_["mean_test_roc_auc"][best_idx],
                "cv_roc_auc_std": grid.cv_results_["std_test_roc_auc"][best_idx],
                "cv_roc_auc_min": grid.cv_results_["mean_test_roc_auc"][best_idx] - grid.cv_results_["std_test_roc_auc"][best_idx],
                "cv_roc_auc_max": grid.cv_results_["mean_test_roc_auc"][best_idx] + grid.cv_results_["std_test_roc_auc"][best_idx],
                # Accuracy statistics
                "cv_accuracy_mean": grid.cv_results_["mean_test_accuracy"][best_idx],
                "cv_accuracy_std": grid.cv_results_["std_test_accuracy"][best_idx],
                "cv_accuracy_min": grid.cv_results_["mean_test_accuracy"][best_idx] - grid.cv_results_["std_test_accuracy"][best_idx],
                "cv_accuracy_max": grid.cv_results_["mean_test_accuracy"][best_idx] + grid.cv_results_["std_test_accuracy"][best_idx],
                "best_params": json.dumps(grid.best_params_),
            }
        )

        cv_result_frame = pd.DataFrame(grid.cv_results_)
        cv_result_frame["sampler"] = sampler_name
        cv_frames.append(cv_result_frame)

        log_block(
            logger,
            f"📈 Sampler '{sampler_name}' detailed CV statistics",
            [
                f"grid candidates tested: {params_count}",
                f"F1:        mean={grid.cv_results_['mean_test_f1'][best_idx]:.4f}, std={grid.cv_results_['std_test_f1'][best_idx]:.4f}",
                f"Precision: mean={grid.cv_results_['mean_test_precision'][best_idx]:.4f}, std={grid.cv_results_['std_test_precision'][best_idx]:.4f}",
                f"Recall:    mean={grid.cv_results_['mean_test_recall'][best_idx]:.4f}, std={grid.cv_results_['std_test_recall'][best_idx]:.4f}",
                f"ROC-AUC:   mean={grid.cv_results_['mean_test_roc_auc'][best_idx]:.4f}, std={grid.cv_results_['std_test_roc_auc'][best_idx]:.4f}",
                f"Accuracy:  mean={grid.cv_results_['mean_test_accuracy'][best_idx]:.4f}, std={grid.cv_results_['std_test_accuracy'][best_idx]:.4f}",
            ],
        )

        if grid.best_score_ > best_score:
            best_score = grid.best_score_
            best_name = sampler_name
            best_estimator = grid.best_estimator_

    summary_df = (
        pd.DataFrame(sampler_results)
        .sort_values("cv_f1_mean", ascending=False)
        .reset_index(drop=True)
    )
    cv_df = pd.concat(cv_frames, ignore_index=True)

    if best_estimator is None:
        raise RuntimeError("No best estimator found during sampler sweep.")

    best_summary = summary_df.loc[summary_df["sampler"] == best_name].iloc[0]

    # Collect CV metrics for the best model
    best_cv_metrics = {
        "f1_mean": float(best_summary["cv_f1_mean"]),
        "f1_std": float(best_summary["cv_f1_std"]),
        "precision_mean": float(best_summary["cv_precision_mean"]),
        "precision_std": float(best_summary["cv_precision_std"]),
        "recall_mean": float(best_summary["cv_recall_mean"]),
        "recall_std": float(best_summary["cv_recall_std"]),
        "roc_auc_mean": float(best_summary["cv_roc_auc_mean"]),
        "roc_auc_std": float(best_summary["cv_roc_auc_std"]),
        "accuracy_mean": float(best_summary["cv_accuracy_mean"]),
        "accuracy_std": float(best_summary["cv_accuracy_std"]),
    }

    log_block(
        logger,
        "🏆 Best sampler locked with detailed CV metrics",
        [
            f"name: {best_name}",
            f"F1:        {best_cv_metrics['f1_mean']:.4f} ± {best_cv_metrics['f1_std']:.4f}",
            f"Precision: {best_cv_metrics['precision_mean']:.4f} ± {best_cv_metrics['precision_std']:.4f}",
            f"Recall:    {best_cv_metrics['recall_mean']:.4f} ± {best_cv_metrics['recall_std']:.4f}",
            f"ROC-AUC:   {best_cv_metrics['roc_auc_mean']:.4f} ± {best_cv_metrics['roc_auc_std']:.4f}",
            f"Accuracy:  {best_cv_metrics['accuracy_mean']:.4f} ± {best_cv_metrics['accuracy_std']:.4f}",
        ],
    )

    return summary_df, cv_df, best_name, best_estimator, best_cv_metrics


def evaluate_on_holdout(
    model: ImbPipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    identifiers: pd.Series,
    logger: logging.Logger,
    dataset_name: str = "Test Set",
) -> Dict[str, object]:
    proba = model.predict_proba(X_test)[:, 1]
    preds_default = model.predict(X_test)
    default_f1 = f1_score(y_test, preds_default)
    default_precision = precision_score(y_test, preds_default, zero_division=0)
    default_recall = recall_score(y_test, preds_default)
    default_accuracy = accuracy_score(y_test, preds_default)

    thresholds = np.linspace(0.10, 0.90, 17)
    f1_scores = []
    for thr in thresholds:
        preds = (proba >= thr).astype(int)
        f1_scores.append(f1_score(y_test, preds))

    best_idx = int(np.argmax(f1_scores))
    best_threshold = float(thresholds[best_idx])
    best_preds = (proba >= best_threshold).astype(int)
    best_f1 = f1_score(y_test, best_preds)
    conf = confusion_matrix(y_test, best_preds)
    precision, recall, pr_thresholds = precision_recall_curve(y_test, proba)
    best_precision = precision_score(y_test, best_preds, zero_division=0)
    best_recall = recall_score(y_test, best_preds)
    best_accuracy = accuracy_score(y_test, best_preds)
    roc_auc = roc_auc_score(y_test, proba)
    avg_precision = average_precision_score(y_test, proba)
    fpr, tpr, roc_thresholds = roc_curve(y_test, proba)
    prob_true, prob_pred = calibration_curve(y_test, proba, n_bins=10)

    # Generate detailed classification report
    detailed_report_default = detailed_classification_report(
        y_test, preds_default, proba, f"{dataset_name} (Default Threshold 0.50)"
    )
    detailed_report_best = detailed_classification_report(
        y_test, best_preds, proba, f"{dataset_name} (Best Threshold {best_threshold:.2f})"
    )

    predictions = pd.DataFrame(
        {
            ID_COLUMN: identifiers,
            "y_true": y_test,
            "proba": proba,
            "pred_default": preds_default,
            f"pred_thr_{best_threshold:.2f}": best_preds,
        }
    )

    threshold_frame = pd.DataFrame(
        {
            "threshold": thresholds,
            "f1": f1_scores,
        }
    )

    report = {
        "default_threshold_f1": default_f1,
        "best_threshold": best_threshold,
        "best_threshold_f1": best_f1,
        "confusion_matrix": conf.tolist(),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "pr_thresholds": pr_thresholds.tolist(),
        "best_precision": best_precision,
        "best_recall": best_recall,
        "best_accuracy": best_accuracy,
        "predictions": predictions,
        "threshold_frame": threshold_frame,
        "roc_auc": roc_auc,
        "average_precision": avg_precision,
        "default_precision": default_precision,
        "default_recall": default_recall,
        "default_accuracy": default_accuracy,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "roc_thresholds": roc_thresholds.tolist(),
        "calibration_prob_true": prob_true.tolist(),
        "calibration_prob_pred": prob_pred.tolist(),
        "detailed_report_default": detailed_report_default,
        "detailed_report_best": detailed_report_best,
    }
    log_block(
        logger,
        f"🧾 {dataset_name} metrics summary",
        [
            f"ROC-AUC: {roc_auc:.4f}",
            f"Average precision: {avg_precision:.4f}",
            f"Default threshold (0.50) F1: {default_f1:.4f}",
            f"Default precision / recall / accuracy: {default_precision:.4f} / {default_recall:.4f} / {default_accuracy:.4f}",
            f"Best threshold ({best_threshold:.2f}) F1: {best_f1:.4f}",
            f"Best precision / recall / accuracy: {best_precision:.4f} / {best_recall:.4f} / {best_accuracy:.4f}",
        ],
    )

    # Log detailed classification reports
    logger.info(detailed_report_default)
    logger.info(detailed_report_best)

    return report


def plot_cv_vs_test_metrics(
    cv_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    path: Path,
) -> None:
    """Compare CV and test metrics side by side."""
    metrics_names = ["F1", "Precision", "Recall", "ROC-AUC", "Accuracy"]
    cv_values = [
        cv_metrics["f1_mean"],
        cv_metrics["precision_mean"],
        cv_metrics["recall_mean"],
        cv_metrics["roc_auc_mean"],
        cv_metrics["accuracy_mean"],
    ]
    cv_stds = [
        cv_metrics["f1_std"],
        cv_metrics["precision_std"],
        cv_metrics["recall_std"],
        cv_metrics["roc_auc_std"],
        cv_metrics["accuracy_std"],
    ]
    test_values = [
        test_metrics["best_threshold_f1"],
        test_metrics["best_precision"],
        test_metrics["best_recall"],
        test_metrics["roc_auc"],
        test_metrics["best_accuracy"],
    ]

    x = np.arange(len(metrics_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, cv_values, width, label="CV (mean)",
                   color="#4c72b0", yerr=cv_stds, capsize=5)
    bars2 = ax.bar(x + width/2, test_values, width, label="Test (best threshold)",
                   color="#55a868")

    ax.set_ylabel("Score")
    ax.set_xlabel("Metrics")
    ax.set_title("CV vs Test Set Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.set_ylim(0, 1.05)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_metrics_detailed_comparison(
    cv_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    path: Path,
) -> None:
    """Detailed comparison showing mean ± std for CV and point estimate for test."""
    metrics_names = ["F1", "Precision", "Recall", "ROC-AUC", "Accuracy"]

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))

    for idx, metric in enumerate(["f1", "precision", "recall", "roc_auc", "accuracy"]):
        ax = axes[idx]

        cv_mean = cv_metrics[f"{metric}_mean"]
        cv_std = cv_metrics[f"{metric}_std"]

        if metric == "f1":
            test_val = test_metrics["best_threshold_f1"]
        elif metric == "precision":
            test_val = test_metrics["best_precision"]
        elif metric == "recall":
            test_val = test_metrics["best_recall"]
        elif metric == "accuracy":
            test_val = test_metrics["best_accuracy"]
        else:  # roc_auc
            test_val = test_metrics["roc_auc"]

        # Plot CV as error bar
        ax.errorbar([0], [cv_mean], yerr=[cv_std], fmt='o', markersize=10,
                   capsize=10, capthick=2, color="#4c72b0", label="CV")
        # Plot test as point
        ax.plot([1], [test_val], 'o', markersize=10, color="#55a868", label="Test")

        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(0, 1.05)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["CV", "Test"])
        ax.set_ylabel("Score")
        ax.set_title(metrics_names[idx])
        ax.grid(axis='y', alpha=0.3)

        # Add value annotations
        ax.text(0, cv_mean + cv_std + 0.02, f"{cv_mean:.3f}±{cv_std:.3f}",
               ha='center', fontsize=8)
        ax.text(1, test_val + 0.02, f"{test_val:.3f}", ha='center', fontsize=8)

    fig.suptitle("Detailed Metrics Comparison: CV (mean±std) vs Test", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_sampler_summary(summary_df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        data=summary_df,
        x="sampler",
        y="cv_f1_mean",
        hue="sampler",
        palette="viridis",
        legend=False,
        ax=ax,
    )
    ax.tick_params(axis="x", rotation=25)
    ax.set_ylabel("Mean CV F1")
    ax.set_xlabel("Sampler")
    ax.set_title("Sampler impact on CV F1")
    for patch in ax.patches:
        height = patch.get_height()
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            height + 0.001,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_threshold_curve(
    threshold_df: pd.DataFrame, best_threshold: float, path: Path
) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(threshold_df["threshold"], threshold_df["f1"], marker="o")
    plt.axvline(best_threshold, color="red", linestyle="--", label="best threshold")
    plt.xlabel("Decision Threshold")
    plt.ylabel("F1 Score")
    plt.title("Threshold sweep on holdout")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_confusion(conf: np.ndarray, path: Path) -> None:
    plt.figure(figsize=(4, 4))
    sns.heatmap(
        conf,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Stay", "Leave"],
        yticklabels=["Stay", "Leave"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion matrix (best threshold)")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_pr_curve(
    precision: Iterable[float],
    recall: Iterable[float],
    best_precision: float,
    best_recall: float,
    path: Path,
) -> None:
    plt.figure(figsize=(5, 4))
    plt.plot(recall, precision, label="PR curve")
    plt.scatter(
        [best_recall],
        [best_precision],
        color="red",
        label="best threshold",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve (holdout)")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_roc_curve_holdout(
    fpr: Iterable[float], tpr: Iterable[float], auc_score: float, path: Path
) -> None:
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC={auc_score:.3f})", color="#4c72b0")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve (holdout)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_probability_density(predictions: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(6, 4))
    sns.kdeplot(
        data=predictions,
        x="proba",
        hue="y_true",
        fill=True,
        common_norm=False,
        alpha=0.4,
        palette=["#c44e52", "#55a868"],
    )
    plt.xlabel("Predicted probability of attrition")
    plt.ylabel("Density")
    plt.title("Probability density by true class")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_calibration(
    prob_true: Iterable[float], prob_pred: Iterable[float], path: Path
) -> None:
    plt.figure(figsize=(5, 4))
    plt.plot(prob_pred, prob_true, marker="o", label="model calibration")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    plt.xlabel("Predicted probability mean")
    plt.ylabel("Observed frequency")
    plt.title("Calibration curve (holdout)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_coefficient_importance(
    coef_frame: pd.DataFrame, path: Path, top_n: int = 15
) -> None:
    if coef_frame.empty:
        return
    top_pos = coef_frame.nlargest(top_n, "coefficient")
    top_neg = coef_frame.nsmallest(top_n, "coefficient")
    display_frame = (
        pd.concat([top_neg, top_pos])
        .sort_values("coefficient")
        .assign(positive=lambda df: df["coefficient"] > 0)
    )
    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=display_frame,
        x="coefficient",
        y="feature",
        hue="positive",
        dodge=False,
        palette={True: "#55a868", False: "#c44e52"},
        legend=False,
    )
    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("Coefficient weight")
    plt.ylabel("Feature")
    plt.title("Top logistic regression coefficients")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_cv_heatmap(cv_df: pd.DataFrame, path: Path) -> None:
    if "param_clf__C" not in cv_df:
        return
    heatmap_df = (
        cv_df.dropna(subset=["param_clf__C"])
        .assign(param_clf__C=lambda df: df["param_clf__C"].astype(float))
        .groupby(["sampler", "param_clf__C"])["mean_test_f1"]
        .max()
        .unstack()
        .sort_index()
    )
    if heatmap_df.empty:
        return
    plt.figure(figsize=(8, max(4, heatmap_df.shape[0] * 0.6)))
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".3f",
        cmap="rocket_r",
        cbar_kws={"label": "mean CV F1"},
    )
    plt.xlabel("Regularization strength (C)")
    plt.ylabel("Sampler")
    plt.title("Best mean CV F1 by sampler & C")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def persist_artifacts(
    summary_df: pd.DataFrame,
    cv_df: pd.DataFrame,
    evaluation: Dict[str, object],
    coef_frame: pd.DataFrame,
    cv_metrics: Dict[str, float],
    logger: logging.Logger,
) -> None:
    DATA_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # Save data tables
    summary_df.to_csv(DATA_ARTIFACT_DIR / "sampler_summary.csv", index=False)
    cv_df.to_csv(DATA_ARTIFACT_DIR / "sampler_grid_results.csv", index=False)
    coef_frame.to_csv(DATA_ARTIFACT_DIR / "best_model_coefficients.csv", index=False)
    evaluation["predictions"].to_csv(
        DATA_ARTIFACT_DIR / "best_model_test_predictions.csv",
        index=False,
    )
    evaluation["threshold_frame"].to_csv(
        DATA_ARTIFACT_DIR / "threshold_sweep.csv", index=False
    )

    # Save detailed JSON report
    with open(DATA_ARTIFACT_DIR / "test_report.json", "w", encoding="utf-8") as fp:
        serializable = {
            "default_threshold_f1": evaluation["default_threshold_f1"],
            "default_precision": evaluation["default_precision"],
            "default_recall": evaluation["default_recall"],
            "default_accuracy": evaluation["default_accuracy"],
            "best_threshold": evaluation["best_threshold"],
            "best_threshold_f1": evaluation["best_threshold_f1"],
            "best_precision": evaluation["best_precision"],
            "best_recall": evaluation["best_recall"],
            "best_accuracy": evaluation["best_accuracy"],
            "roc_auc": evaluation["roc_auc"],
            "average_precision": evaluation["average_precision"],
            "confusion_matrix": evaluation["confusion_matrix"],
        }
        json.dump(serializable, fp, indent=2)

    # Save CV metrics
    with open(DATA_ARTIFACT_DIR / "cv_metrics.json", "w", encoding="utf-8") as fp:
        json.dump(cv_metrics, fp, indent=2)

    # Save detailed classification reports to text files
    with open(DATA_ARTIFACT_DIR / "detailed_report_default.txt", "w", encoding="utf-8") as fp:
        fp.write(evaluation["detailed_report_default"])
    with open(DATA_ARTIFACT_DIR / "detailed_report_best.txt", "w", encoding="utf-8") as fp:
        fp.write(evaluation["detailed_report_best"])

    # Generate all plots
    plot_sampler_summary(summary_df, FIG_ARTIFACT_DIR / "sampler_f1.png")
    plot_cv_vs_test_metrics(
        cv_metrics,
        evaluation,
        FIG_ARTIFACT_DIR / "cv_vs_test_metrics.png",
    )
    plot_metrics_detailed_comparison(
        cv_metrics,
        evaluation,
        FIG_ARTIFACT_DIR / "metrics_detailed_comparison.png",
    )
    plot_threshold_curve(
        evaluation["threshold_frame"],
        evaluation["best_threshold"],
        FIG_ARTIFACT_DIR / "threshold_sweep.png",
    )
    plot_confusion(
        np.array(evaluation["confusion_matrix"]),
        FIG_ARTIFACT_DIR / "confusion_matrix.png",
    )
    plot_pr_curve(
        evaluation["precision"],
        evaluation["recall"],
        evaluation["best_precision"],
        evaluation["best_recall"],
        FIG_ARTIFACT_DIR / "precision_recall.png",
    )
    plot_roc_curve_holdout(
        evaluation["fpr"],
        evaluation["tpr"],
        evaluation["roc_auc"],
        FIG_ARTIFACT_DIR / "roc_curve.png",
    )
    plot_probability_density(
        evaluation["predictions"],
        FIG_ARTIFACT_DIR / "probability_density.png",
    )
    plot_calibration(
        evaluation["calibration_prob_true"],
        evaluation["calibration_prob_pred"],
        FIG_ARTIFACT_DIR / "calibration_curve.png",
    )
    plot_coefficient_importance(
        coef_frame,
        FIG_ARTIFACT_DIR / "top_coefficients.png",
    )
    plot_cv_heatmap(
        cv_df,
        FIG_ARTIFACT_DIR / "cv_sampler_heatmap.png",
    )

    log_block(
        logger,
        "💾 Artifacts persisted",
        [
            f"data tables stored under: {DATA_ARTIFACT_DIR}",
            f"figures stored under: {FIG_ARTIFACT_DIR}",
            f"models stored under: {MODEL_ARTIFACT_DIR}",
            "detailed classification reports saved as .txt files",
        ],
    )


def main() -> None:
    sns.set_theme(style="whitegrid")
    logger, log_path = setup_logger()
    log_block(
        logger,
        "🚀 Stable LR lab run started",
        [
            f"log file: {log_path}",
            f"random state: {RANDOM_STATE}",
            f"cv folds: {CV_FOLDS}",
        ],
    )
    try:
        train_df, test_df = load_dataset()
        log_block(
            logger,
            "📦 Data loaded",
            [
                f"train shape: {train_df.shape}",
                f"test shape: {test_df.shape}",
            ],
        )
        train_df = add_manual_features(train_df)
        test_df = add_manual_features(test_df)
        log_block(
            logger,
            "🧩 Manual features applied",
            [
                f"feature columns now: {train_df.shape[1]}",
                "feature set aligned across train/test",
            ],
        )

        y_train = train_df[TARGET].astype(int)
        X_train = train_df.drop(columns=[TARGET])
        y_test = test_df[TARGET].astype(int)
        X_test = test_df.drop(columns=[TARGET])

        identifiers = (
            X_test[ID_COLUMN]
            if ID_COLUMN in X_test.columns
            else pd.Series(np.arange(len(X_test)), name=ID_COLUMN)
        )
        if ID_COLUMN in X_train.columns:
            X_train = X_train.drop(columns=[ID_COLUMN])
        if ID_COLUMN in X_test.columns:
            X_test = X_test.drop(columns=[ID_COLUMN])

        preprocessor = build_preprocessor(train_df)
        summary_df, cv_df, best_sampler_name, best_estimator, cv_metrics = run_sampler_sweeps(
            X_train, y_train, preprocessor, logger
        )

        coef_frame = extract_coefficient_frame(best_estimator)
        evaluation = evaluate_on_holdout(
            best_estimator, X_test, y_test, identifiers, logger, dataset_name="Test Set"
        )
        persist_artifacts(summary_df, cv_df, evaluation, coef_frame, cv_metrics, logger)
        model_path = MODEL_ARTIFACT_DIR / f"stable_lr_best_{best_sampler_name}.pkl"
        joblib.dump(best_estimator, model_path)
        log_block(
            logger,
            "✅ Run complete - Summary",
            [
                f"best sampler: {best_sampler_name}",
                f"CV F1: {cv_metrics['f1_mean']:.4f} ± {cv_metrics['f1_std']:.4f}",
                f"CV ROC-AUC: {cv_metrics['roc_auc_mean']:.4f} ± {cv_metrics['roc_auc_std']:.4f}",
                f"Test ROC-AUC: {evaluation['roc_auc']:.4f}",
                f"Test F1 (best threshold {evaluation['best_threshold']:.2f}): {evaluation['best_threshold_f1']:.4f}",
                f"Test Accuracy (best threshold): {evaluation['best_accuracy']:.4f}",
                f"model stored at: {model_path}",
            ],
        )
    except Exception as exc:  # pragma: no cover - logging for visibility
        logger.exception("❌ Run failed due to: %s", exc)
        raise


if __name__ == "__main__":
    main()
