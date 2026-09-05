from pathlib import Path
from typing import Dict, List, Optional, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve


def _setup_plot_style():
    """Applies clean, publication-quality aesthetic parameters."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def plot_metric_comparison(
    metrics_df: pd.DataFrame,
    output_path: Union[str, Path],
    title: str = "Comparative Model Performance",
) -> None:
    """Generates a grouped bar chart comparing key evaluation metrics across models."""
    _setup_plot_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_cols = [c for c in ["Accuracy", "F1", "ROC-AUC", "MCC"] if c in metrics_df.columns]
    if not plot_cols:
        plot_cols = [c for c in metrics_df.columns if c != "Model"]

    models = metrics_df["Model"].tolist()
    n_models = len(models)
    n_metrics = len(plot_cols)

    x = np.arange(n_models)
    width = 0.8 / n_metrics

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, metric in enumerate(plot_cols):
        vals = pd.to_numeric(metrics_df[metric], errors="coerce").fillna(0).values
        ax.bar(x + (i - n_metrics / 2.0 + 0.5) * width, vals, width, label=metric)

    ax.set_title(title, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.legend(frameon=True, loc="lower right")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_roc_pr_curves(
    y_true: np.ndarray,
    models_probs: Dict[str, np.ndarray],
    output_path: Union[str, Path],
) -> None:
    """Plots side-by-side ROC and Precision-Recall curves for binary classification."""
    _setup_plot_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5))

    for name, probs in models_probs.items():
        p1 = probs[:, 1] if probs.ndim == 2 and probs.shape[1] == 2 else probs.ravel()
        fpr, tpr, _ = roc_curve(y_true, p1)
        prec, rec, _ = precision_recall_curve(y_true, p1)

        ax_roc.plot(fpr, tpr, label=name, lw=2)
        ax_pr.plot(rec, prec, label=name, lw=2)

    # ROC Baseline
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.7)
    ax_roc.set_title("Receiver Operating Characteristic (ROC)")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(loc="lower right", frameon=True)

    # PR Baseline
    no_skill = np.sum(y_true == 1) / len(y_true)
    ax_pr.axhline(no_skill, color="k", linestyle="--", lw=1, alpha=0.7)
    ax_pr.set_title("Precision-Recall Curve")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.legend(loc="lower left", frameon=True)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_confusion_matrices(
    y_true: np.ndarray,
    models_preds: Dict[str, np.ndarray],
    output_path: Union[str, Path],
    class_names: Optional[List[str]] = None,
) -> None:
    """Generates normalized confusion matrix subplots for each model."""
    _setup_plot_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_models = len(models_preds)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5), squeeze=False)
    classes = class_names or [str(c) for c in np.unique(y_true)]

    for idx, (name, preds) in enumerate(models_preds.items()):
        ax = axes[0, idx]
        cm = confusion_matrix(y_true, preds, normalize="true")
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1)

        ax.set_title(f"{name}\nNormalized CM")
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(classes, rotation=45)
        ax.set_yticklabels(classes)
        ax.set_ylabel("True Label" if idx == 0 else "")
        ax.set_xlabel("Predicted Label")

        for i in range(len(classes)):
            for j in range(len(classes)):
                val = cm[i, j]
                color = "white" if val > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_calibration_curves(
    curves_dict: Dict[str, Dict[str, np.ndarray]],
    output_path: Union[str, Path],
) -> None:
    """Plots probability calibration reliability diagrams."""
    _setup_plot_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect Calibration")

    for name, curve in curves_dict.items():
        ax.plot(
            curve["mean_predicted_value"],
            curve["fraction_of_positives"],
            "s-",
            lw=2,
            label=name,
        )

    ax.set_title("Probability Calibration (Reliability Diagram)")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-0.05, 1.05)
    ax.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_stability_heatmap(
    pairwise_matrix: pd.DataFrame,
    output_path: Union[str, Path],
    title: str = "Explanation Stability Matrix",
) -> None:
    """Plots pairwise correlation heatmap across cross-validation folds."""
    _setup_plot_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(pairwise_matrix.values, cmap="viridis", vmin=0.0, vmax=1.0)
    fig.colorbar(cax)

    ax.set_title(title, pad=20)
    ax.set_xticks(range(len(pairwise_matrix.columns)))
    ax.set_yticks(range(len(pairwise_matrix.index)))
    ax.set_xticklabels(pairwise_matrix.columns, rotation=45, ha="left")
    ax.set_yticklabels(pairwise_matrix.index)

    for i in range(len(pairwise_matrix.index)):
        for j in range(len(pairwise_matrix.columns)):
            val = pairwise_matrix.iloc[i, j]
            color = "white" if val < 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_robustness_degradation(
    robustness_df: pd.DataFrame,
    output_path: Union[str, Path],
) -> None:
    """Plots metric degradation under increasing perturbation severity."""
    _setup_plot_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    noise_df = robustness_df[robustness_df["experiment_type"] == "feature_noise"].copy()
    if noise_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name in noise_df["model_name"].unique():
        sub = noise_df[noise_df["model_name"] == model_name].sort_values(by="severity_level")
        ax.plot(
            pd.to_numeric(sub["severity_level"]),
            sub["f1"],
            "o-",
            lw=2,
            label=f"{model_name} (F1)",
        )

    ax.set_title("Robustness: F1 Degradation vs Gaussian Noise Level")
    ax.set_xlabel("Noise Severity (Standard Deviation Scale $\\sigma$)")
    ax.set_ylabel("F1 Score")
    ax.legend(frameon=True)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
