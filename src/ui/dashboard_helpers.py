from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.explainability.shap_explainer import ShapExplainer


def get_available_datasets(dataset_dir: Path = Path("./dataset")) -> List[Path]:
    """Finds all CSV files inside the dataset directory."""
    if not dataset_dir.exists():
        return []
    return sorted(list(dataset_dir.glob("*.csv")))


@st.cache_data(show_spinner=False)
def load_all_reports(reports_dir_str: str) -> Dict[str, pd.DataFrame]:
    """Loads all CSV experiment reports into a cacheable dictionary."""
    reports_dir = Path(reports_dir_str)
    reports = {}
    if not reports_dir.exists():
        return reports

    for csv_file in reports_dir.glob("*.csv"):
        try:
            reports[csv_file.stem] = pd.read_csv(csv_file)
        except Exception:
            continue
    return reports


def load_model_checkpoint(model_path: Path) -> Optional[Any]:
    """Loads a serialized model checkpoint from outputs/models/."""
    if not model_path.exists():
        return None
    try:
        return joblib.load(model_path)
    except Exception:
        return None


def plot_local_shap_waterfall(
    feature_names: List[str],
    shap_values: np.ndarray,
    feature_values: np.ndarray,
    base_value: float,
    predicted_prob: float,
    model_name: str,
    top_n: int = 10,
) -> plt.Figure:
    """
    Renders a publication-ready horizontal attribution bar chart for a local sample.
    Features with positive SHAP push towards Attack/Positive class (crimson),
    negative SHAP push towards Benign/Normal class (teal).
    """
    df_local = pd.DataFrame({
        "Feature": feature_names,
        "SHAP": shap_values,
        "Abs_SHAP": np.abs(shap_values),
        "Value": feature_values,
    }).sort_values(by="Abs_SHAP", ascending=False).head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.4 * top_n)), dpi=150)
    colors = ["#e63946" if val >= 0 else "#1d3557" for val in df_local["SHAP"]]

    bars = ax.barh(
        range(len(df_local)),
        df_local["SHAP"],
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.85,
    )

    y_labels = [
        f"{row['Feature']} = {row['Value']:.3g}" if isinstance(row['Value'], (int, float, np.number))
        else f"{row['Feature']} = {str(row['Value'])[:8]}"
        for _, row in df_local.iterrows()
    ]
    ax.set_yticks(range(len(df_local)))
    ax.set_yticklabels(y_labels, fontsize=10, fontweight="medium")

    # Add numeric annotations
    for bar, val in zip(bars, df_local["SHAP"]):
        offset = 0.005 if val >= 0 else -0.005
        ha = "left" if val >= 0 else "right"
        ax.text(
            val + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}",
            va="center",
            ha=ha,
            fontsize=9,
            fontweight="bold",
            color="#e63946" if val >= 0 else "#1d3557",
        )

    ax.axvline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("SHAP Attribution (Margin Impact on Threat Likelihood)", fontsize=10, fontweight="bold")
    ax.set_title(
        f"Local Instance Explanation [{model_name}]\nBase Value: {base_value:.3f} | Predicted P(Attack): {predicted_prob:.3f}",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )
    ax.grid(axis="x", alpha=0.3, linestyle=":")
    plt.tight_layout()
    return fig


def format_p_value(p: float) -> str:
    """Formats statistical p-values with scientific notation or stars."""
    if p < 0.001:
        return f"{p:.2e} (***)"
    elif p < 0.01:
        return f"{p:.4f} (**)"
    elif p < 0.05:
        return f"{p:.4f} (*)"
    else:
        return f"{p:.4f} (ns)"
