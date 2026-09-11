from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.reporting.plots import (
    plot_calibration_curves,
    plot_confusion_matrices,
    plot_metric_comparison,
    plot_robustness_degradation,
    plot_roc_pr_curves,
    plot_stability_heatmap,
)
from src.reporting.report_generator import generate_research_report
from src.reporting.tables import format_latex_table, format_results_markdown_table


def test_plot_generation(tmp_path):
    # Test metric comparison plot
    metrics_df = pd.DataFrame({
        "Model": ["RF", "XGB", "LGBM"],
        "Accuracy": [0.95, 0.96, 0.94],
        "F1": [0.93, 0.94, 0.92],
        "ROC-AUC": [0.98, 0.99, 0.97],
        "MCC": [0.90, 0.92, 0.88],
    })
    metric_plot_path = tmp_path / "metrics.png"
    plot_metric_comparison(metrics_df, metric_plot_path)
    assert metric_plot_path.exists()
    assert metric_plot_path.stat().st_size > 0

    # Test ROC/PR curves
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0] * 5)
    models_probs = {
        "RF": np.column_stack([1.0 - y_true * 0.8, y_true * 0.8 + 0.1]),
        "XGB": np.column_stack([1.0 - y_true * 0.9, y_true * 0.9 + 0.05]),
    }
    roc_plot_path = tmp_path / "roc_pr.png"
    plot_roc_pr_curves(y_true, models_probs, roc_plot_path)
    assert roc_plot_path.exists()

    # Test confusion matrix plot
    models_preds = {
        "RF": y_true,
        "XGB": y_true,
    }
    cm_plot_path = tmp_path / "cm.png"
    plot_confusion_matrices(y_true, models_preds, cm_plot_path)
    assert cm_plot_path.exists()

    # Test stability heatmap
    mat = pd.DataFrame([[1.0, 0.8], [0.8, 1.0]], index=["F1", "F2"], columns=["F1", "F2"])
    stab_plot_path = tmp_path / "stability.png"
    plot_stability_heatmap(mat, stab_plot_path)
    assert stab_plot_path.exists()


def test_table_and_report_generation(tmp_path):
    df = pd.DataFrame({"Model": ["RF"], "F1": [0.95]})
    md_table = format_results_markdown_table(df)
    latex_table = format_latex_table(df)

    assert "| Model |" in md_table
    assert "\\begin{table}" in latex_table

    report_path = tmp_path / "report.md"
    report_text = generate_research_report(
        dataset_metadata={"n_samples": 1000, "n_raw_features": 20, "n_selected_features": 10},
        config_dict={},
        performance_df=df,
        output_path=str(report_path),
    )

    assert "# Machine Learning Research Experiment Report" in report_text
    assert "## 9. Final Conclusion & Strategic Recommendations" in report_text
    assert "Deployment Decision Matrix" in report_text
    assert report_path.exists()
