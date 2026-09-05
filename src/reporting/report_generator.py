from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from src.reporting.tables import format_results_markdown_table


def generate_research_report(
    dataset_metadata: Dict[str, Any],
    config_dict: Dict[str, Any],
    performance_df: pd.DataFrame,
    efficiency_df: Optional[pd.DataFrame] = None,
    calibration_df: Optional[pd.DataFrame] = None,
    stability_df: Optional[pd.DataFrame] = None,
    robustness_df: Optional[pd.DataFrame] = None,
    statistical_tests: Optional[List[Dict[str, Any]]] = None,
    bootstrap_df: Optional[pd.DataFrame] = None,
    output_path: Optional[str] = None,
) -> str:
    """
    Generates a publication-grade, Markdown research experiment report.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sections = []

    # Title & Metadata
    sections.append(f"# Machine Learning Research Experiment Report\n")
    sections.append(f"**Generated:** {timestamp}  ")
    sections.append(f"**Classification Mode:** {dataset_metadata.get('classification_mode', 'N/A').upper()}  ")
    sections.append(f"**Evaluation Strategy:** {dataset_metadata.get('cv_folds', 5)}-Fold Stratified Cross-Validation + Independent Test Set\n")
    sections.append("---\n")

    # 1. Dataset Overview
    sections.append("## 1. Dataset Overview & Data Hygiene\n")
    sections.append(f"- **Total Observations:** {dataset_metadata.get('n_samples', 'N/A'):,}")
    sections.append(f"- **Raw Feature Count:** {dataset_metadata.get('n_raw_features', 'N/A')}")
    sections.append(f"- **Selected Feature Count:** {dataset_metadata.get('n_selected_features', 'N/A')} (via Mutual Information)")
    sections.append(f"- **Class Distribution:** {dataset_metadata.get('class_distribution', 'N/A')}")
    sections.append(f"- **Infinite Values Detected:** {dataset_metadata.get('infinite_count', 0):,} (converted to missing for systematic imputation)")
    sections.append(f"- **Missing Values Imputed:** {dataset_metadata.get('missing_count', 0):,} ({dataset_metadata.get('imputation_strategy', 'mean')} imputation)\n")

    # 2. Predictive Performance
    sections.append("## 2. Predictive Performance Comparison\n")
    sections.append("The table below presents the cross-validated and held-out test evaluation across candidate algorithms:\n")
    if performance_df is not None and not performance_df.empty:
        sections.append(format_results_markdown_table(performance_df) + "\n")
    else:
        sections.append("*No performance data available.*\n")

    # 2.1 Bootstrap Confidence Intervals
    if bootstrap_df is not None and not bootstrap_df.empty:
        sections.append("### 2.1 Non-Parametric Bootstrap Confidence Intervals (95% CI)\n")
        sections.append(format_results_markdown_table(bootstrap_df) + "\n")

    # 3. Computational Benchmarks
    if efficiency_df is not None and not efficiency_df.empty:
        sections.append("## 3. Computational Efficiency & Resource Footprint\n")
        sections.append(format_results_markdown_table(efficiency_df) + "\n")

    # 4. Probability Calibration Analysis
    if calibration_df is not None and not calibration_df.empty:
        sections.append("## 4. Probability Calibration Analysis\n")
        sections.append("Reliability of confidence scores evaluated via Brier Score and Expected Calibration Error (ECE):\n")
        sections.append(format_results_markdown_table(calibration_df) + "\n")

    # 5. Model Explainability & Stability
    if stability_df is not None and not stability_df.empty:
        sections.append("## 5. Model Explainability & Explanation Stability\n")
        sections.append("Consistency of global SHAP feature attributions measured across cross-validation folds:\n")
        sections.append(format_results_markdown_table(stability_df) + "\n")

    # 6. Robustness & Perturbation Analysis
    if robustness_df is not None and not robustness_df.empty:
        sections.append("## 6. Robustness Under Controlled Perturbations\n")
        sections.append("Performance degradation under Gaussian feature noise, random missingness, and feature corruption:\n")
        # Display top 12 rows
        sections.append(format_results_markdown_table(robustness_df.head(12)) + "\n")

    # 7. Statistical Hypothesis Testing
    if statistical_tests:
        sections.append("## 7. Statistical Validation & Hypothesis Testing\n")
        for test in statistical_tests:
            sections.append(f"### {test.get('test_name', 'Test')}: {test.get('model_a', '')} vs {test.get('model_b', '')}")
            sections.append(f"- **Test Statistic:** {test.get('statistic', 0.0):.4f}")
            sections.append(f"- **P-Value:** {test.get('p_value', 1.0):.5f} (Threshold: $\\alpha={test.get('significance_threshold', 0.05)}$)")
            sections.append(f"- **Statistically Significant:** **{'YES' if test.get('is_statistically_significant') else 'NO'}**")
            sections.append(f"- **Effect Size:** {test.get('effect_size_name', 'Effect Size')} = {test.get('effect_size', 0.0):.4f}")
            sections.append(f"- **Interpretation:** {test.get('interpretation', '')}\n")

    # 8. Scientifically Important Caveats & Limitations
    sections.append("## 8. Scientifically Important Caveats & Limitations\n")
    sections.append("1. **Data Leakage Safeguards:** All preprocessing, scalers, imputers, and feature selectors were fitted strictly on the training partitions and applied downstream without inspecting validation or test instances.")
    sections.append("2. **Statistical vs Practical Significance:** A statistically significant difference between models (e.g. via McNemar's or Wilcoxon tests) does not automatically denote practical or operational superiority.")
    sections.append("3. **SHAP Semantics:** SHAP provides additive attributions under coalition game assumptions. It explains model behavior relative to a baseline expectation, not direct real-world causal mechanisms.")
    sections.append("4. **Imbalance Metrics:** Accuracy alone is misleading on imbalanced data; decision-making should be grounded in MCC, PR-AUC, and F1.")
    sections.append("5. **Distribution Shift:** Robustness experiments apply synthetic noise and missingness; operational domain shift in network environments may exhibit more complex temporal or topological variations.")

    full_report = "\n".join(sections)

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(full_report, encoding="utf-8")

    return full_report
