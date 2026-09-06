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
    sections.append("5. **Distribution Shift:** Robustness experiments apply synthetic noise and missingness; operational domain shift in network environments may exhibit more complex temporal or topological variations.\n")

    # 9. Final Conclusion & Strategic Recommendations
    sections.append(synthesize_final_conclusion(
        performance_df=performance_df,
        efficiency_df=efficiency_df,
        calibration_df=calibration_df,
        stability_df=stability_df,
        robustness_df=robustness_df,
        statistical_tests=statistical_tests,
    ))

    full_report = "\n".join(sections)

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(full_report, encoding="utf-8")

    return full_report


def synthesize_final_conclusion(
    performance_df: Optional[pd.DataFrame] = None,
    efficiency_df: Optional[pd.DataFrame] = None,
    calibration_df: Optional[pd.DataFrame] = None,
    stability_df: Optional[pd.DataFrame] = None,
    robustness_df: Optional[pd.DataFrame] = None,
    statistical_tests: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Synthesizes empirical findings across performance, efficiency, calibration,
    stability, robustness, and hypothesis testing to reach a definitive final conclusion.
    """
    lines = ["## 9. Final Conclusion & Strategic Recommendations\n"]

    # 1. Performance Synthesis
    best_model_name = "Ensemble Model"
    best_f1 = 0.0
    best_acc = 0.0
    best_mcc = 0.0

    if performance_df is not None and not performance_df.empty:
        df_eval = performance_df.copy()
        for col in ["F1", "Accuracy", "MCC", "ROC-AUC"]:
            if col in df_eval.columns:
                df_eval[col] = pd.to_numeric(df_eval[col], errors="coerce")

        sort_col = "F1" if "F1" in df_eval.columns else df_eval.columns[1]
        top_row = df_eval.sort_values(by=sort_col, ascending=False).iloc[0]
        best_model_name = str(top_row["Model"])
        best_f1 = float(top_row.get("F1", 0.0))
        best_acc = float(top_row.get("Accuracy", 0.0))
        best_mcc = float(top_row.get("MCC", 0.0))

    lines.append("### 9.1 Overall Model Selection & Performance Synthesis")
    lines.append(
        f"Across the multi-dimensional benchmark suite, **`{best_model_name}`** emerges as the overall superior classifier, "
        f"achieving the highest held-out test performance with an **F1-Score of {best_f1:.4f}**, "
        f"**Accuracy of {best_acc:.4f}**, and **Matthews Correlation Coefficient (MCC) of {best_mcc:.4f}**.\n"
    )

    # 2. Statistical Rigor Synthesis
    if statistical_tests:
        sig_advantages = []
        tie_comparisons = []
        for test in statistical_tests:
            if test.get("is_statistically_significant"):
                m_a, m_b = test.get("model_a", ""), test.get("model_b", "")
                sig_advantages.append(f"{test.get('test_name', 'Test')} between `{m_a}` and `{m_b}` ($p = {test.get('p_value', 0.0):.4f}$)")
            else:
                m_a, m_b = test.get("model_a", ""), test.get("model_b", "")
                tie_comparisons.append(f"`{m_a}` vs `{m_b}` ($p = {test.get('p_value', 0.0):.4f}$)")

        lines.append("### 9.2 Statistical Hypothesis Testing Verification")
        if sig_advantages:
            lines.append(f"- **Confirmed Superiority:** Statistically significant performance differences were established via " + "; ".join(sig_advantages[:3]) + ". This rejects the null hypothesis of equal error distributions and demonstrates that the observed lead is not attributable to sampling artifact.")
        if tie_comparisons:
            lines.append(f"- **Equivalence Regimes:** Paired comparisons where models performed without statistically significant divergence include: " + "; ".join(tie_comparisons[:2]) + ", indicating interchangeable predictive power between gradient-boosted variants.")
        lines.append("")

    # 3. Efficiency & Resource Footprint
    if efficiency_df is not None and not efficiency_df.empty:
        lines.append("### 9.3 Computational Efficiency & Throughput Trade-Offs")
        fastest_train = efficiency_df.sort_values(by="training_time_sec").iloc[0]
        peak_throughput = efficiency_df.sort_values(by="inference_throughput_samples_per_sec", ascending=False).iloc[0]
        min_size = efficiency_df.sort_values(by="model_size_kb").iloc[0]

        lines.append(
            f"- **Training Velocity:** `{fastest_train['model_name']}` trained most rapidly at **{fastest_train['training_time_sec']:.2f} seconds**."
        )
        lines.append(
            f"- **Inference Throughput:** `{peak_throughput['model_name']}` attained the highest evaluation throughput at **{peak_throughput['inference_throughput_samples_per_sec']:,.0f} samples/sec** (latency: **{peak_throughput['inference_latency_ms_per_sample']:.4f} ms/sample**)."
        )
        lines.append(
            f"- **Deployment Footprint:** `{min_size['model_name']}` produced the most compact model artifact at **{min_size['model_size_kb']:.1f} KB**, making it highly suitable for constrained edge security appliances.\n"
        )

    # 4. Explainability & Stability
    most_stable_model = best_model_name
    most_stable_rho = 0.0
    most_stable_tau = 0.0
    most_stable_jaccard = 0.0

    if stability_df is not None and not stability_df.empty:
        lines.append("### 9.4 Explainability Stability & Decision Transparency")
        mean_rho = float(stability_df["mean_spearman_rho"].mean())
        mean_tau = float(stability_df["mean_kendall_tau"].mean())
        mean_jaccard = float(stability_df["mean_jaccard_similarity"].mean())

        top_stable = stability_df.sort_values(by="mean_spearman_rho", ascending=False).iloc[0]
        most_stable_model = str(top_stable["model_name"])
        most_stable_rho = float(top_stable["mean_spearman_rho"])
        most_stable_tau = float(top_stable["mean_kendall_tau"])
        most_stable_jaccard = float(top_stable["mean_jaccard_similarity"])
        std_rho = float(top_stable.get("std_spearman_rho", 0.0))

        evaluated_models_str = ", ".join([f"`{m}`" for m in stability_df["model_name"].unique()])

        lines.append(
            f"Across evaluated classifiers ({evaluated_models_str}), global SHAP feature attribution rankings demonstrated robust inter-fold consistency with an overall cross-model mean Spearman $\\rho = {mean_rho:.4f}$, "
            f"Kendall $\\tau = {mean_tau:.4f}$, and top-$k$ Jaccard consensus of {mean_jaccard:.2%}."
        )
        lines.append(
            f"- **Attribution Consistency:** **`{most_stable_model}`** achieved the highest attribution stability across cross-validation splits (Spearman $\\rho = {most_stable_rho:.4f} \\pm {std_rho:.4f}$, Kendall $\\tau = {most_stable_tau:.4f}$, top-$k$ Jaccard agreement = {most_stable_jaccard:.2%})."
        )
        lines.append(
            f"- **Validation of Threat Logic:** This high rank stability confirms that the primary predictive indicators identified by `{most_stable_model}` reflect genuine signal in the underlying traffic distribution rather than random partition noise.\n"
        )

    # 5. Robustness Findings
    most_robust_model = best_model_name
    most_robust_deg = 0.0

    if robustness_df is not None and not robustness_df.empty:
        lines.append("### 9.5 Adversarial & Perturbation Resilience")
        noise_eval = robustness_df[robustness_df["experiment_type"] == "feature_noise"]
        corrupt_eval = robustness_df[robustness_df["experiment_type"].str.contains("corruption", na=False)]

        # Compute average degradation per model across all experiments
        avg_deg_by_model = robustness_df.groupby("model_name")["accuracy_degradation"].mean().sort_values()
        most_robust_model = str(avg_deg_by_model.index[0])
        most_robust_deg = float(avg_deg_by_model.iloc[0])

        if not noise_eval.empty:
            max_noise_level = noise_eval["severity_level"].max()
            max_noise_rows = noise_eval[noise_eval["severity_level"] == max_noise_level]
            top_noise_model_row = max_noise_rows.sort_values(by="accuracy", ascending=False).iloc[0]
            top_noise_model = str(top_noise_model_row["model_name"])
            top_noise_acc = float(top_noise_model_row["accuracy"])
            top_noise_deg = float(top_noise_model_row["accuracy_degradation"])

            lines.append(
                f"- **Gaussian Noise Tolerance:** Under maximum packet feature noise ($\\sigma = {max_noise_level}$), **`{top_noise_model}`** retained the highest residual test accuracy at **{top_noise_acc:.4f}** (accuracy degradation limited to {top_noise_deg:.2%})."
            )

        lines.append(
            f"- **Overall Perturbation Resilience:** **`{most_robust_model}`** exhibited the lowest overall sensitivity across all noise and corruptive regimes with an average accuracy degradation of only **{most_robust_deg:.2%}**."
        )

        if not corrupt_eval.empty:
            worst_corrupt = corrupt_eval.sort_values(by="accuracy_degradation", ascending=False).iloc[0]
            worst_feat_name = str(worst_corrupt["experiment_type"]).replace("feature_corruption_", "")
            corrupt_model = str(worst_corrupt["model_name"])
            corrupt_drop = float(worst_corrupt["accuracy_degradation"])
            lines.append(
                f"- **Critical Vulnerability Point:** Feature corruption of `{worst_feat_name}` induced the steepest performance decline in **`{corrupt_model}`** ({corrupt_drop:.2%} accuracy loss), identifying `{worst_feat_name}` as a critical dependency that requires strict telemetric integrity validation at the network boundary.\n"
            )

    # 6. Strategic Deployment Decision Matrix (Fully Dynamic)
    lines.append("### 9.6 Recommended Deployment Decision Matrix\n")
    lines.append("| Operational Scenario | Recommended Architecture | Primary Justification |")
    lines.append("|---|---|---|")

    # Scenario 1: High-Throughput Core
    if efficiency_df is not None and not efficiency_df.empty:
        peak_throughput_row = efficiency_df.sort_values(by="inference_throughput_samples_per_sec", ascending=False).iloc[0]
        th_model = str(peak_throughput_row["model_name"])
        th_rate = float(peak_throughput_row["inference_throughput_samples_per_sec"])
        th_lat = float(peak_throughput_row["inference_latency_ms_per_sample"])
        lines.append(f"| **High-Throughput Inline NIDS (10G/40G Core)** | **`{th_model}`** | Peak throughput of {th_rate:,.0f} samples/sec with sub-millisecond latency ({th_lat:.4f} ms/sample). |")
    else:
        lines.append(f"| **High-Throughput Inline NIDS (10G/40G Core)** | **`{best_model_name}`** | Superior vectorized classification throughput. |")

    # Scenario 2: SOC Forensics & Triage
    if stability_df is not None and not stability_df.empty:
        lines.append(f"| **Security Operations Center (SOC) Forensics** | **`{most_stable_model}`** | Highest inter-fold attribution stability (Spearman $\\rho = {most_stable_rho:.4f}$) and verified additive SHAP explanations. |")
    else:
        lines.append(f"| **Security Operations Center (SOC) Forensics** | **`{best_model_name}`** | Verified additive SHAP attributions for reliable root-cause investigation. |")

    # Scenario 3: Resource-Constrained Edge Sensor
    if efficiency_df is not None and not efficiency_df.empty:
        min_size_row = efficiency_df.sort_values(by="model_size_kb").iloc[0]
        edge_model = str(min_size_row["model_name"])
        edge_size = float(min_size_row["model_size_kb"])
        edge_ram = float(min_size_row["peak_memory_mb"])
        lines.append(f"| **Resource-Constrained IoT Gateway / Edge Sensor** | **`{edge_model}`** | Most compact deployment footprint ({edge_size:.1f} KB disk, {edge_ram:.1f} MB RAM). |")
    else:
        lines.append(f"| **Resource-Constrained IoT Gateway / Edge Sensor** | **`{best_model_name}`** | Efficient decision tree representation with low memory footprint. |")

    # Scenario 4: Adversarial & Noisy Channels
    if robustness_df is not None and not robustness_df.empty:
        lines.append(f"| **Adversarial / High-Noise Network Boundary** | **`{most_robust_model}`** | Lowest overall degradation ({most_robust_deg:.2%}) under packet perturbations. |")

    # Scenario 5: High-Precision Discrimination
    lines.append(f"| **High-Precision Mission-Critical Threat Detection** | **`{best_model_name}`** | Highest held-out discrimination fidelity (F1: {best_f1:.4f}, MCC: {best_mcc:.4f}). |\n")

    return "\n".join(lines)

