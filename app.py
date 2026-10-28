from pathlib import Path
import subprocess
import sys
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.ui.dashboard_helpers import (
    format_p_value,
    get_available_datasets,
    load_all_reports,
    load_model_checkpoint,
    plot_local_shap_waterfall,
)


# --- Streamlit Configuration ---
st.set_page_config(
    page_title="Explainable ML Cyber Defense Framework",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for Academic & Industrial Presentation
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 800;
        color: #1d3557;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #457b9d;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .status-success {
        background-color: #d1e7dd;
        color: #0f5132;
    }
    .status-warning {
        background-color: #fff3cd;
        color: #664d03;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def run_pipeline_process(args_list: List[str], log_placeholder) -> bool:
    """Runs run_pipeline.py as a subprocess and streams logs to the UI."""
    cmd = [sys.executable, "run_pipeline.py"] + args_list
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    log_lines = []
    with st.spinner("Executing End-to-End Pipeline across Phases 1-8..."):
        if process.stdout:
            for line in iter(process.stdout.readline, ""):
                log_lines.append(line)
                # Keep last 15 lines visible in terminal widget
                log_placeholder.code("".join(log_lines[-15:]), language="bash")
        process.wait()

    if process.returncode == 0:
        st.success("Pipeline executed successfully! Artifacts refreshed.")
        st.cache_data.clear()
        return True
    else:
        st.error(f"Pipeline failed with exit code {process.returncode}.")
        return False


def main():
    # Header
    st.markdown('<div class="main-header">🛡️ Explainable Machine Learning Framework</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Dataset-Agnostic Cyber Attack Detection Pipeline with Rigorous Calibration, SHAP Attributions, Robustness Stress-Testing & Hypothesis Testing</div>',
        unsafe_allow_html=True,
    )

    # Directories
    output_dir = Path("./outputs")
    reports_dir = output_dir / "reports"
    plots_dir = output_dir / "plots"
    models_dir = output_dir / "models"
    dataset_dir = Path("./dataset")

    reports = load_all_reports(str(reports_dir))
    has_results = len(reports) > 0

    # --- SIDEBAR CONTROLS ---
    with st.sidebar:
        st.header("⚙️ Experiment Controls")

        if has_results:
            st.markdown('<span class="status-badge status-success">● Experiment Artifacts Loaded</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge status-warning">▲ No Artifacts Found</span>', unsafe_allow_html=True)

        st.divider()
        st.subheader("1. Data Ingestion")
        available_csvs = get_available_datasets(dataset_dir)
        data_source_mode = st.radio(
            "Source Mode",
            ["Existing Directory CSV", "Generate Synthetic Sample", "Upload Custom CSV"],
            index=0 if available_csvs else 1,
        )

        selected_dataset_path = None
        if data_source_mode == "Existing Directory CSV":
            if available_csvs:
                csv_names = [p.name for p in available_csvs]
                chosen_idx = 0
                chosen_name = st.selectbox("Select CSV file", csv_names, index=chosen_idx)
                selected_dataset_path = dataset_dir / chosen_name
                st.caption(f"Path: `{selected_dataset_path}` ({selected_dataset_path.stat().st_size / (1024*1024):.2f} MB)")
            else:
                st.warning("No CSV files found in `./dataset/`.")
        elif data_source_mode == "Upload Custom CSV":
            uploaded_file = st.file_uploader("Upload Network Traffic CSV", type=["csv"])
            if uploaded_file is not None:
                save_path = dataset_dir / uploaded_file.name
                dataset_dir.mkdir(parents=True, exist_ok=True)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                selected_dataset_path = save_path
                st.success(f"Uploaded and saved: `{save_path.name}`")

        st.divider()
        st.subheader("2. Hyperparameters")
        classification_mode = st.selectbox("Classification Mode", ["binary", "multiclass"], index=0)
        top_k_features = st.slider("Mutual Info Top-k Features", min_value=5, max_value=30, value=15, step=1)
        cv_folds = st.slider("Stratified CV Folds", min_value=3, max_value=10, value=5, step=1)
        active_models = st.multiselect(
            "Active Classifiers",
            ["random_forest", "xgboost", "lightgbm"],
            default=["random_forest", "xgboost", "lightgbm"],
        )

        st.divider()
        st.subheader("3. Pipeline Execution")
        run_button = st.button("🚀 Run Experiment Pipeline", type="primary")
        refresh_button = st.button("🔄 Refresh Artifacts")

        if refresh_button:
            st.cache_data.clear()
            st.rerun()

        log_placeholder = st.empty()

        if run_button:
            pipeline_args = []
            if data_source_mode == "Generate Synthetic Sample":
                pipeline_args.append("--generate-sample")
            elif selected_dataset_path:
                pipeline_args.extend(["--data-path", str(selected_dataset_path)])

            pipeline_args.extend(["--mode", classification_mode, "--k-features", str(top_k_features)])

            success = run_pipeline_process(pipeline_args, log_placeholder)
            if success:
                st.rerun()

    # --- MAIN TABS ---
    tab_overview, tab_fs, tab_models, tab_calib, tab_shap, tab_robust, tab_stats, tab_report = st.tabs([
        "📊 Overview",
        "🔍 Feature Selection",
        "📈 Modeling & Benchmarks",
        "🎯 Calibration",
        "💡 SHAP Explainability",
        "🛡️ Robustness",
        "📐 Hypothesis Testing",
        "📝 Full Report",
    ])

    # ==========================================
    # TAB 1: OVERVIEW & SYSTEM ARCHITECTURE
    # ==========================================
    with tab_overview:
        st.subheader("System Architecture & Pipeline Flow")

        col1, col2, col3, col4 = st.columns(4)
        if has_results:
            n_models = len(reports.get("cv_training_summary", pd.DataFrame()))
            bench_df = reports.get("efficiency_benchmarks", pd.DataFrame())
            n_samples = int(bench_df["n_train_samples"].iloc[0] + bench_df["n_test_samples"].iloc[0]) if not bench_df.empty else "N/A"
            k_feats = int(bench_df["n_selected_features"].iloc[0]) if not bench_df.empty else "N/A"

            col1.metric("Ensemble Models", f"{n_models} Trained", "RF, XGB, LGBM")
            col2.metric("Dataset Size", f"{n_samples:,} Rows" if isinstance(n_samples, int) else n_samples, "80/20 Stratified Split")
            col3.metric("Selected Features", f"{k_feats} Features", "Mutual Information")
            col4.metric("Pipeline Phases", "8 Completed", "Leak-free Stratification")
        else:
            col1.metric("Status", "Awaiting Execution")
            col2.metric("Models", "3 Configured")
            col3.metric("Features", f"Top {top_k_features}")
            col4.metric("CV Folds", f"{cv_folds} Folds")

        st.markdown("---")

        st.markdown("### 🏛️ The 8-Phase Research Methodology")
        st.markdown(
            """
            This framework implements a standardized, reproducible research methodology for cyber-attack detection:
            1. **Phase 1: Leak-Free Preprocessing** — Cleans infinities/NaNs, applies Eq (1) median imputation, Eq (2) skewness-aware $\\log(1+x)$ transformations, and Eq (3) $[0, 1]$ Min-Max scaling fitted strictly on training folds.
            2. **Phase 2: Feature Selection** — Computes Algorithm 1 Mutual Information scores $I(X_i; Y)$ to select top-$k$ predictive network flow signals.
            3. **Phase 3: Stratified K-Fold CV Ensembles** — Fits Random Forest, XGBoost, and LightGBM models with out-of-fold log-loss validation.
            4. **Phase 4: Benchmarking & Calibration** — Profiles memory, latency, throughput, and tests reliability via Platt scaling and Isotonic regression.
            5. **Phase 5: SHAP Explainability & Stability** — Generates TreeExplainer attributions, verifies additivity, and evaluates rank stability (Spearman $\\rho$, Kendall $\\tau$, Jaccard) across folds.
            6. **Phase 6: Robustness Suite** — Evaluates resilience against Gaussian noise, feature corruption, and simulated packet dropouts.
            7. **Phase 7: Statistical Hypothesis Testing** — Conducts Bootstrap 95% CIs, McNemar's paired test, Wilcoxon signed-rank test, and Friedman omnibus analysis.
            8. **Phase 8: Automated Publication Reporting** — Generates 300-DPI visual artifacts and comprehensive Markdown/LaTeX technical summaries.
            """
        )

        # Show dataset preview if a file is available
        if selected_dataset_path and selected_dataset_path.exists():
            st.markdown("### 📂 Active Dataset Sample Preview")
            try:
                preview_df = pd.read_csv(selected_dataset_path, nrows=5)
                st.dataframe(preview_df)
            except Exception as e:
                st.warning(f"Could not preview dataset: {e}")

    # ==========================================
    # TAB 2: FEATURE SELECTION (PHASES 1-2)
    # ==========================================
    with tab_fs:
        st.subheader("Mutual Information Feature Ranking (Algorithm 1)")
        if "feature_importance_mi" in reports:
            mi_df = reports["feature_importance_mi"].copy()
            # Determine score column (Mutual_Information or MI_Score)
            score_col = "Mutual_Information" if "Mutual_Information" in mi_df.columns else (
                "MI_Score" if "MI_Score" in mi_df.columns else mi_df.columns[1]
            )
            mi_df = mi_df.sort_values(by=score_col, ascending=False).reset_index(drop=True)

            col_left, col_right = st.columns([1.2, 1])
            with col_left:
                st.markdown("**Top Selected Features Ranked by Mutual Information $I(X_i; Y)$:**")
                st.bar_chart(mi_df.set_index("Feature")[score_col], color="#1d3557")
            with col_right:
                st.markdown("**Feature Scores Table:**")
                st.dataframe(mi_df, height=350)
        else:
            st.info("Run the pipeline to inspect Mutual Information feature selection results.")

    # ==========================================
    # TAB 3: MODEL PERFORMANCE & BENCHMARKS
    # ==========================================
    with tab_models:
        st.subheader("Cross-Validation & Test Set Evaluation")
        if "cv_training_summary" in reports:
            cv_df = reports["cv_training_summary"]
            st.markdown("##### 5-Fold Stratified Cross-Validation Log-Loss")
            st.dataframe(cv_df)

        col_left, col_right = st.columns(2)
        with col_left:
            roc_plot = plots_dir / "roc_pr_curves.png"
            if roc_plot.exists():
                st.image(str(roc_plot), caption="Test Set ROC & Precision-Recall Curves")
        with col_right:
            cm_plot = plots_dir / "confusion_matrices.png"
            if cm_plot.exists():
                st.image(str(cm_plot), caption="Test Set Confusion Matrices")

        st.markdown("---")
        st.subheader("⚡ Computational Efficiency Benchmarking")
        if "efficiency_benchmarks" in reports:
            eff_df = reports["efficiency_benchmarks"]
            key_cols = [
                "model_name", "training_time_sec", "inference_latency_ms_per_sample",
                "inference_throughput_samples_per_sec", "peak_memory_mb", "model_size_kb"
            ]
            avail_cols = [c for c in key_cols if c in eff_df.columns]
            display_eff = eff_df[avail_cols].copy()

            rename_map = {
                "model_name": "Model",
                "training_time_sec": "Train Time (s)",
                "inference_latency_ms_per_sample": "Latency (ms/sample)",
                "inference_throughput_samples_per_sec": "Throughput (samples/s)",
                "peak_memory_mb": "Peak RAM (MB)",
                "model_size_kb": "Size (KB)",
            }
            display_eff = display_eff.rename(columns=rename_map)
            st.dataframe(display_eff)

            if "Throughput (samples/s)" in display_eff.columns and "Model" in display_eff.columns:
                st.markdown("**Inference Throughput Comparison (Samples per Second):**")
                st.bar_chart(display_eff.set_index("Model")["Throughput (samples/s)"], color="#457b9d")
        else:
            st.info("Run the pipeline to view computational efficiency benchmarks.")

    # ==========================================
    # TAB 4: CALIBRATION ANALYSIS (PHASE 4)
    # ==========================================
    with tab_calib:
        st.subheader("Probability Calibration & Reliability Diagnostics")
        st.markdown(
            "Evaluates whether predicted probabilities accurately reflect actual empirical threat frequencies. "
            "Includes uncalibrated baselines, **Platt Scaling** (Logistic Sigmoid), and **Isotonic Regression**."
        )

        col_cal_plot, col_cal_table = st.columns([1.2, 1])
        with col_cal_plot:
            cal_plot = plots_dir / "calibration_curves.png"
            if cal_plot.exists():
                st.image(str(cal_plot), caption="Reliability Curves (Uncalibrated vs Calibrated)")
            else:
                st.info("Calibration curve plot not found.")

        with col_cal_table:
            if "calibration_metrics" in reports:
                cal_df = reports["calibration_metrics"]
                st.markdown("**Brier Score & Expected Calibration Error (ECE):**")
                st.dataframe(cal_df)
            else:
                st.info("Calibration metrics not found.")

    # ==========================================
    # TAB 5: SHAP EXPLAINABILITY & STABILITY (PHASE 5)
    # ==========================================
    with tab_shap:
        st.subheader("Global & Local Feature Attribution via TreeExplainer")

        if "shap_global_importance" in reports:
            shap_df = reports["shap_global_importance"]
            st.markdown("##### Global Feature Importance (Mean Absolute SHAP Value)")

            models_in_shap = shap_df["Model"].unique().tolist()
            selected_shap_model = st.selectbox("Select Model for Global Attribution", models_in_shap, index=0)
            model_shap_df = shap_df[shap_df["Model"] == selected_shap_model].sort_values(by="Mean_Absolute_SHAP", ascending=False)

            st.bar_chart(model_shap_df.set_index("Feature")["Mean_Absolute_SHAP"], color="#e63946")

            st.markdown("---")
            st.subheader("🔎 Interactive Local Instance Attribution")
            st.markdown(
                "Select a specific network flow instance to explain *why* the model made its classification decision. "
                "Red bars indicate features pushing towards Attack classification; dark blue bars indicate features pushing towards Benign traffic."
            )

            sample_dataset_path = selected_dataset_path if selected_dataset_path and selected_dataset_path.exists() else Path("./dataset/cicids2017_sample.csv")
            if not sample_dataset_path.exists():
                sample_dataset_path = Path("./dataset/cicids2017_generated_sample.csv")

            if sample_dataset_path.exists():
                try:
                    df_sample_inspect = pd.read_csv(sample_dataset_path)
                    target_candidate = [c for c in ["Label", "label", "class", "target"] if c in df_sample_inspect.columns]
                    t_col = target_candidate[0] if target_candidate else df_sample_inspect.columns[-1]

                    X_inspect = df_sample_inspect.drop(columns=[t_col]).select_dtypes(include=[np.number])
                    if "feature_importance_mi" in reports:
                        top_feats = reports["feature_importance_mi"]["Feature"].tolist()[:15]
                        top_feats = [f for f in top_feats if f in X_inspect.columns]
                        X_inspect = X_inspect[top_feats]

                    col_pick1, col_pick2, col_pick3 = st.columns([1, 1, 1])
                    with col_pick1:
                        instance_idx = st.number_input("Sample Row Index", min_value=0, max_value=min(len(X_inspect)-1, 1000), value=0, step=1)
                    with col_pick2:
                        local_model_name = st.selectbox("Classifier", ["xgboost", "random_forest", "lightgbm"], index=0)
                    with col_pick3:
                        top_n_local = st.slider("Top Attributions", min_value=5, max_value=15, value=10)

                    model_ckpt = load_model_checkpoint(models_dir / f"{local_model_name}.joblib")
                    if model_ckpt is not None:
                        from src.explainability.shap_explainer import ShapExplainer
                        explainer = ShapExplainer(max_background_samples=50, max_eval_samples=50)
                        raw_model = getattr(model_ckpt, "model", model_ckpt)

                        single_x = X_inspect.iloc[[instance_idx]]
                        exp_res = explainer.explain(raw_model, single_x, model_name=local_model_name)

                        prob = float(model_ckpt.predict_proba(single_x)[0, 1]) if hasattr(model_ckpt, "predict_proba") else 0.5
                        shap_row = exp_res.shap_values_raw[0] if exp_res.shap_values_raw.ndim > 1 else exp_res.shap_values_raw

                        fig_local = plot_local_shap_waterfall(
                            feature_names=list(single_x.columns),
                            shap_values=shap_row,
                            feature_values=single_x.iloc[0].values,
                            base_value=exp_res.base_value,
                            predicted_prob=prob,
                            model_name=local_model_name,
                            top_n=top_n_local,
                        )
                        st.pyplot(fig_local)
                    else:
                        st.info(f"Model checkpoint for `{local_model_name}` not found in `{models_dir}`.")
                except Exception as e:
                    st.warning(f"Could not compute interactive local explanation: {e}")

            st.markdown("---")
            st.subheader("Explanation Stability Across CV Folds")
            if "explanation_stability" in reports:
                stab_df = reports["explanation_stability"]
                st.dataframe(stab_df)
            else:
                st.info("Explanation stability summary not found.")
        else:
            st.info("Run the pipeline to inspect SHAP explainability and stability metrics.")

    # ==========================================
    # TAB 6: ROBUSTNESS SUITE (PHASE 6)
    # ==========================================
    with tab_robust:
        st.subheader("Robustness & Perturbation Stress Testing")
        st.markdown(
            "Quantifies how resilient classifiers remain when subject to adversarial packet noise, "
            "random missing telemetry values, and targeted sensor corruption."
        )

        col_rob_plot, col_rob_table = st.columns([1.2, 1])
        with col_rob_plot:
            rob_plot = plots_dir / "robustness_degradation.png"
            if rob_plot.exists():
                st.image(str(rob_plot), caption="Performance Degradation Curves Under Stress")
            else:
                st.info("Robustness degradation plot not found.")

        with col_rob_table:
            if "robustness_degradation" in reports:
                rob_df = reports["robustness_degradation"]
                st.markdown("**Degradation Log:**")
                avail_cols = [c for c in ["model_name", "experiment_type", "severity_level", "accuracy", "f1", "mcc", "accuracy_degradation"] if c in rob_df.columns]
                st.dataframe(rob_df[avail_cols], height=400)
            else:
                st.info("Robustness degradation data not found.")

    # ==========================================
    # TAB 7: HYPOTHESIS TESTING (PHASE 7)
    # ==========================================
    with tab_stats:
        st.subheader("Statistical Rigor & Significance Analysis")

        col_boot, col_pairs = st.columns([1, 1.3])
        with col_boot:
            st.markdown("##### Bootstrap 95% Confidence Intervals")
            if "bootstrap_ci" in reports:
                boot_df = reports["bootstrap_ci"]
                st.dataframe(boot_df)
            else:
                st.info("Bootstrap CI table not found.")

        with col_pairs:
            st.markdown("##### Paired Statistical Hypothesis Tests")
            if "statistical_comparisons" in reports:
                stat_df = reports["statistical_comparisons"]
                disp_cols = [c for c in ["test_name", "model_a", "model_b", "p_value", "is_statistically_significant", "interpretation"] if c in stat_df.columns]
                disp_stat = stat_df[disp_cols].copy()
                if "p_value" in disp_stat.columns:
                    disp_stat["p_value"] = disp_stat["p_value"].apply(lambda p: format_p_value(float(p)))
                st.dataframe(disp_stat)
            else:
                st.info("Statistical comparisons not found.")

    # ==========================================
    # TAB 8: EXPERIMENT REPORT (PHASE 8)
    # ==========================================
    with tab_report:
        st.subheader("Comprehensive Research Experiment Report")
        report_md_path = reports_dir / "experiment_report.md"

        if report_md_path.exists():
            with open(report_md_path, "r", encoding="utf-8") as f:
                report_content = f.read()

            st.download_button(
                label="📥 Download Experiment Report (Markdown)",
                data=report_content,
                file_name="experiment_report.md",
                mime="text/markdown",
            )
            st.markdown("---")
            st.markdown(report_content)
        else:
            st.info("Experiment report markdown not found. Run the pipeline to generate it.")


if __name__ == "__main__":
    main()
