import argparse
from dataclasses import asdict
import logging
from pathlib import Path
import sys
import time
import pandas as pd
from sklearn.model_selection import train_test_split

from src.benchmarking.resource_tracker import EfficiencyReport, ResourceTracker
from src.calibration.calibrator import ModelCalibrator
from src.config import PipelineConfig
from src.explainability.shap_explainer import ShapExplainer
from src.explainability.stability import ExplanationStabilityAnalyzer
from src.feature_selection.mutual_info import MIFeatureSelector
from src.modeling.ensemble import get_model
from src.modeling.trainer import ModelTrainer
from src.preprocessing.cleaner import DataPreprocessor
from src.robustness.evaluator import RobustnessEvaluator
from src.sample_data import generate_cicids2017_sample
from src.utils import load_dataset, set_seed, setup_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dataset-Agnostic Explainable ML Research Pipeline (Phases 1-6)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to pipeline configuration YAML file",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Override path to dataset directory or CSV file",
    )
    parser.add_argument(
        "--k-features",
        type=int,
        default=None,
        help="Override top-k features to select with Mutual Information",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["binary", "multiclass"],
        default=None,
        help="Override classification mode ('binary' or 'multiclass')",
    )
    parser.add_argument(
        "--generate-sample",
        action="store_true",
        help="Generate synthetic dataset in ./dataset/ if no data exists",
    )
    return parser.parse_args()


def run():
    args = parse_args()

    # 1. Load configuration
    cfg = PipelineConfig.from_yaml(args.config)
    if args.data_path:
        cfg.data.dataset_path = args.data_path
    if args.k_features:
        cfg.feature_selection.top_k = args.k_features
    if args.mode:
        cfg.data.classification_mode = args.mode

    # Setup directories and logger
    reports_dir = Path(cfg.output.reports_dir)
    models_dir = Path(cfg.output.models_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("pipeline", log_dir=cfg.output.output_dir)
    logger.info("=" * 70)
    logger.info("DATASET-AGNOSTIC EXPLAINABLE ML PIPELINE (PHASES 1 - 6)")
    logger.info("=" * 70)
    logger.info(f"Classification Mode:   {cfg.data.classification_mode.upper()}")
    logger.info(f"Mutual Info Top-k:     {cfg.feature_selection.top_k}")
    logger.info(f"CV Folds:              {cfg.modeling.cv_folds}")
    logger.info(f"Benchmarking Enabled:  {cfg.benchmarking.enabled}")
    logger.info(f"Calibration Enabled:   {cfg.calibration.enabled}")
    logger.info(f"Explainability/SHAP:   {cfg.explainability.enabled}")
    logger.info(f"Robustness Suite:      {cfg.robustness.enabled}")

    set_seed(cfg.data.random_state)
    resource_tracker = ResourceTracker()
    resource_tracker.start_memory_tracking()

    # 2. Dataset Loading / Preparation
    data_path = Path(cfg.data.dataset_path)
    if not data_path.exists() or (data_path.is_dir() and not list(data_path.glob("*.csv"))):
        if args.generate_sample:
            logger.warning(f"Dataset path {data_path} is empty or not found. Generating synthetic sample...")
            sample_file = data_path / "cicids2017_sample.csv" if data_path.is_dir() else Path("./dataset/cicids2017_sample.csv")
            generate_cicids2017_sample(n_samples=5000, output_path=sample_file, random_state=cfg.data.random_state)
            data_path = sample_file
        else:
            logger.error(f"Dataset path '{data_path}' not found or contains no CSVs.")
            logger.error("Please place CSV files into './dataset/' or run with '--generate-sample'.")
            sys.exit(1)

    df_raw = load_dataset(data_path, logger=logger)

    target_col = cfg.data.target_column
    if target_col not in df_raw.columns:
        matched_col = [c for c in df_raw.columns if c.strip().lower() == target_col.lower()]
        if matched_col:
            target_col = matched_col[0]
        else:
            raise KeyError(f"Target column '{target_col}' not found in dataset columns: {list(df_raw.columns)}")

    X_raw = df_raw.drop(columns=[target_col])
    y_raw = df_raw[target_col]

    # 3. Module 1: Preprocessing & Normalization
    logger.info(">>> Running Module 1: Data Preprocessing & Normalization...")
    t0_prep = time.perf_counter()

    preprocessor = DataPreprocessor(
        imputation_strategy=cfg.preprocessing.imputation_strategy,
        skewness_threshold=cfg.preprocessing.skewness_threshold,
        scaling_range=cfg.preprocessing.scaling_range,
        drop_columns=cfg.preprocessing.drop_columns,
        classification_mode=cfg.data.classification_mode,
        benign_label=cfg.data.benign_label,
    )

    y_encoded = preprocessor.encode_labels(y_raw, fit=True)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw,
        y_encoded,
        test_size=cfg.data.test_size,
        random_state=cfg.data.random_state,
        stratify=y_encoded,
    )

    logger.info(f"Train split: {len(X_train_raw):,} samples | Test split: {len(X_test_raw):,} samples")

    preprocessor.fit(X_train_raw)
    X_train_prep = preprocessor.transform(X_train_raw)
    X_test_prep = preprocessor.transform(X_test_raw)

    time_prep = time.perf_counter() - t0_prep
    preprocessor_path = models_dir / "preprocessor.joblib"
    preprocessor.save(preprocessor_path)
    logger.info(f"Module 1 complete in {time_prep:.2f}s. Skewed features: {len(preprocessor.skewed_columns_)}")

    # 4. Module 2: Feature Selection (Mutual Information)
    logger.info(f">>> Running Module 2: Feature Selection (Top-k={cfg.feature_selection.top_k})...")
    t0_fs = time.perf_counter()

    selector = MIFeatureSelector(
        top_k=cfg.feature_selection.top_k,
        n_neighbors=cfg.feature_selection.n_neighbors,
        sample_size=cfg.feature_selection.sample_size,
        random_state=cfg.data.random_state,
    )
    selector.fit(X_train_prep, y_train)
    X_train_selected = selector.transform(X_train_prep)
    X_test_selected = selector.transform(X_test_prep)

    time_fs = time.perf_counter() - t0_fs
    scores_path = reports_dir / "feature_importance_mi.csv"
    selector.save_scores(scores_path)
    selector_path = models_dir / "feature_selector.joblib"
    selector.save(selector_path)

    logger.info(f"Module 2 complete in {time_fs:.2f}s. Selected {len(selector.selected_features_)} features.")
    logger.info(f"Top 5 features: {selector.selected_features_[:5]}")

    # 5. Module 3: Predictive Modeling Framework
    logger.info(">>> Running Module 3: Predictive Modeling Framework (Algorithm 2)...")
    model_instances = []
    for model_name in cfg.modeling.active_models:
        params = getattr(cfg.modeling, model_name, {})
        model_instances.append(get_model(model_name, params=params))

    trainer = ModelTrainer(
        cv_folds=cfg.modeling.cv_folds,
        random_state=cfg.data.random_state,
        logger=logger,
    )

    training_results = trainer.train_pipeline(
        X_train=X_train_selected,
        y_train=y_train,
        models=model_instances,
        X_test=X_test_selected,
        save_dir=models_dir,
    )

    summary_df = training_results.summary()
    summary_csv = reports_dir / "cv_training_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    logger.info(f"Saved CV summary to {summary_csv}")

    # 6. Phase 4: Computational Benchmarking & Resource Tracking
    peak_memory = resource_tracker.stop_memory_tracking()
    efficiency_reports: List[EfficiencyReport] = []

    if cfg.benchmarking.enabled:
        logger.info(">>> Running Phase 4: Computational Efficiency Benchmarking...")
        for model_name in cfg.modeling.active_models:
            model = training_results.get_model(model_name)
            model_file = models_dir / f"{model_name}.joblib"
            train_time = training_results.results[model_name]["training_time_seconds"]

            perf = ResourceTracker.measure_inference_performance(
                model=model,
                X_test=X_test_selected,
                n_repeats=cfg.benchmarking.n_repeats,
            )

            report = ResourceTracker.create_efficiency_report(
                model_name=model_name,
                preprocessing_time_sec=time_prep,
                feature_selection_time_sec=time_fs,
                training_time_sec=train_time,
                inference_time_sec=perf["inference_time_sec"],
                inference_latency_ms=perf["latency_ms"],
                inference_throughput=perf["throughput"],
                peak_memory_mb=peak_memory,
                model_file_path=model_file,
                n_raw_features=X_raw.shape[1],
                n_selected_features=X_train_selected.shape[1],
                n_train_samples=len(X_train_selected),
                n_test_samples=len(X_test_selected),
            )
            efficiency_reports.append(report)

        bench_df = ResourceTracker.reports_to_dataframe(efficiency_reports)
        bench_csv = reports_dir / "efficiency_benchmarks.csv"
        bench_df.to_csv(bench_csv, index=False)
        logger.info(f"Saved computational benchmarks to {bench_csv}")

    # 7. Phase 4: Probability Calibration Analysis
    if cfg.calibration.enabled:
        logger.info(">>> Running Phase 4: Probability Calibration Analysis (Brier, ECE, Platt, Isotonic)...")
        # Split a validation set from training for leak-free calibrator fitting
        X_cal_tr, X_cal_val, y_cal_tr, y_cal_val = train_test_split(
            X_train_selected,
            y_train,
            test_size=0.25,
            random_state=cfg.data.random_state,
            stratify=y_train,
        )

        all_cal_reports = []
        for model_name in cfg.modeling.active_models:
            base_model = training_results.get_model(model_name)
            calibrator = ModelCalibrator(n_bins=cfg.calibration.n_bins)
            calibrator.fit(
                base_estimator=base_model,
                X_val=X_cal_val,
                y_val=y_cal_val,
                methods=cfg.calibration.methods,
            )
            cal_reports = calibrator.evaluate(
                base_estimator=base_model,
                X_eval=X_test_selected,
                y_eval=y_test,
                model_name=model_name,
            )
            all_cal_reports.extend(cal_reports)

        cal_df = ModelCalibrator.reports_to_dataframe(all_cal_reports)
        cal_csv = reports_dir / "calibration_metrics.csv"
        cal_df.to_csv(cal_csv, index=False)
        logger.info(f"Saved calibration analysis to {cal_csv}")

    # 8. Phase 5: SHAP Explainability & Explanation Stability
    if cfg.explainability.enabled:
        logger.info(">>> Running Phase 5: SHAP Explainability & Explanation Stability...")
        explainer = ShapExplainer(
            max_background_samples=100,
            max_eval_samples=cfg.explainability.max_eval_samples,
            random_state=cfg.data.random_state,
        )

        shap_dfs = []
        for model_name in cfg.modeling.active_models:
            model = training_results.get_model(model_name)
            exp_res = explainer.explain(model, X_test_selected, model_name=model_name)
            df_imp = exp_res.feature_importance_df.copy()
            df_imp["Model"] = model_name
            shap_dfs.append(df_imp)
            logger.info(f"  [{model_name}] SHAP additive property verified: {exp_res.is_additive_verified}")

        all_shap_df = pd.concat(shap_dfs, ignore_index=True)
        shap_csv = reports_dir / "shap_global_importance.csv"
        all_shap_df.to_csv(shap_csv, index=False)
        logger.info(f"Saved SHAP global importance to {shap_csv}")

        # Explanation Stability Analysis across CV Folds
        if cfg.explainability.stability.enabled:
            logger.info("  Calculating Explanation Stability across CV folds...")
            stability_analyzer = ExplanationStabilityAnalyzer(top_k=cfg.explainability.stability.top_k)
            stability_summaries = []

            for model_name in cfg.modeling.active_models:
                fold_models = training_results.get_fold_models(model_name)
                if len(fold_models) >= 2:
                    fold_importances = []
                    for f_idx, f_model in enumerate(fold_models):
                        f_exp = explainer.explain(f_model, X_train_selected.iloc[:150], model_name=f"{model_name}_fold{f_idx}")
                        fold_importances.append(f_exp.feature_importance_df)

                    stab_res = stability_analyzer.analyze_stability(fold_importances, model_name=model_name)
                    stability_summaries.append(stab_res.summary_dict())
                    # Save pairwise Spearman matrix
                    mat_file = reports_dir / f"stability_spearman_{model_name}.csv"
                    stab_res.pairwise_spearman_matrix.to_csv(mat_file)

            if stability_summaries:
                stab_df = pd.DataFrame(stability_summaries)
                stab_csv = reports_dir / "explanation_stability.csv"
                stab_df.to_csv(stab_csv, index=False)
                logger.info(f"Saved explanation stability summary to {stab_csv}")

    # 9. Phase 6: Robustness Experiments
    if cfg.robustness.enabled:
        logger.info(">>> Running Phase 6: Robustness & Perturbation Suite...")
        evaluator = RobustnessEvaluator(random_state=cfg.data.random_state)
        all_robustness_reports = []

        top_feats = selector.selected_features_[:3]
        impute_fn = lambda df_x: preprocessor.transform(df_x)[selector.selected_features_]

        for model_name in cfg.modeling.active_models:
            model = training_results.get_model(model_name)
            rob_reports = evaluator.evaluate_model_robustness(
                model=model,
                X_test=X_test_selected,
                y_test=y_test,
                model_name=model_name,
                noise_levels=cfg.robustness.noise_levels,
                missingness_rates=cfg.robustness.missingness_rates,
                top_features_to_corrupt=top_feats,
            )
            all_robustness_reports.extend(rob_reports)

        rob_df = RobustnessEvaluator.reports_to_dataframe(all_robustness_reports)
        rob_csv = reports_dir / "robustness_degradation.csv"
        rob_df.to_csv(rob_csv, index=False)
        logger.info(f"Saved robustness degradation report to {rob_csv}")

    # 10. Phase 7: Statistical Analysis & Hypothesis Testing
    logger.info(">>> Running Phase 7: Statistical Analysis & Hypothesis Testing...")
    from src.statistics.bootstrap import compute_bootstrap_ci, bootstrap_results_to_dataframe
    from src.statistics.hypothesis_tests import run_mcnemar_test, run_wilcoxon_test
    from src.statistics.omnibus import run_friedman_test, run_nemenyi_posthoc

    # Compute test performance metrics and predictions
    test_metrics_records = []
    models_preds_dict = {}
    models_probs_dict = {}

    for model_name in cfg.modeling.active_models:
        model = training_results.get_model(model_name)
        preds = model.predict(X_test_selected)
        probs = model.predict_proba(X_test_selected)
        models_preds_dict[model_name] = preds
        models_probs_dict[model_name] = probs

        # Point metrics
        is_bin = len(np.unique(y_test)) <= 2
        avg_m = "binary" if is_bin else "weighted"
        from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average=avg_m, zero_division=0)
        mcc = matthews_corrcoef(y_test, preds)
        p1 = probs[:, 1] if is_bin and probs.ndim == 2 else probs
        auc = roc_auc_score(y_test, p1) if is_bin else 0.0

        test_metrics_records.append({
            "Model": model_name,
            "Accuracy": f"{acc:.4f}",
            "F1": f"{f1:.4f}",
            "ROC-AUC": f"{auc:.4f}",
            "MCC": f"{mcc:.4f}",
        })

    perf_df = pd.DataFrame(test_metrics_records)

    # 10.1 Bootstrap Confidence Intervals (First model or best model)
    primary_model = cfg.modeling.active_models[0]
    boot_ci_dict = compute_bootstrap_ci(
        y_true=y_test,
        y_pred=models_preds_dict[primary_model],
        y_prob=models_probs_dict[primary_model],
        n_bootstraps=200,
        random_state=cfg.data.random_state,
    )
    boot_df = bootstrap_results_to_dataframe(boot_ci_dict)
    boot_csv = reports_dir / "bootstrap_ci.csv"
    boot_df.to_csv(boot_csv, index=False)
    logger.info(f"Saved bootstrap confidence intervals to {boot_csv}")

    # 10.2 Paired Comparisons & Omnibus Tests
    stat_test_records = []
    active_m = cfg.modeling.active_models
    for i in range(len(active_m)):
        for j in range(i + 1, len(active_m)):
            m1, m2 = active_m[i], active_m[j]
            # McNemar test on test predictions
            mcn_res = run_mcnemar_test(y_test, models_preds_dict[m1], models_preds_dict[m2], m1, m2)
            stat_test_records.append(asdict(mcn_res) if hasattr(mcn_res, "__dataclass_fields__") else mcn_res.__dict__)

            # Wilcoxon test on CV log-loss folds
            losses_1 = training_results.results[m1]["cv_log_losses"]
            losses_2 = training_results.results[m2]["cv_log_losses"]
            wilc_res = run_wilcoxon_test(losses_1, losses_2, m1, m2)
            stat_test_records.append(asdict(wilc_res) if hasattr(wilc_res, "__dataclass_fields__") else wilc_res.__dict__)

    # Friedman Omnibus Test on CV fold losses
    cv_losses_dict = {m: training_results.results[m]["cv_log_losses"] for m in active_m}
    friedman_res = run_friedman_test(cv_losses_dict, higher_is_better=False)
    logger.info(f"Friedman Omnibus Test p-value: {friedman_res.p_value_f:.5f} (Significant: {friedman_res.is_significant})")

    stat_df = pd.DataFrame(stat_test_records)
    stat_csv = reports_dir / "statistical_comparisons.csv"
    stat_df.to_csv(stat_csv, index=False)
    logger.info(f"Saved statistical comparisons to {stat_csv}")

    # 11. Phase 8: Publication-Quality Reporting & Visualizations
    logger.info(">>> Running Phase 8: Publication-Quality Plots & Research Report...")
    plots_dir = Path(cfg.output.output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    from src.reporting.plots import (
        plot_calibration_curves,
        plot_confusion_matrices,
        plot_metric_comparison,
        plot_robustness_degradation,
        plot_roc_pr_curves,
        plot_stability_heatmap,
    )
    from src.reporting.report_generator import generate_research_report

    # Save Plots
    plot_metric_comparison(perf_df, plots_dir / "metric_comparison.png")
    plot_roc_pr_curves(y_test, models_probs_dict, plots_dir / "roc_pr_curves.png")
    plot_confusion_matrices(y_test, models_preds_dict, plots_dir / "confusion_matrices.png")

    # Calibration curves plot
    from src.calibration.metrics import compute_calibration_curve
    cal_curves = {m: compute_calibration_curve(y_test, models_probs_dict[m]) for m in active_m}
    plot_calibration_curves(cal_curves, plots_dir / "calibration_curves.png")

    # Robustness plot
    plot_robustness_degradation(rob_df, plots_dir / "robustness_degradation.png")
    logger.info(f"Saved publication-quality plots to {plots_dir}")

    # Generate Full Research Markdown Report
    dataset_metadata = {
        "n_samples": len(df_raw),
        "n_raw_features": X_raw.shape[1],
        "n_selected_features": X_train_selected.shape[1],
        "class_distribution": str(dict(df_raw[target_col].value_counts())),
        "infinite_count": int(np.isinf(df_raw.select_dtypes(include=[np.number]).values).sum()) if len(df_raw.select_dtypes(include=[np.number]).columns) > 0 else 0,
        "missing_count": int(df_raw.isna().sum().sum()),
        "imputation_strategy": cfg.preprocessing.imputation_strategy,
        "cv_folds": cfg.modeling.cv_folds,
        "classification_mode": cfg.data.classification_mode,
    }

    report_file = reports_dir / "experiment_report.md"
    generate_research_report(
        dataset_metadata=dataset_metadata,
        config_dict={},
        performance_df=perf_df,
        efficiency_df=bench_df if cfg.benchmarking.enabled else None,
        calibration_df=cal_df if cfg.calibration.enabled else None,
        stability_df=stab_df if cfg.explainability.stability.enabled and "stab_df" in locals() else None,
        robustness_df=rob_df if cfg.robustness.enabled else None,
        statistical_tests=stat_test_records,
        bootstrap_df=boot_df,
        output_path=str(report_file),
    )
    logger.info(f"Saved comprehensive research experiment report to {report_file}")

    logger.info("=" * 70)
    logger.info("ALL PHASES (1 - 8) COMPLETED SUCCESSFULLY!")
    logger.info("=" * 70)


if __name__ == "__main__":
    run()
