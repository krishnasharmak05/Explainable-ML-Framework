# Research-Grade Explainable ML Framework (Phases 1 - 8)

A modular, reusable, and **dataset-agnostic** machine learning research framework for supervised tabular classification, adhering to strict data-leakage prevention, comprehensive statistical comparisons, and explainability standards.

---

## 1. Final Repository Structure

```
predictive-project/
├── app.py                         # Interactive Streamlit Research Web Interface
├── config.yaml                    # Master pipeline configuration
├── dataset/                       # Dataset folder (CSV files)
├── outputs/
│   ├── models/                    # Serialized model checkpoints (.joblib)
│   ├── plots/                     # Publication-quality figures (300 DPI)
│   └── reports/                   # Performance, benchmarks, SHAP, stability, and research reports
├── src/
│   ├── config.py                  # Dataclass configuration loader
│   ├── utils.py                   # Data loader, logger, reproducibility seed
│   ├── sample_data.py             # Realistic synthetic data generator with anomalies
│   ├── cli.py                     # CLI tool (validate, run, evaluate, explain, report)
│   │
│   ├── ui/                        # Web Dashboard Helpers
│   │   └── dashboard_helpers.py   # Cached report loading, local SHAP waterfall, formatting
│   │
│   ├── preprocessing/             # PHASE 1: Data Preprocessing & Normalization
│   │   └── cleaner.py             # Inf resolution, mean/median imputation, log1p, Min-Max [0, 1]
│   │
│   ├── feature_selection/         # PHASE 2/3: Feature Selection
│   │   └── mutual_info.py         # Algorithm 1: Top-k Mutual Information selection
│   │
│   ├── modeling/                  # PHASE 2/3: Predictive Modeling Framework
│   │   ├── base.py                # Standardized BaseModelWrapper interface
│   │   ├── ensemble.py            # Random Forest, XGBoost, LightGBM wrappers
│   │   └── trainer.py             # Algorithm 2: Stratified K-Fold CV with fold model tracking
│   │
│   ├── benchmarking/              # PHASE 4: Computational Benchmarking
│   │   └── resource_tracker.py    # Latency, throughput, peak memory (psutil), model size
│   │
│   ├── calibration/               # PHASE 4: Probability Calibration Analysis
│   │   ├── metrics.py             # Brier score, Expected Calibration Error (ECE), calibration curve
│   │   └── calibrator.py          # Platt scaling (sigmoid), Isotonic regression, uncalibrated baseline
│   │
│   ├── explainability/            # PHASE 5: SHAP & Explanation Stability
│   │   ├── shap_explainer.py      # TreeExplainer, global attributions, local sample attribution
│   │   └── stability.py           # Stability across CV folds: Spearman rho, Kendall tau, Jaccard
│   │
│   ├── robustness/                # PHASE 6: Robustness & Perturbations
│   │   ├── perturbations.py       # Gaussian noise, random missingness, feature corruption, sample fractions
│   │   └── evaluator.py           # Robustness degradation evaluation across metric suite
│   │
│   ├── statistics/                # PHASE 7: Statistical Analysis & Hypothesis Testing
│   │   ├── bootstrap.py           # Non-parametric bootstrap percentile confidence intervals
│   │   ├── hypothesis_tests.py    # Paired McNemar's test & Wilcoxon signed-rank test with effect sizes
│   │   └── omnibus.py             # Friedman omnibus test & Nemenyi post-hoc Critical Difference
│   │
│   └── reporting/                 # PHASE 8: Publication Reports & Visualizations
│       ├── plots.py               # ROC/PR curves, confusion matrices, calibration, stability, robustness
│       ├── tables.py              # Native Markdown & LaTeX table generators
│       └── report_generator.py    # Automated comprehensive research report with Section 9 synthesis
│
├── tests/                         # Automated Unit & Integration Tests (24 tests)
│   ├── test_preprocessing.py
│   ├── test_feature_selection.py
│   ├── test_modeling.py
│   ├── test_benchmarking.py
│   ├── test_calibration.py
│   ├── test_explainability.py
│   ├── test_robustness.py
│   ├── test_statistics.py
│   └── test_reporting.py
│
├── docs/                          # Comprehensive Research Documentation
│   ├── architecture.md            # Software architecture & Mermaid data flow
│   ├── methodology.md             # Theoretical formulations & leakage prevention
│   └── experiments.md             # Guide to reproducing experiments & adding datasets
│
├── run_pipeline.py                # Master CLI runner for Phases 1-8
├── main.py                        # Default entrypoint
└── pyproject.toml                 # Managed with uv
```

---

## 2. Summary of Implemented Modules

| Module | Phase | Key Capabilities |
|---|---|---|
| **Data Preprocessing & Normalization** | Phase 1 | Capping/resolving $\pm\infty$ to `NaN`, mean/median imputation (Eq. 1), $\log(X+1)$ for skewed features (Eq. 2), Min-Max scaling to $[0, 1]$ (Eq. 3), binary and multi-class label encoding. Leak-free scikit-learn API. |
| **Feature Selection** | Phase 2/3 | Algorithm 1: Non-parametric Mutual Information ($I(f; Y)$) ranking and top-$k$ feature subset extraction without test leakage. |
| **Predictive Modeling Framework** | Phase 2/3 | Algorithm 2: Standardized ensemble wrappers (Random Forest, XGBoost, LightGBM) evaluated via Stratified $k$-Fold CV with fold model tracking and log-loss calculation. |
| **Computational Benchmarking** | Phase 4 | Distinguishes preprocessing, feature selection, training, and inference time; calculates latency per sample (ms), throughput (samples/sec), peak memory (`psutil`), and model file size. |
| **Probability Calibration** | Phase 4 | Evaluates Brier score and Expected Calibration Error (ECE). Compares uncalibrated predictions with Platt scaling (`sigmoid`) and Isotonic regression (`isotonic`). Publication reliability diagrams omit empty-bin distortions and support quantile equal-frequency binning. |
| **SHAP Explainability** | Phase 5 | Model-aware `TreeExplainer` computing global mean absolute SHAP values, feature importance rankings, interactive single-instance local explanations, and verification of additive attributions across all tree models. |
| **Explanation Stability** | Phase 5 | Measures attribution stability across CV folds: computes pairwise Spearman rank correlation ($\rho$), Kendall rank correlation ($\tau$), and Jaccard similarity ($J$) of top-$k$ features. |
| **Robustness Suite** | Phase 6 | Evaluates metric degradation curves under controlled Gaussian feature noise ($\sigma \in [0.01, 0.20]$), random missingness, top-feature corruption, and training sample subsampling. |
| **Statistical Analysis** | Phase 7 | Non-parametric percentile bootstrap confidence intervals ($B=1000$ resamples), paired McNemar's test, Wilcoxon signed-rank test with effect sizes, Friedman omnibus test, and Nemenyi post-hoc testing. |
| **Reporting & Visualizations** | Phase 8 | Automated generation of 300-DPI publication plots (ROC/PR, confusion matrices, multi-panel calibration curves, robustness curves) and structured experiment report (`outputs/reports/experiment_report.md`) with dynamic Section 9 synthesis and deployment matrix. |
| **Streamlit Web Dashboard** | Interactive | Full-featured 8-tab web interface (`app.py`) for live pipeline runs, real-time local SHAP instance attribution, diagnostic curve browsing, and one-click report export. |

---

## 3. Quickstart: Running the Synthetic Example

To verify the end-to-end pipeline immediately without any external dataset:
```powershell
uv run python run_pipeline.py --generate-sample
```
This generates a synthetic dataset with realistic network anomalies (infs, NaNs, skewness, benign/attack classes) at `./dataset/cicids2017_sample.csv` and executes Phases 1 through 8.

---

## 4. Interactive Web Interface (Streamlit)

A full-featured, research-grade web dashboard is provided via Streamlit:

```powershell
uv run streamlit run app.py
```

### Dashboard Capabilities:
- **Pipeline Controls**: Ingest custom CSVs or synthetic benchmarks, tune hyperparameters ($k$ features, CV folds, active classifiers), and execute the pipeline with live streaming terminal logs.
- **Interactive SHAP Attributions**: Select individual test flow observations to generate real-time local explanation charts with directional feature impacts (crimson for attack push, navy for benign push).
- **Reliability Diagnostics**: Inspect reliability curves and Brier/ECE calibrations across Platt and Isotonic scaling.
- **Robustness Degradation**: Examine real-time stress testing curves under Gaussian noise, missingness injection, and top-feature corruption.
- **Statistical Significance Tables**: View bootstrap 95% confidence intervals and paired hypothesis test decisions (McNemar, Wilcoxon, Friedman) with automatic effect size interpretations.
- **One-Click Report Export**: View and download the comprehensive research experiment report in Markdown format.

---

## 5. Instructions for Adding a Real Dataset

The framework contains no hardcoded assumptions, column names, or dataset-specific terminology:
1. **Place your dataset**: Put your CSV or Parquet file into `./dataset/` (e.g., `./dataset/my_data.csv`).
2. **Configure `config.yaml`**:
   ```yaml
   data:
     dataset_path: "./dataset/my_data.csv"
     target_column: "Label"                  # Your label column name
     classification_mode: "binary"           # "binary" or "multiclass"
     benign_label: "BENIGN"                  # Value of the negative/benign class
   feature_selection:
     top_k: 15                               # Number of features to select
   modeling:
     cv_folds: 5
     active_models:
       - "random_forest"
       - "xgboost"
       - "lightgbm"
   ```
3. **Validate dataset hygiene**:
   ```powershell
   uv run python -m src.cli validate --config config.yaml
   ```
4. **Execute complete research pipeline**:
   ```powershell
   uv run python -m src.cli run --config config.yaml
   ```

For a comprehensive walkthrough on customizing parameters, running scenario evaluations, and interpreting all machine-readable outputs, refer to the [Experiment Guide](docs/experiments.md).

---

## 5. Scientifically Important Caveats & Limitations

1. **Leakage Safeguards**: All preprocessing, imputation, scaling, and feature selection must be fitted strictly inside training folds. The framework enforces this design to avoid overestimating generalization.
2. **Statistical vs. Practical Significance**: A statistically significant difference ($p < 0.05$ on McNemar or Wilcoxon test) does not necessarily imply a meaningful operational advantage. Effect sizes (Odds Ratio, Rank-Biserial correlation) and confidence intervals must be reported alongside p-values.
3. **SHAP Semantics**: SHAP feature attributions reflect feature importance within the mathematical structure of the learned model relative to a baseline expectation, not direct causal mechanisms in the underlying real-world data generation process.
4. **Imbalanced Data Metrics**: Accuracy is misleading on heavily imbalanced tabular datasets; decisions should prioritize MCC, Precision-Recall AUC, and F1.
5. **Distribution Shift**: Synthetic perturbation tests (Gaussian noise, missingness) test model sensitivity to data corruption, but real-world operational domain shifts (e.g. new network protocols or evolving attack vectors) may exhibit complex structured or temporal shifts.

---

## 6. Test Suite & Verification Results

Run the complete automated test suite:
```powershell
uv run pytest tests/ -v
```

**Results**: All **23 tests passed** across all modules:
- Preprocessing: `test_missing_and_infinite_value_handling`, `test_log_transformation`, `test_label_encoding`, `test_preprocessor_save_load` (4 passed)
- Feature Selection: `test_mutual_info_selection`, `test_selector_save_load` (2 passed)
- Modeling: `test_ensemble_model_wrappers`, `test_model_trainer_cv_and_algorithm2` (2 passed)
- Benchmarking: `test_resource_tracker_inference`, `test_resource_tracker_memory`, `test_efficiency_report_generation` (3 passed)
- Calibration: `test_calibration_metrics`, `test_model_calibrator` (2 passed)
- Explainability: `test_shap_explainer`, `test_explanation_stability_analyzer` (2 passed)
- Robustness: `test_perturbation_functions`, `test_robustness_evaluator` (2 passed)
- Statistics: `test_bootstrap_ci`, `test_mcnemar_test`, `test_wilcoxon_test`, `test_friedman_and_nemenyi` (4 passed)
- Reporting: `test_plot_generation`, `test_table_and_report_generation` (2 passed)

---

## 7. Documentation & Deep Dives

For deeper technical explorations, consult the detailed documentation in the `docs/` directory:
- [Architecture & System Design](docs/architecture.md): Conceptual data flow, component decoupling, and Mermaid diagrams.
- [Methodology & Formulations](docs/methodology.md): Mathematical formulations, algorithms, and data leakage safeguards.
- [Experiment Guide](docs/experiments.md): In-depth guide to experiment reproduction, dataset validation, and output report analysis.
