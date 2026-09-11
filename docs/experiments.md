# Experimental Guide: Reproducing Experiments & Adding Datasets

This guide explains how to execute experiments, inspect outputs, and adapt the framework to an arbitrary tabular dataset.

---

## 1. Quickstart: Synthetic Tabular Dataset

To verify the end-to-end framework immediately without downloading external data:

```powershell
uv run python run_pipeline.py --generate-sample
```

This generates a representative synthetic dataset at `./dataset/cicids2017_sample.csv` and executes all 8 phases:
- **Phase 1**: Preprocessing, infinite value resolution, missing value imputation, log1p, Min-Max scaling.
- **Phase 2/3**: Top-$k$ Mutual Information feature selection and model cross-validation.
- **Phase 4**: Latency, throughput, memory, disk size profiling, and probability calibration (Platt & Isotonic).
- **Phase 5**: SHAP feature importance attributions and fold-level explanation stability analysis.
- **Phase 6**: Robustness perturbation tests (noise, missingness, feature corruption).
- **Phase 7**: Statistical hypothesis testing (bootstrap CIs, McNemar's, Wilcoxon, Friedman tests).
- **Phase 8**: Publication-quality plots and experiment markdown report.

---

## 2. Adding a Real-World Dataset

The framework is completely **dataset-agnostic**. To benchmark on your own tabular classification dataset:

### Step 1: Place your data
Put your CSV or Parquet file into `./dataset/` (e.g. `./dataset/my_dataset.csv`) or any local folder.

### Step 2: Update `config.yaml`
```yaml
data:
  dataset_path: "./dataset/my_dataset.csv"  # or directory path
  target_column: "Target"                   # Your label column name
  classification_mode: "binary"             # "binary" or "multiclass"
  benign_label: "0"                         # Negative/normal class identifier
  test_size: 0.2
  random_state: 42

feature_selection:
  top_k: 20                                 # Number of features to select

modeling:
  cv_folds: 5
  active_models:
    - "random_forest"
    - "xgboost"
    - "lightgbm"
```

### Step 3: Validate Dataset Hygiene
Run the pre-flight dataset inspector:
```powershell
uv run python -m src.cli validate --config config.yaml
```
This inspects:
- Total rows and columns
- Numerical vs categorical feature counts
- Detected infinite values
- Detected missing values (NaN)
- Duplicate row count
- Target class balance distribution

### Step 4: Execute Full Research Pipeline
```powershell
uv run python -m src.cli run --config config.yaml
```

---

## 3. Interpreting Output Artifacts

All outputs are saved in machine-readable and publication-ready formats:

### Reports Directory (`outputs/reports/`)
- `cv_training_summary.csv`: Fold-level and mean cross-validation log-loss and training time per model.
- `efficiency_benchmarks.csv`: Training time, inference time, latency per sample (ms), throughput (samples/sec), peak memory (MB), and serialized model size on disk (bytes).
- `calibration_metrics.csv`: Brier score and Expected Calibration Error (ECE) comparing uncalibrated predictions with Platt scaling and Isotonic regression.
- `shap_global_importance.csv`: Global SHAP feature attributions and importance ranks.
- `explanation_stability.csv`: Pairwise Spearman $\rho$, Kendall $\tau$, and Jaccard similarity across CV folds.
- `robustness_degradation.csv`: Performance degradation curves under Gaussian noise, missingness, and corruption.
- `bootstrap_ci.csv`: 95% non-parametric bootstrap confidence intervals for all metrics.
- `statistical_comparisons.csv`: Paired McNemar and Wilcoxon test statistics, p-values, and effect sizes.
- `experiment_report.md`: Automated research report with dedicated Section 9 dynamic synthesis and strategic deployment decision matrix.

### Plots Directory (`outputs/plots/`)
- `metric_comparison.png`: Grouped comparative bar chart of Accuracy, F1, ROC-AUC, and MCC.
- `roc_pr_curves.png`: Side-by-side ROC and Precision-Recall curves.
- `confusion_matrices.png`: Normalized confusion matrix heatmaps.
- `calibration_curves.png`: Multi-panel reliability diagrams (Uncalibrated vs Sigmoid vs Isotonic) with quantile binning.
- `robustness_degradation.png`: Degradation curve across perturbation severities.

---

## 4. Interactive Web Interface (Streamlit)

Launch the interactive dashboard to run experiments and explore outputs visually:

```powershell
uv run streamlit run app.py
```

### Dashboard Tabs:
1. **Overview**: Key metrics (models trained, total samples, selected features, pipeline status), live active dataset preview, and 8-phase pipeline architecture flowchart.
2. **Feature Selection**: Top-$k$ Mutual Information feature scores and interactive ranking chart.
3. **Modeling & Benchmarks**: Stratified CV log-loss, ROC/PR curves, confusion matrices, and throughput/latency/RAM/size benchmarking comparisons.
4. **Calibration**: Multi-panel reliability diagrams (Uncalibrated vs Platt Scaling vs Isotonic Regression), Brier score, and ECE reduction tables.
5. **SHAP Explainability**: Global feature importance comparisons across all tree ensembles, **interactive local instance attribution** (inspect any test sample to see why an intrusion alert was triggered), and explanation stability across CV folds.
6. **Robustness**: Real-time stress testing curves under Gaussian noise, random missingness, and top-feature corruption.
7. **Hypothesis Testing**: Bootstrap 95% confidence intervals, paired McNemar and Wilcoxon signed-rank tests with formatted p-values and effect sizes.
8. **Research Report**: Rendered full Markdown report with one-click `.md` download and Section 9 strategic synthesis.
