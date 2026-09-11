# Architecture: Explainable ML Research Framework

## 1. System Overview

The framework is architected around three decoupled layers:
1. **Generic Tabular ML Engine (`src/`)**: A modular, leak-free, dataset-agnostic pipeline implementing preprocessing, feature selection, ensemble modeling, computational benchmarking, probability calibration, SHAP explanations, explanation stability, robustness perturbations, non-parametric hypothesis testing, and automated synthesis reporting.
2. **User Interfaces**: Dual operational surfaces including the master CLI runner (`run_pipeline.py` & `src/cli.py`) and a modern interactive research dashboard (`app.py` & `src/ui/`).
3. **Dataset Configuration (`config.yaml`)**: A lightweight specification defining data paths, target columns, task mode (binary vs multiclass), feature counts, and model hyperparameters.

No dataset-specific assumptions, hardcoded column names, or attack labels exist in the generic framework.

---

## 2. Mermaid Conceptual Data Flow

```mermaid
graph TD
    A[Arbitrary Tabular Dataset CSV] --> B[Data Validation & Schema Inspection]
    B --> C[Stratified Train / Test Split]
    
    subgraph Cross-Validation & Modeling
        C --> D[Stratified K-Fold Splitter]
        D --> E[Data Preprocessor: Inf Resolution, Imputation, Log1p, Min-Max]
        E --> F[Feature Selection: Mutual Information Top-k]
        F --> G[Model Training: Random Forest, XGBoost, LightGBM]
        G --> H[Fold Log-Loss Evaluation & Model Checkpoints]
    end
    
    subgraph Multi-Faceted Evaluation Suite
        G --> I[Computational Benchmarking: Latency, Throughput, Peak Memory, Disk Size]
        G --> J[Probability Calibration: Brier Score, ECE, Platt & Isotonic Regression]
        G --> K[SHAP Explainability: TreeExplainer, Global & Local Attributions]
        K --> L[Explanation Stability: Spearman Rho, Kendall Tau, Jaccard Index]
        G --> M[Robustness Perturbations: Noise, Missingness, Feature Corruption]
    end
    
    subgraph Statistical Analysis & Reporting
        H --> N[Statistical Tests: Bootstrap CIs, McNemar, Wilcoxon, Friedman]
        I --> O[Automated Research Report with Section 9 Synthesis]
        J --> O
        L --> O
        M --> O
        N --> O
    end

    subgraph User Experience & Visualization
        O --> P[Publication Plots: ROC/PR, Multi-Panel Reliability, Confusion Matrices]
        O --> Q[Interactive Streamlit Research Web Interface: app.py]
        P --> Q
    end
```

---

## 3. Component Separation of Concerns

- **`src/ui/`**: Web interface helper components managing cached report ingestion, single-observation directional SHAP attribution waterfalls, and formatted metric badges.
- **`src/preprocessing/`**: Leak-free scikit-learn transformers. Imputation parameters, skewness transformations, and min-max scalers are fitted strictly on the training partition.
- **`src/feature_selection/`**: Mutual Information-based ranking ($I(f; Y)$) and top-$k$ feature extraction without inspecting test labels.
- **`src/modeling/`**: Standardized `BaseModelWrapper` interface for tree ensembles (Random Forest, XGBoost, LightGBM) with Stratified K-Fold CV fold tracking.
- **`src/benchmarking/`**: Real-time process profiling (`psutil`, `tracemalloc`) measuring peak resident memory, inference throughput, latency, and serialization size.
- **`src/calibration/`**: Probability reliability analysis (Brier score, ECE) comparing uncalibrated predictions against Platt scaling and Isotonic regression with empty-bin omitted reliability curves.
- **`src/explainability/`**: SHAP feature attributions (`TreeExplainer` with path-dependent exact evaluation) and multi-fold explanation stability analysis ($\rho$, $\tau$, $J$).
- **`src/robustness/`**: Controlled perturbation testing evaluating model resilience under noise, missingness, and feature corruption.
- **`src/statistics/`**: Non-parametric hypothesis testing providing bootstrap confidence intervals, paired McNemar/Wilcoxon tests, and Friedman/Nemenyi omnibus analysis.
- **`src/reporting/`**: Automated generation of publication-quality figures (300 DPI multi-panel reliability subplots) and structured research Markdown reports with evidence-based Section 9 strategic synthesis.
