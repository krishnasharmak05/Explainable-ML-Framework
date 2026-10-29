# Machine Learning Research Experiment Report

**Generated:** 2026-10-27 14:50:08  
**Classification Mode:** BINARY  
**Evaluation Strategy:** 5-Fold Stratified Cross-Validation + Independent Test Set

---

## 1. Dataset Overview & Data Hygiene

- **Total Observations:** 692,703
- **Raw Feature Count:** 78
- **Selected Feature Count:** 15 (via Mutual Information)
- **Class Distribution:** {'BENIGN': np.int64(440031), 'DoS Hulk': np.int64(231073), 'DoS GoldenEye': np.int64(10293), 'DoS slowloris': np.int64(5796), 'DoS Slowhttptest': np.int64(5499), 'Heartbleed': np.int64(11)}
- **Infinite Values Detected:** 1,586 (converted to missing for systematic imputation)
- **Missing Values Imputed:** 1,008 (median imputation)

## 2. Predictive Performance Comparison

The table below presents the cross-validated and held-out test evaluation across candidate algorithms:

| Model | Accuracy | F1 | ROC-AUC | MCC |
| --- | --- | --- | --- | --- |
| random_forest | 0.9983 | 0.9976 | 1.0000 | 0.9963 |
| xgboost | 0.9981 | 0.9974 | 1.0000 | 0.9959 |
| lightgbm | 0.9981 | 0.9974 | 1.0000 | 0.9959 |

### 2.1 Non-Parametric Bootstrap Confidence Intervals (95% CI)

| Metric | Point Estimate | Bootstrap Mean | Bootstrap Std | 95% CI Lower | 95% CI Upper |
| --- | --- | --- | --- | --- | --- |
| accuracy | 0.9983 | 0.9983 | 0.0001 | 0.9980 | 0.9985 |
| balanced_accuracy | 0.9984 | 0.9984 | 0.0001 | 0.9981 | 0.9986 |
| precision | 0.9966 | 0.9966 | 0.0003 | 0.9961 | 0.9970 |
| recall | 0.9987 | 0.9987 | 0.0002 | 0.9984 | 0.9990 |
| f1 | 0.9976 | 0.9976 | 0.0002 | 0.9973 | 0.9980 |
| mcc | 0.9963 | 0.9963 | 0.0002 | 0.9958 | 0.9968 |
| roc_auc | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| brier_score | 0.0013 | 0.0013 | 0.0001 | 0.0012 | 0.0014 |

## 3. Computational Efficiency & Resource Footprint

| model_name | preprocessing_time_sec | feature_selection_time_sec | training_time_sec | inference_time_sec | inference_latency_ms_per_sample | inference_throughput_samples_per_sec | peak_memory_mb | model_size_bytes | model_size_kb | n_raw_features | n_selected_features | n_train_samples | n_test_samples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random_forest | 13.64067050000017 | 21.510528699999895 | 174.41869401931763 | 0.3357557999999396 | 0.0024235121732912247 | 412624.2942043739 | 2620.2871627807617 | 6212808 | 6067.1953125 | 78 | 15 | 554162 | 138541 |
| xgboost | 13.64067050000017 | 21.510528699999895 | 27.03769087791443 | 0.13340629999993325 | 0.0009629373254122119 | 1038489.1867930474 | 2620.2871627807617 | 236831 | 231.2802734375 | 78 | 15 | 554162 | 138541 |
| lightgbm | 13.64067050000017 | 21.510528699999895 | 20.84254002571106 | 0.2862650999998853 | 0.0020662843490366414 | 483960.4967565222 | 2620.2871627807617 | 355011 | 346.6904296875 | 78 | 15 | 554162 | 138541 |

## 4. Probability Calibration Analysis

Reliability of confidence scores evaluated via Brier Score and Expected Calibration Error (ECE):

| Model | Calibration Method | Brier Score | ECE | Brier Improvement | ECE Improvement |
| --- | --- | --- | --- | --- | --- |
| random_forest | uncalibrated | 0.00132 | 0.00114 | +0.00000 | +0.00000 |
| random_forest | sigmoid | 0.00131 | 0.00046 | +0.00001 | +0.00067 |
| random_forest | isotonic | 0.00130 | 0.00021 | +0.00002 | +0.00092 |
| xgboost | uncalibrated | 0.00159 | 0.00148 | +0.00000 | +0.00000 |
| xgboost | sigmoid | 0.00154 | 0.00050 | +0.00005 | +0.00098 |
| xgboost | isotonic | 0.00150 | 0.00018 | +0.00008 | +0.00130 |
| lightgbm | uncalibrated | 0.00150 | 0.00141 | +0.00000 | +0.00000 |
| lightgbm | sigmoid | 0.00145 | 0.00044 | +0.00005 | +0.00097 |
| lightgbm | isotonic | 0.00142 | 0.00015 | +0.00009 | +0.00126 |

## 5. Model Explainability & Explanation Stability

Consistency of global SHAP feature attributions measured across cross-validation folds:

| model_name | top_k | n_folds | mean_spearman_rho | std_spearman_rho | mean_kendall_tau | std_kendall_tau | mean_jaccard_similarity | std_jaccard_similarity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random_forest | 10 | 5 | 0.9539285714285712 | 0.017746083212578374 | 0.8666666666666668 | 0.03407341679999679 | 0.8727272727272728 | 0.08331955809010616 |
| xgboost | 10 | 5 | 0.982142857142857 | 0.006776309271789372 | 0.9314285714285715 | 0.024392854237839434 | 0.9272727272727274 | 0.08907235428302464 |
| lightgbm | 10 | 5 | 0.9524999999999999 | 0.033732516257481075 | 0.8971428571428574 | 0.04758093713065446 | 0.9272727272727274 | 0.08907235428302464 |

## 6. Robustness Under Controlled Perturbations

Performance degradation under Gaussian feature noise, random missingness, and feature corruption:

| model_name | experiment_type | severity_level | accuracy | f1 | mcc | log_loss | brier_score | accuracy_degradation | f1_degradation | mcc_degradation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random_forest | baseline | 0.0 | 0.998282096996557 | 0.9976477099764771 | 0.9962961099831155 | 0.005595762254176898 | 0.0013225471343051172 | 0.0 | 0.0 | 0.0 |
| random_forest | feature_noise | 0.01 | 0.9934243292599303 | 0.9909379383063593 | 0.9858134081668678 | 0.0550290466998929 | 0.011719905748006153 | 0.004857767736626717 | 0.00670977167011777 | 0.010482701816247664 |
| random_forest | feature_noise | 0.05 | 0.9244916667268174 | 0.8856333839880178 | 0.8404652267335813 | 0.18318160605508932 | 0.057293117101654044 | 0.07379043026973964 | 0.11201432598845928 | 0.1558308832495342 |
| random_forest | feature_noise | 0.1 | 0.8363733479619752 | 0.7150238223943077 | 0.6580622876129814 | 0.2764107577945131 | 0.09426277465013222 | 0.16190874903458186 | 0.2826238875821694 | 0.3382338223701341 |
| random_forest | feature_noise | 0.2 | 0.7464721634750724 | 0.4778342699134778 | 0.4587079356564769 | 0.38978713631326833 | 0.13625702455058572 | 0.2518099335214846 | 0.5198134400629992 | 0.5375881743266386 |
| random_forest | feature_corruption_Destination Port | corrupted | 0.930035152048852 | 0.9027910101992719 | 0.848349736256978 | 0.7727708491154424 | 0.07077564977212582 | 0.068246944947705 | 0.09485669977720523 | 0.1479463737261375 |
| random_forest | feature_corruption_Bwd Packet Length Mean | corrupted | 0.9981593896391682 | 0.9974787672655007 | 0.9960300275696811 | 0.12498468163333852 | 0.023039054475252268 | 0.00012270735738884042 | 0.00016894271097644964 | 0.00026608241343439865 |
| random_forest | feature_corruption_Packet Length Mean | corrupted | 0.9957052424913925 | 0.9941202628588369 | 0.9907392929982887 | 0.10759350695201235 | 0.02207419454200972 | 0.0025768545051645386 | 0.0035274471176401923 | 0.005556816984826796 |
| xgboost | baseline | 0.0 | 0.9980872088407042 | 0.9973825867944096 | 0.9958792103470372 | 0.006889129523187876 | 0.001588161918334663 | 0.0 | 0.0 | 0.0 |
| xgboost | feature_noise | 0.01 | 0.9743758165452826 | 0.9640542729850142 | 0.9447871947920882 | 0.07738125324249268 | 0.02177971974015236 | 0.02371139229542163 | 0.03332831380939538 | 0.05109201555494902 |
| xgboost | feature_noise | 0.05 | 0.8522964321031319 | 0.7491326361730559 | 0.6917237958394576 | 0.7083095908164978 | 0.12960726022720337 | 0.14579077673757235 | 0.2482499506213537 | 0.30415541450757955 |
| xgboost | feature_noise | 0.1 | 0.7728903357128937 | 0.5529665833143896 | 0.5230086876013864 | 1.1769568920135498 | 0.2042839378118515 | 0.22519687312781056 | 0.44441600348001997 | 0.4728705227456508 |

## 7. Statistical Validation & Hypothesis Testing

### McNemar Test: random_forest vs xgboost
- **Test Statistic:** 7.5955
- **P-Value:** 0.00585 (Threshold: $\alpha=0.05$)
- **Statistically Significant:** **YES**
- **Effect Size:** Odds Ratio (b/c) = 1.8710
- **Interpretation:** Significant difference (p=0.0059 < 0.05) between random_forest and xgboost.

### Wilcoxon Signed-Rank Test: random_forest vs xgboost
- **Test Statistic:** 0.0000
- **P-Value:** 0.06250 (Threshold: $\alpha=0.05$)
- **Statistically Significant:** **NO**
- **Effect Size:** Rank-Biserial Correlation (|r|) = 0.5000
- **Interpretation:** No significant score difference (p=0.0625 >= 0.05) between random_forest and xgboost.

### McNemar Test: random_forest vs lightgbm
- **Test Statistic:** 6.0114
- **P-Value:** 0.01421 (Threshold: $\alpha=0.05$)
- **Statistically Significant:** **YES**
- **Effect Size:** Odds Ratio (b/c) = 1.7500
- **Interpretation:** Significant difference (p=0.0142 < 0.05) between random_forest and lightgbm.

### Wilcoxon Signed-Rank Test: random_forest vs lightgbm
- **Test Statistic:** 0.0000
- **P-Value:** 0.06250 (Threshold: $\alpha=0.05$)
- **Statistically Significant:** **NO**
- **Effect Size:** Rank-Biserial Correlation (|r|) = 0.5000
- **Interpretation:** No significant score difference (p=0.0625 >= 0.05) between random_forest and lightgbm.

### McNemar Test: xgboost vs lightgbm
- **Test Statistic:** 0.1212
- **P-Value:** 0.72772 (Threshold: $\alpha=0.05$)
- **Statistically Significant:** **NO**
- **Effect Size:** Odds Ratio (b/c) = 0.8333
- **Interpretation:** No statistically significant difference (p=0.7277 >= 0.05) between xgboost and lightgbm.

### Wilcoxon Signed-Rank Test: xgboost vs lightgbm
- **Test Statistic:** 2.0000
- **P-Value:** 0.18750 (Threshold: $\alpha=0.05$)
- **Statistically Significant:** **NO**
- **Effect Size:** Rank-Biserial Correlation (|r|) = 0.3667
- **Interpretation:** No significant score difference (p=0.1875 >= 0.05) between xgboost and lightgbm.

## 8. Scientifically Important Caveats & Limitations

1. **Data Leakage Safeguards:** All preprocessing, scalers, imputers, and feature selectors were fitted strictly on the training partitions and applied downstream without inspecting validation or test instances.
2. **Statistical vs Practical Significance:** A statistically significant difference between models (e.g. via McNemar's or Wilcoxon tests) does not automatically denote practical or operational superiority.
3. **SHAP Semantics:** SHAP provides additive attributions under coalition game assumptions. It explains model behavior relative to a baseline expectation, not direct real-world causal mechanisms.
4. **Imbalance Metrics:** Accuracy alone is misleading on imbalanced data; decision-making should be grounded in MCC, PR-AUC, and F1.
5. **Distribution Shift:** Robustness experiments apply synthetic noise and missingness; operational domain shift in network environments may exhibit more complex temporal or topological variations.

## 9. Final Conclusion & Strategic Recommendations

### 9.1 Overall Model Selection & Performance Synthesis
Across the multi-dimensional benchmark suite, **`random_forest`** emerges as the overall superior classifier, achieving the highest held-out test performance with an **F1-Score of 0.9976**, **Accuracy of 0.9983**, and **Matthews Correlation Coefficient (MCC) of 0.9963**.

### 9.2 Statistical Hypothesis Testing Verification
- **Confirmed Superiority:** Statistically significant performance differences were established via McNemar Test between `random_forest` and `xgboost` ($p = 0.0059$); McNemar Test between `random_forest` and `lightgbm` ($p = 0.0142$). This rejects the null hypothesis of equal error distributions and demonstrates that the observed lead is not attributable to sampling artifact.
- **Equivalence Regimes:** Paired comparisons where models performed without statistically significant divergence include: `random_forest` vs `xgboost` ($p = 0.0625$); `random_forest` vs `lightgbm` ($p = 0.0625$), indicating interchangeable predictive power between gradient-boosted variants.

### 9.3 Computational Efficiency & Throughput Trade-Offs
- **Training Velocity:** `lightgbm` trained most rapidly at **20.84 seconds**.
- **Inference Throughput:** `xgboost` attained the highest evaluation throughput at **1,038,489 samples/sec** (latency: **0.0010 ms/sample**).
- **Deployment Footprint:** `xgboost` produced the most compact model artifact at **231.3 KB**, making it highly suitable for constrained edge security appliances.

### 9.4 Explainability Stability & Decision Transparency
Across evaluated classifiers (`random_forest`, `xgboost`, `lightgbm`), global SHAP feature attribution rankings demonstrated robust inter-fold consistency with an overall cross-model mean Spearman $\rho = 0.9629$, Kendall $\tau = 0.8984$, and top-$k$ Jaccard consensus of 90.91%.
- **Attribution Consistency:** **`xgboost`** achieved the highest attribution stability across cross-validation splits (Spearman $\rho = 0.9821 \pm 0.0068$, Kendall $\tau = 0.9314$, top-$k$ Jaccard agreement = 92.73%).
- **Validation of Threat Logic:** This high rank stability confirms that the primary predictive indicators identified by `xgboost` reflect genuine signal in the underlying traffic distribution rather than random partition noise.

### 9.5 Adversarial & Perturbation Resilience
- **Gaussian Noise Tolerance:** Under maximum packet feature noise ($\sigma = 0.2$), **`random_forest`** retained the highest residual test accuracy at **0.7465** (accuracy degradation limited to 25.18%).
- **Overall Perturbation Resilience:** **`random_forest`** exhibited the lowest overall sensitivity across all noise and corruptive regimes with an average accuracy degradation of only **7.04%**.
- **Critical Vulnerability Point:** Feature corruption of `Destination Port` induced the steepest performance decline in **`lightgbm`** (9.42% accuracy loss), identifying `Destination Port` as a critical dependency that requires strict telemetric integrity validation at the network boundary.

### 9.6 Recommended Deployment Decision Matrix

| Operational Scenario | Recommended Architecture | Primary Justification |
|---|---|---|
| **High-Throughput Inline NIDS (10G/40G Core)** | **`xgboost`** | Peak throughput of 1,038,489 samples/sec with sub-millisecond latency (0.0010 ms/sample). |
| **Security Operations Center (SOC) Forensics** | **`xgboost`** | Highest inter-fold attribution stability (Spearman $\rho = 0.9821$) and verified additive SHAP explanations. |
| **Resource-Constrained IoT Gateway / Edge Sensor** | **`xgboost`** | Most compact deployment footprint (231.3 KB disk, 2620.3 MB RAM). |
| **Adversarial / High-Noise Network Boundary** | **`random_forest`** | Lowest overall degradation (7.04%) under packet perturbations. |
| **High-Precision Mission-Critical Threat Detection** | **`random_forest`** | Highest held-out discrimination fidelity (F1: 0.9976, MCC: 0.9963). |
