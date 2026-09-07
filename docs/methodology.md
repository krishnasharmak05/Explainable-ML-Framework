# Research Methodology & Mathematical Formulations

This document describes the foundational ML and statistical methodologies implemented across Phases 1 through 8.

---

## 1. Leakage Prevention Principles

Data leakage occurs when information from outside the training dataset is inadvertently used to train or select features for a model. The framework prevents leakage by design:
1. **Split-First Architecture**: Train/test splitting occurs prior to any parameter estimation.
2. **Preprocessor Encapsulation**: Imputation statistics ($\mu_j$), skewness detections, and min/max ranges ($X_{min}, X_{max}$) are learned exclusively on the training split and applied to test data without re-estimation.
3. **Feature Selection Isolation**: Mutual Information scores $I(f; Y)$ are computed strictly using training observations and training labels.
4. **Calibration Validation Partitioning**: Probability calibrators (Platt scaling, Isotonic regression) are fitted on internal validation partitions, never on the held-out test set.

---

## 2. Module 1: Data Preprocessing & Normalization

- **Infinite Value Resolution**: `+inf` and `-inf` (e.g. from zero-duration flow rates) are resolved to `NaN` values for structured imputation.
- **Missing Value Imputation (Eq. 1)**:
  $$x'_{i,j} = \begin{cases} x_{i,j} & \text{if } x_{i,j} \text{ is observed} \\ \mu_j & \text{if } x_{i,j} \text{ is missing} \end{cases}$$
- **Logarithmic Transformation (Eq. 2)**:
  $$X_{log} = \log(X + 1)$$
  Applied strictly to features exhibiting absolute sample skewness $> 1.0$ to stabilize variance.
- **Min-Max Normalization (Eq. 3)**:
  $$X_{norm} = \frac{X - X_{min}}{X_{max} - X_{min}} \in [0, 1]$$

---

## 3. Module 2: Feature Selection (Mutual Information)

- Target entropy:
  $$H(Y) = - \sum_{y \in Y} P(y) \log P(y)$$
- Mutual Information between continuous feature $X$ and label $Y$:
  $$I(X; Y) = \sum_{x \in X} \sum_{y \in Y} P(x, y) \log \left( \frac{P(x, y)}{P(x)P(y)} \right)$$
- Optimal subset of top-$k$ informative features:
  $$S^* = \arg\max_{S, |S|=k} \sum_{f \in S} I(f; Y)$$

---

## 4. Module 3: Predictive Modeling Framework

- **Random Forest (Eq. 7)**: Majority vote over $B$ decision trees:
  $$\hat{y} = \text{mode} \{ h_1(x), h_2(x), ..., h_B(x) \}$$
- **Gradient Boosting (XGBoost & LightGBM, Eq. 8-9)**: Minimizes regularized cross-entropy objective:
  $$\mathcal{L}^{(t)} = \sum_{i=1}^n l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t)$$
  $$L = - \frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

---

## 5. Phase 4: Probability Calibration Analysis

- **Brier Score**:
  $$\text{Brier} = \frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C (p_{i,c} - y_{i,c})^2$$
- **Expected Calibration Error (ECE)**:
  $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
- **Platt Scaling**: Fits univariate logistic sigmoid on model logits:
  $$P(y=1|z) = \frac{1}{1 + \exp(A \cdot z + B)}$$
- **Isotonic Regression**: Non-parametric piecewise constant monotonic calibration minimizing squared residuals under ordering constraints.
- **Reliability Diagram Formulation**: Evaluates empirical accuracy $\text{acc}(B_m)$ vs mean predicted confidence $\text{conf}(B_m)$ across non-empty bins. Empty bins ($|B_m| = 0$) are strictly omitted to eliminate artificial zero-fraction oscillation artifacts, supporting equal-frequency quantile binning for skewed prediction distributions.

---

## 6. Phase 5: SHAP Explainability & Explanation Stability

- **Coalitional Game Theory Attributions (Eq. 15-16)**:
  $$\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|! (|F| - |S| - 1)!}{|F|!} \left[ f_x(S \cup \{i\}) - f_x(S) \right]$$
  Additive efficiency property: $g(z') = \phi_0 + \sum_{i=1}^M \phi_i z'_i$.
- **Explanation Stability Across CV Folds**:
  - **Spearman Rank Correlation ($\rho$)**:
    $$\rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}$$
  - **Kendall Rank Correlation ($\tau$)**:
    $$\tau = \frac{C - D}{\frac{1}{2} n (n - 1)}$$
  - **Jaccard Similarity of Top-$k$ Features**:
    $$J(S_a, S_b) = \frac{|S_a \cap S_b|}{|S_a \cup S_b|}$$

---

## 7. Phase 6: Robustness Perturbations

Evaluates metric degradation ($\Delta M = M_{baseline} - M_{perturbed}$) across:
1. **Feature Noise**: Zero-mean Gaussian noise scaled by empirical feature variance: $X + \mathcal{N}(0, (\sigma \cdot \text{std}(X))^2)$.
2. **Random Missingness**: Random masking with `NaN` evaluated through pre-fitted imputation pipelines.
3. **Feature Corruption**: Value permutation of top-$k$ informative features.
4. **Learning Curve Degradation**: Training set subsampling across $100\%, 75\%, 50\%, 25\%$.

---

## 8. Phase 7: Statistical Analysis & Hypothesis Testing

- **Non-Parametric Bootstrap Confidence Intervals**: Percentile intervals computed over $B=1000$ empirical resamples.
- **McNemar's Test**: Evaluates paired classification performance between two models on the same test observations:
  $$\chi^2 = \frac{(|b - c| - 1)^2}{b + c} \quad (\text{with exact binomial test when } b+c < 25)$$
- **Wilcoxon Signed-Rank Test**: Evaluates paired fold-level CV scores with Rank-Biserial Correlation effect size.
- **Friedman Omnibus Test & Nemenyi Post-Hoc**:
  $$\chi_F^2 = \frac{12N}{K(K+1)} \left[\sum_{j=1}^K R_j^2 - \frac{K(K+1)^2}{4}\right]$$
  with Critical Difference:
  $$CD = q_{\alpha} \sqrt{\frac{K(K+1)}{6N}}$$

---

## 9. Phase 8: Automated Publication Reporting & Synthesis

- **300-DPI Publication Figures**: Multi-panel calibration reliability diagrams (Uncalibrated vs Sigmoid vs Isotonic), side-by-side ROC and PR curves, normalized confusion matrices, and perturbation degradation profiles.
- **Dynamic Decision Synthesis (Section 9)**: Fully data-driven algorithm ranking and strategic deployment decision matrix:
  - Selects the superior classifier based on held-out test F1-score and MCC.
  - Verifies whether performance superiority is statistically significant ($p < 0.05$ on McNemar/Wilcoxon) or falls within an equivalence regime.
  - Identifies Pareto-optimal operational deployments across high-throughput wire-speed inline inspection, forensics auditing, and resource-constrained edge gateways.
