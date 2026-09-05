### III. Proposed Methodology

The proposed system is a reusable, modular explainable machine learning framework designed specifically for cyber attack detection using the CIC-IDS2017 dataset. Instead of relying on a rigid, single-use script, our system is engineered as a structured pipeline that standardizes how data is  processed, how models are trained and compared, and how results are interpreted. By decoupling these stages into distinct modules, the framework allows researchers and analysts to easily swap out datasets, feature selection methods, or algorithms without rewriting the underlying architecture. This approach ensures high reproducibility and shifts the focus from simply maximizing accuracy to understanding the complete predictive analytics workflow. 

The architecture of the proposed system is visually represented in Figure 1 [Insert Figure 1 here]. The pipeline begins with the raw CIC-IDS2017 network traffic data, which flows into the Data Preprocessing module where anomalies like infinite or missing values are resolved, and numerical values are scaled. The cleaned data then moves to the Feature Selection module to filter out redundant variables. Next, the Predictive Modeling module trains multiple tree-based ensemble models (such as Random Forest, XGBoost, and LightGBM) in parallel. The Evaluation module rigorously assesses model performance using robust metrics tailored for imbalanced data. Finally, the Explainability module leverages SHAP (SHapley Additive exPlanations) to demystify the "black-box" predictions, outputting both global feature importance and local attack explanations into a comprehensive final report.

#### Module 1: Data Preprocessing and Normalization
Raw network traffic data often contains inconsistencies, such as missing values, duplicate rows, or extreme outliers (like infinite values generated during packet capture). This module systematically cleans the CIC-IDS2017 dataset to ensure stable model training. First, infinite values are capped, and missing values are imputed using the mean or median of the respective feature. To handle highly skewed numerical features without distorting their underlying distribution, a logarithmic transformation is applied. Finally, the data is scaled so that all features contribute equally to the learning process, preventing features with large numeric ranges (e.g., total bytes) from dominating the model. 

The imputation of a missing value $x_{i,j}$ for a feature $j$ using the feature mean $\mu_j$ is given by:
$$x'_{i,j} = \begin{cases} x_{i,j} & \text{if } x_{i,j} \text{ is observed} \\ \mu_j & \text{if } x_{i,j} \text{ is missing} \end{cases} \quad \text{--- (1)}$$

For highly skewed network features, a logarithmic transformation is applied to stabilize the variance:
$$X_{log} = \log(X + 1) \quad \text{--- (2)}$$

Finally, Min-Max scaling is used to normalize the feature vector $X$ into a standard range $[0, 1]$:
$$X_{norm} = \frac{X - X_{min}}{X_{max} - X_{min}} \quad \text{--- (3)}$$

#### Module 2: Feature Selection
Network flow datasets like CIC-IDS2017 contain dozens of features, many of which are redundant or irrelevant to detecting malicious activity. Training models on all available features increases computational overhead and the risk of overfitting. This module evaluates the importance of each feature and selects a streamlined subset that retains the highest predictive power. We utilize Mutual Information (MI), which measures the reduction in uncertainty for the target label given a specific network feature.

The entropy of the target class $Y$ (Benign vs. Attack) is defined as:
$$H(Y) = - \sum_{y \in Y} P(y) \log P(y) \quad \text{--- (4)}$$

The Mutual Information between a network feature $X$ and the target class $Y$ is computed as:
$$I(X; Y) = \sum_{x \in X} \sum_{y \in Y} P(x, y) \log \left( \frac{P(x, y)}{P(x)P(y)} \right) \quad \text{--- (5)}$$

The feature selection process aims to find a subset $S$ that maximizes this mutual information with the target:
$$S^* = \arg\max_{S} I(S; Y) \quad \text{--- (6)}$$

**Algorithm 1: Mutual Information-Based Feature Selection**
```text
Input: Dataset D with feature set F and target labels Y, desired feature count k
Output: Selected feature subset S*
1: Initialize S* = ∅
2: For each feature f in F do:
3:     Calculate probability distributions P(f), P(Y), and P(f, Y)
4:     Compute Mutual Information I(f; Y) using Equation (5)
5:     Store I(f; Y) in a dictionary score_map
6: End For
7: Sort score_map in descending order based on I(f; Y) values
8: Select the top k features from the sorted score_map
9: Add selected features to S*
10: Return S*
```

#### Module 3: Predictive Modeling Framework
This module serves as the core classification engine. Because our goal is building a robust and reusable pipeline rather than inventing a new algorithm, we rely on proven tree-based ensemble models: Random Forest, XGBoost, and LightGBM. These models are highly effective for tabular network data and handle non-linear relationships well. The framework wraps these models in a unified interface, ensuring they all receive the same training data, undergo the same cross-validation process, and output predictions in a standard format for comparison.

For a Random Forest model, the final prediction $\hat{y}$ is the majority vote of $B$ individual decision trees $h_b(x)$:
$$\hat{y} = \text{mode} \{ h_1(x), h_2(x), ..., h_B(x) \} \quad \text{--- (7)}$$

For gradient boosting frameworks (like XGBoost), the objective function to be minimized at iteration $t$ includes a loss term $l$ and a regularization term $\Omega$:
$$\mathcal{L}^{(t)} = \sum_{i=1}^n l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t) \quad \text{--- (8)}$$

During binary classification (Benign vs. Attack), the models minimize the log-loss (cross-entropy) error:
$$L = - \frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right] \quad \text{--- (9)}$$

**Algorithm 2: Reusable Model Training Pipeline**
```text
Input: Processed training data (X_train, Y_train), List of initialized models M
Output: Dictionary of trained models and their prediction probabilities
1: Initialize results dictionary R = {}
2: For each model m in M do:
3:     Initialize k-fold cross-validation
4:     For each fold (train_fold, val_fold) do:
5:          Train m on train_fold
6:          Calculate log-loss L on val_fold using Equation (9)
7:     End For
8:     Train final model m on full (X_train, Y_train)
9:     Extract prediction probabilities P = m.predict_proba(X_test)
10:    Store m and P in R
11: End For
12: Return R
```

#### Module 4: Model Evaluation Module
Cyber attack datasets are notoriously imbalanced; benign traffic vastly outnumbers attack traffic. Therefore, evaluating a model solely on accuracy is misleading. This evaluation module computes a diverse suite of metrics to provide a realistic assessment of how well the framework detects rare attacks without triggering false alarms. 

Accuracy measures the overall correctness of the model:
$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN} \quad \text{--- (10)}$$

Precision indicates the proportion of flagged attacks that were genuinely malicious (minimizing false alarms):
$$Precision = \frac{TP}{TP + FP} \quad \text{--- (11)}$$

Recall (or Detection Rate) measures the proportion of actual attacks the system successfully caught:
$$Recall = \frac{TP}{TP + FN} \quad \text{--- (12)}$$

The F1-Score provides a harmonic mean between Precision and Recall, which is crucial for imbalanced network data:
$$F1\text{-}Score = 2 \times \frac{Precision \times Recall}{Precision + Recall} \quad \text{--- (13)}$$

Finally, the Matthews Correlation Coefficient (MCC) is used as the most reliable statistical rate of binary classification performance, as it takes all four confusion matrix categories into account:
$$MCC = \frac{(TP \times TN) - (FP \times FN)}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}} \quad \text{--- (14)}$$
*(Where TP = True Positives, TN = True Negatives, FP = False Positives, FN = False Negatives)*

#### Module 5: Explainability and Reporting
A high-performing cyber detection system is only useful if security analysts can trust and understand its alerts. This module applies SHAP (SHapley Additive exPlanations) to open the "black box" of the ensemble models. SHAP calculates the exact contribution of every network feature (e.g., packet length, flow duration) to the final prediction. The module generates global plots (which features matter most overall) and local plots (why a specific packet was flagged as an attack), compiling these alongside the evaluation metrics into a clear, reproducible experiment report.

The SHAP value $\phi_i$ for a specific feature $i$ is calculated based on coalitional game theory:
$$\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|! (|F| - |S| - 1)!}{|F|!} \left[ f_x(S \cup \{i\}) - f_x(S) \right] \quad \text{--- (15)}$$
*(Where $F$ is the set of all features, $S$ is a subset of features without $i$, and $f_x$ is the model prediction).*

SHAP guarantees that the sum of the individual feature attributions equals the difference between the actual prediction and the base (average) prediction, known as the additive feature attribution property:
$$g(z') = \phi_0 + \sum_{i=1}^M \phi_i z'_i \quad \text{--- (16)}$$
*(Where $g$ is the explanation model, $z' \in \{0,1\}^M$ indicates feature presence, and $\phi_0$ is the baseline expectation).*
