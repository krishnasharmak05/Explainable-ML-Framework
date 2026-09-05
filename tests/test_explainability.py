import numpy as np
import pandas as pd
import pytest
from src.explainability.shap_explainer import ShapExplainer
from src.explainability.stability import ExplanationStabilityAnalyzer
from src.modeling.ensemble import RandomForestWrapper, XGBoostWrapper


def test_shap_explainer():
    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.normal(size=(60, 4)), columns=["feat_A", "feat_B", "feat_C", "feat_D"])
    y = (X["feat_A"] + X["feat_B"] > 0).astype(int).values

    model = RandomForestWrapper(params={"n_estimators": 5, "random_state": 42})
    model.fit(X, y)

    explainer = ShapExplainer(max_background_samples=30, max_eval_samples=40)
    result = explainer.explain(model, X, model_name="rf")

    assert result.model_name == "rf"
    assert len(result.feature_importance_df) == 4
    assert result.is_additive_verified
    assert "Mean_Absolute_SHAP" in result.feature_importance_df.columns

    # Check local explanation
    local_df = ShapExplainer.explain_local_sample(result, sample_index=0, feature_values=X.iloc[0])
    assert len(local_df) == 4
    assert "SHAP_Attribution" in local_df.columns


def test_explanation_stability_analyzer():
    # Simulate 3 fold importance dataframes with strong agreement
    fold_dfs = [
        pd.DataFrame({"Feature": ["f1", "f2", "f3", "f4"], "Mean_Absolute_SHAP": [0.8, 0.6, 0.3, 0.1]}),
        pd.DataFrame({"Feature": ["f1", "f2", "f3", "f4"], "Mean_Absolute_SHAP": [0.75, 0.65, 0.25, 0.15]}),
        pd.DataFrame({"Feature": ["f1", "f2", "f4", "f3"], "Mean_Absolute_SHAP": [0.85, 0.55, 0.20, 0.10]}),
    ]

    analyzer = ExplanationStabilityAnalyzer(top_k=2)
    stability = analyzer.analyze_stability(fold_dfs, model_name="test_model")

    assert stability.n_folds == 3
    assert stability.top_k == 2
    assert stability.mean_spearman_rho > 0.8
    assert stability.mean_kendall_tau > 0.7
    assert stability.mean_jaccard_similarity == 1.0  # Top 2 features (f1, f2) match in all 3 folds
    assert stability.pairwise_spearman_matrix.shape == (3, 3)
    assert len(stability.top_k_consensus_features) == 2
