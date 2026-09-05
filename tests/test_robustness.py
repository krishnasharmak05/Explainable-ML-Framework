import numpy as np
import pandas as pd
import pytest
from src.modeling.ensemble import RandomForestWrapper
from src.robustness.evaluator import RobustnessEvaluator
from src.robustness.perturbations import (
    apply_feature_corruption,
    apply_feature_noise,
    apply_random_missingness,
    subsample_training_data,
)


def test_perturbation_functions():
    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.normal(size=(50, 4)), columns=[f"f{i}" for i in range(4)])
    y = rng.choice([0, 1], size=50)

    # 1. Feature Noise
    X_noisy = apply_feature_noise(X, noise_level=0.1)
    assert X_noisy.shape == X.shape
    assert not np.allclose(X.values, X_noisy.values)

    # 2. Random Missingness
    X_missing = apply_random_missingness(X, missing_rate=0.2)
    assert X_missing.isna().sum().sum() > 0

    # 3. Feature Corruption
    X_corrupt = apply_feature_corruption(X, corrupt_columns=["f0"])
    assert not np.allclose(X["f0"].values, X_corrupt["f0"].values)
    assert np.allclose(X["f1"].values, X_corrupt["f1"].values)

    # 4. Subsample Training Data
    X_sub, y_sub = subsample_training_data(X, y, fraction=0.5)
    assert len(X_sub) == 25
    assert len(y_sub) == 25


def test_robustness_evaluator():
    rng = np.random.RandomState(42)
    X_tr = pd.DataFrame(rng.normal(size=(80, 4)), columns=[f"f{i}" for i in range(4)])
    y_tr = rng.choice([0, 1], size=80)

    X_te = pd.DataFrame(rng.normal(size=(30, 4)), columns=[f"f{i}" for i in range(4)])
    y_te = rng.choice([0, 1], size=30)

    model = RandomForestWrapper(params={"n_estimators": 5, "random_state": 42})
    model.fit(X_tr, y_tr)

    evaluator = RobustnessEvaluator(random_state=42)
    reports = evaluator.evaluate_model_robustness(
        model=model,
        X_test=X_te,
        y_test=y_te,
        model_name="rf",
        noise_levels=[0.05],
        top_features_to_corrupt=["f0"],
    )

    assert len(reports) >= 3  # baseline, feature_noise, feature_corruption
    df_rep = RobustnessEvaluator.reports_to_dataframe(reports)
    assert "accuracy_degradation" in df_rep.columns
    assert "mcc" in df_rep.columns
