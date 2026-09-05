import numpy as np
import pandas as pd
import pytest
from src.modeling.ensemble import RandomForestWrapper, XGBoostWrapper, LightGBMWrapper, get_model
from src.modeling.trainer import ModelTrainer


def test_ensemble_model_wrappers():
    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.normal(size=(100, 5)), columns=[f"f{i}" for i in range(5)])
    y = rng.choice([0, 1], size=100)

    for model_name in ["random_forest", "xgboost", "lightgbm"]:
        wrapper = get_model(model_name, params={"n_estimators": 5, "random_state": 42})
        wrapper.fit(X, y)
        preds = wrapper.predict(X)
        probs = wrapper.predict_proba(X)

        assert len(preds) == 100
        assert probs.shape == (100, 2)
        assert np.allclose(probs.sum(axis=1), 1.0)


def test_model_trainer_cv_and_algorithm2(tmp_path):
    rng = np.random.RandomState(42)
    X_train = pd.DataFrame(rng.normal(size=(120, 4)), columns=[f"feat_{i}" for i in range(4)])
    y_train = rng.choice([0, 1], size=120)
    X_test = pd.DataFrame(rng.normal(size=(30, 4)), columns=[f"feat_{i}" for i in range(4)])

    models = [
        RandomForestWrapper(params={"n_estimators": 10, "random_state": 42}),
        XGBoostWrapper(params={"n_estimators": 10, "random_state": 42}),
        LightGBMWrapper(params={"n_estimators": 10, "random_state": 42}),
    ]

    trainer = ModelTrainer(cv_folds=3, random_state=42)
    results = trainer.train_pipeline(
        X_train=X_train,
        y_train=y_train,
        models=models,
        X_test=X_test,
        save_dir=tmp_path,
    )

    summary = results.summary()
    assert len(summary) == 3
    for name in ["random_forest", "xgboost", "lightgbm"]:
        probs = results.get_probabilities(name)
        assert probs is not None
        assert probs.shape == (30, 2)
        assert (tmp_path / f"{name}.joblib").exists()
