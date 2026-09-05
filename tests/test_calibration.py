import numpy as np
import pandas as pd
import pytest
from src.calibration.calibrator import ModelCalibrator
from src.calibration.metrics import compute_brier_score, compute_calibration_curve, compute_ece
from src.modeling.ensemble import RandomForestWrapper


def test_calibration_metrics():
    y_true = np.array([0, 1, 1, 0, 1])
    # Perfect probabilities
    y_prob_perf = np.array([0.0, 1.0, 1.0, 0.0, 1.0])
    # Overconfident wrong probabilities
    y_prob_wrong = np.array([1.0, 0.0, 0.0, 1.0, 0.0])

    brier_perf = compute_brier_score(y_true, y_prob_perf)
    brier_wrong = compute_brier_score(y_true, y_prob_wrong)
    assert brier_perf == 0.0
    assert brier_wrong == 1.0

    ece_perf = compute_ece(y_true, y_prob_perf, n_bins=5)
    assert ece_perf == 0.0

    curve = compute_calibration_curve(y_true, y_prob_perf, n_bins=5)
    assert "mean_predicted_value" in curve
    assert "fraction_of_positives" in curve


def test_model_calibrator():
    rng = np.random.RandomState(42)
    X_tr = pd.DataFrame(rng.normal(size=(100, 4)), columns=[f"f{i}" for i in range(4)])
    y_tr = rng.choice([0, 1], size=100)

    X_val = pd.DataFrame(rng.normal(size=(50, 4)), columns=[f"f{i}" for i in range(4)])
    y_val = rng.choice([0, 1], size=50)

    X_test = pd.DataFrame(rng.normal(size=(30, 4)), columns=[f"f{i}" for i in range(4)])
    y_test = rng.choice([0, 1], size=30)

    model = RandomForestWrapper(params={"n_estimators": 5, "random_state": 42})
    model.fit(X_tr, y_tr)

    calibrator = ModelCalibrator(n_bins=5)
    calibrator.fit(model, X_val, y_val, methods=["sigmoid", "isotonic"])

    probs_sig = calibrator.predict_proba("sigmoid", X_test)
    probs_iso = calibrator.predict_proba("isotonic", X_test)
    assert probs_sig.shape == (30, 2)
    assert probs_iso.shape == (30, 2)

    reports = calibrator.evaluate(model, X_test, y_test, model_name="rf_test")
    assert len(reports) == 3  # uncalibrated, sigmoid, isotonic
    df_rep = ModelCalibrator.reports_to_dataframe(reports)
    assert len(df_rep) == 3
