from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from src.calibration.metrics import compute_brier_score, compute_ece, compute_calibration_curve


@dataclass
class CalibrationReport:
    """Stores comparative calibration metrics across methods."""
    model_name: str
    method: str
    brier_score: float
    ece: float
    brier_score_improvement: float  # baseline - calibrated (positive is better)
    ece_improvement: float


class ModelCalibrator:
    """
    Phase 4: Probability Calibration Analysis
    Supports:
    - Uncalibrated baseline
    - Platt Scaling ('sigmoid'): Logistic regression on validation probabilities / logits
    - Isotonic Regression ('isotonic'): Non-parametric monotonic calibration
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.fitted_calibrators_: Dict[str, Any] = {}
        self.is_binary_: bool = True

    def fit(
        self,
        base_estimator: Any,
        X_val: Union[pd.DataFrame, np.ndarray],
        y_val: Union[pd.Series, np.ndarray],
        methods: Optional[List[str]] = None,
    ) -> "ModelCalibrator":
        """
        Fits probability calibrators on held-out validation data using the
        pre-fitted base estimator without data leakage.
        """
        self.base_estimator_ = base_estimator
        if methods is None:
            methods = ["sigmoid", "isotonic"]

        y_v = np.asarray(y_val)
        val_probs = base_estimator.predict_proba(X_val)

        # Check if binary or multiclass
        if val_probs.ndim == 1 or val_probs.shape[1] == 2:
            self.is_binary_ = True
            p = val_probs[:, 1] if val_probs.ndim == 2 else val_probs.ravel()
            eps = 1e-7
            p_clipped = np.clip(p, eps, 1.0 - eps)
            logits = np.log(p_clipped / (1.0 - p_clipped)).reshape(-1, 1)

            for method in methods:
                if method == "sigmoid":
                    # Platt scaling: univariate logistic regression on logits
                    lr = LogisticRegression(C=1.0, solver="lbfgs")
                    lr.fit(logits, y_v)
                    self.fitted_calibrators_["sigmoid"] = lr

                elif method == "isotonic":
                    # Isotonic regression: non-parametric monotonic mapping
                    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                    iso.fit(p, y_v)
                    self.fitted_calibrators_["isotonic"] = iso
        else:
            self.is_binary_ = False
            for method in methods:
                if method == "sigmoid":
                    # Multi-class Platt scaling: multinomial logistic regression on probability features
                    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200)
                    lr.fit(val_probs, y_v)
                    self.fitted_calibrators_["sigmoid"] = lr
                elif method == "isotonic":
                    # One-vs-rest Isotonic regressors per class
                    iso_dict = {}
                    n_classes = val_probs.shape[1]
                    for c in range(n_classes):
                        iso_c = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                        iso_c.fit(val_probs[:, c], (y_v == c).astype(float))
                        iso_dict[c] = iso_c
                    self.fitted_calibrators_["isotonic"] = iso_dict

        return self

    def predict_proba(
        self,
        method: str,
        X: Union[pd.DataFrame, np.ndarray],
        base_estimator: Optional[Any] = None,
    ) -> np.ndarray:
        """
        Returns calibrated predicted probabilities.
        """
        estimator = base_estimator or getattr(self, "base_estimator_", None)
        if estimator is None:
            raise ValueError("No base_estimator provided or fitted.")

        raw_probs = estimator.predict_proba(X)

        if method == "uncalibrated":
            return raw_probs

        if method not in self.fitted_calibrators_:
            raise ValueError(f"Method '{method}' is not fitted. Fitted: {list(self.fitted_calibrators_.keys())}")

        calibrator = self.fitted_calibrators_[method]

        if self.is_binary_:
            p = raw_probs[:, 1] if raw_probs.ndim == 2 else raw_probs.ravel()
            eps = 1e-7
            p_clipped = np.clip(p, eps, 1.0 - eps)

            if method == "sigmoid":
                logits = np.log(p_clipped / (1.0 - p_clipped)).reshape(-1, 1)
                calibrated_probs = calibrator.predict_proba(logits)
                return calibrated_probs
            elif method == "isotonic":
                cal_p1 = calibrator.predict(p)
                cal_p0 = 1.0 - cal_p1
                return np.column_stack([cal_p0, cal_p1])
        else:
            if method == "sigmoid":
                return calibrator.predict_proba(raw_probs)
            elif method == "isotonic":
                n_classes = raw_probs.shape[1]
                cal_probs = np.zeros_like(raw_probs)
                for c in range(n_classes):
                    cal_probs[:, c] = calibrator[c].predict(raw_probs[:, c])
                # Normalize across classes
                row_sums = cal_probs.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1.0
                return cal_probs / row_sums

        return raw_probs

    def evaluate(
        self,
        base_estimator: Any,
        X_eval: Union[pd.DataFrame, np.ndarray],
        y_eval: Union[pd.Series, np.ndarray],
        model_name: str = "model",
    ) -> List[CalibrationReport]:
        """
        Evaluates calibration metrics on evaluation data across all fitted methods.
        """
        y_eval_arr = np.asarray(y_eval)
        reports = []

        # 1. Baseline uncalibrated evaluation
        base_probs = base_estimator.predict_proba(X_eval)
        base_brier = compute_brier_score(y_eval_arr, base_probs)
        base_ece = compute_ece(y_eval_arr, base_probs, n_bins=self.n_bins)

        reports.append(
            CalibrationReport(
                model_name=model_name,
                method="uncalibrated",
                brier_score=base_brier,
                ece=base_ece,
                brier_score_improvement=0.0,
                ece_improvement=0.0,
            )
        )

        # 2. Calibrated methods
        for method in self.fitted_calibrators_.keys():
            cal_probs = self.predict_proba(method, X_eval, base_estimator=base_estimator)
            brier = compute_brier_score(y_eval_arr, cal_probs)
            ece = compute_ece(y_eval_arr, cal_probs, n_bins=self.n_bins)

            reports.append(
                CalibrationReport(
                    model_name=model_name,
                    method=method,
                    brier_score=brier,
                    ece=ece,
                    brier_score_improvement=base_brier - brier,
                    ece_improvement=base_ece - ece,
                )
            )

        return reports

    @staticmethod
    def reports_to_dataframe(reports: List[CalibrationReport]) -> pd.DataFrame:
        """Formats calibration reports into a readable DataFrame."""
        return pd.DataFrame([
            {
                "Model": r.model_name,
                "Calibration Method": r.method,
                "Brier Score": f"{r.brier_score:.5f}",
                "ECE": f"{r.ece:.5f}",
                "Brier Improvement": f"{r.brier_score_improvement:+.5f}",
                "ECE Improvement": f"{r.ece_improvement:+.5f}",
            }
            for r in reports
        ])

    def save(self, file_path: Union[str, Path]) -> None:
        """Serializes fitted calibrators to disk."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "ModelCalibrator":
        """Loads fitted calibrator from disk."""
        return joblib.load(Path(file_path))
