from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    log_loss,
)
from src.calibration.metrics import compute_brier_score
from src.robustness.perturbations import (
    apply_feature_corruption,
    apply_feature_noise,
    apply_random_missingness,
    subsample_training_data,
)


@dataclass
class RobustnessReport:
    model_name: str
    experiment_type: str
    severity_level: Union[float, str]
    accuracy: float
    f1: float
    mcc: float
    log_loss: float
    brier_score: float
    accuracy_degradation: float
    f1_degradation: float
    mcc_degradation: float


class RobustnessEvaluator:
    """
    Phase 6: Robustness & Perturbation Suite.
    Evaluates model resilience under controlled feature noise, missingness,
    corruption, and reduced training sample sizes.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    @staticmethod
    def _compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
    ) -> Dict[str, float]:
        """Calculates standard classification performance metrics."""
        is_binary = len(np.unique(y_true)) <= 2
        avg_mode = "binary" if is_binary else "weighted"

        acc = float(accuracy_score(y_true, y_pred))
        f1 = float(f1_score(y_true, y_pred, average=avg_mode, zero_division=0))
        mcc = float(matthews_corrcoef(y_true, y_pred))

        try:
            ll = float(log_loss(y_true, y_prob, labels=np.unique(y_true)))
        except Exception:
            ll = np.nan

        brier = compute_brier_score(y_true, y_prob)

        return {
            "accuracy": acc,
            "f1": f1,
            "mcc": mcc,
            "log_loss": ll,
            "brier_score": brier,
        }

    def evaluate_model_robustness(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        model_name: str = "model",
        noise_levels: Optional[List[float]] = None,
        missingness_rates: Optional[List[float]] = None,
        top_features_to_corrupt: Optional[List[str]] = None,
        preprocessor_impute_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    ) -> List[RobustnessReport]:
        """
        Runs comprehensive perturbation experiments on test data.
        """
        noise_levels = noise_levels or [0.01, 0.05, 0.10, 0.20]
        missingness_rates = missingness_rates or [0.05, 0.10, 0.20]
        reports: List[RobustnessReport] = []

        # 1. Baseline Evaluation (no perturbation)
        base_preds = model.predict(X_test)
        base_probs = model.predict_proba(X_test)
        base_m = self._compute_metrics(y_test, base_preds, base_probs)

        reports.append(
            RobustnessReport(
                model_name=model_name,
                experiment_type="baseline",
                severity_level=0.0,
                accuracy=base_m["accuracy"],
                f1=base_m["f1"],
                mcc=base_m["mcc"],
                log_loss=base_m["log_loss"],
                brier_score=base_m["brier_score"],
                accuracy_degradation=0.0,
                f1_degradation=0.0,
                mcc_degradation=0.0,
            )
        )

        # 2. Feature Noise Perturbations
        for sigma in noise_levels:
            X_noisy = apply_feature_noise(X_test, noise_level=sigma, random_state=self.random_state)
            preds = model.predict(X_noisy)
            probs = model.predict_proba(X_noisy)
            m = self._compute_metrics(y_test, preds, probs)

            reports.append(
                RobustnessReport(
                    model_name=model_name,
                    experiment_type="feature_noise",
                    severity_level=sigma,
                    accuracy=m["accuracy"],
                    f1=m["f1"],
                    mcc=m["mcc"],
                    log_loss=m["log_loss"],
                    brier_score=m["brier_score"],
                    accuracy_degradation=base_m["accuracy"] - m["accuracy"],
                    f1_degradation=base_m["f1"] - m["f1"],
                    mcc_degradation=base_m["mcc"] - m["mcc"],
                )
            )

        # 3. Missingness Perturbations (with pre-fitted imputer)
        if preprocessor_impute_fn is not None:
            for rate in missingness_rates:
                X_missing = apply_random_missingness(X_test, missing_rate=rate, random_state=self.random_state)
                # Pass through pre-fitted imputer
                X_imputed = preprocessor_impute_fn(X_missing)
                preds = model.predict(X_imputed)
                probs = model.predict_proba(X_imputed)
                m = self._compute_metrics(y_test, preds, probs)

                reports.append(
                    RobustnessReport(
                        model_name=model_name,
                        experiment_type="missingness",
                        severity_level=rate,
                        accuracy=m["accuracy"],
                        f1=m["f1"],
                        mcc=m["mcc"],
                        log_loss=m["log_loss"],
                        brier_score=m["brier_score"],
                        accuracy_degradation=base_m["accuracy"] - m["accuracy"],
                        f1_degradation=base_m["f1"] - m["f1"],
                        mcc_degradation=base_m["mcc"] - m["mcc"],
                    )
                )

        # 4. Feature Corruption on Top Features
        if top_features_to_corrupt:
            for feat in top_features_to_corrupt:
                if feat in X_test.columns:
                    X_corrupt = apply_feature_corruption(X_test, corrupt_columns=[feat], random_state=self.random_state)
                    preds = model.predict(X_corrupt)
                    probs = model.predict_proba(X_corrupt)
                    m = self._compute_metrics(y_test, preds, probs)

                    reports.append(
                        RobustnessReport(
                            model_name=model_name,
                            experiment_type=f"feature_corruption_{feat}",
                            severity_level="corrupted",
                            accuracy=m["accuracy"],
                            f1=m["f1"],
                            mcc=m["mcc"],
                            log_loss=m["log_loss"],
                            brier_score=m["brier_score"],
                            accuracy_degradation=base_m["accuracy"] - m["accuracy"],
                            f1_degradation=base_m["f1"] - m["f1"],
                            mcc_degradation=base_m["mcc"] - m["mcc"],
                        )
                    )

        return reports

    @staticmethod
    def reports_to_dataframe(reports: List[RobustnessReport]) -> pd.DataFrame:
        """Formats robustness reports into a DataFrame."""
        return pd.DataFrame([asdict(r) for r in reports])
