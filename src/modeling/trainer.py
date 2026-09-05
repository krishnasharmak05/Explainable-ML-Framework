import copy
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from src.modeling.base import BaseModelWrapper


class TrainingResults:
    """Stores and summarizes results from Algorithm 2."""

    def __init__(self):
        self.results: Dict[str, Dict[str, Any]] = {}

    def add(
        self,
        model_name: str,
        model: BaseModelWrapper,
        prediction_probabilities: Optional[np.ndarray],
        cv_log_losses: List[float],
        fit_time_seconds: float,
        fold_models: Optional[List[BaseModelWrapper]] = None,
    ) -> None:
        mean_loss = float(np.mean(cv_log_losses)) if cv_log_losses else 0.0
        std_loss = float(np.std(cv_log_losses)) if cv_log_losses else 0.0

        self.results[model_name] = {
            "model": model,
            "prediction_probabilities": prediction_probabilities,
            "cv_log_losses": cv_log_losses,
            "mean_cv_log_loss": mean_loss,
            "std_cv_log_loss": std_loss,
            "training_time_seconds": fit_time_seconds,
            "fold_models": fold_models or [],
        }

    def summary(self) -> pd.DataFrame:
        """Returns a formatted DataFrame summary of all trained models."""
        rows = []
        for name, data in self.results.items():
            rows.append({
                "Model": name,
                "Mean CV Log-Loss": f"{data['mean_cv_log_loss']:.5f}",
                "Std CV Log-Loss": f"{data['std_cv_log_loss']:.5f}",
                "Training Time (s)": f"{data['training_time_seconds']:.2f}",
            })
        return pd.DataFrame(rows)

    def get_probabilities(self, model_name: str) -> Optional[np.ndarray]:
        return self.results.get(model_name, {}).get("prediction_probabilities")

    def get_model(self, model_name: str) -> Optional[BaseModelWrapper]:
        return self.results.get(model_name, {}).get("model")

    def get_fold_models(self, model_name: str) -> List[BaseModelWrapper]:
        return self.results.get(model_name, {}).get("fold_models", [])


class ModelTrainer:
    """
    Module 3: Predictive Modeling Framework
    Implements Algorithm 2: Reusable Model Training Pipeline.
    """

    def __init__(
        self,
        cv_folds: int = 5,
        random_state: int = 42,
        logger: Optional[logging.Logger] = None,
    ):
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.logger = logger or logging.getLogger(__name__)

    def train_pipeline(
        self,
        X_train: pd.DataFrame,
        y_train: Union[pd.Series, np.ndarray],
        models: List[BaseModelWrapper],
        X_test: Optional[pd.DataFrame] = None,
        save_dir: Optional[Union[str, Path]] = None,
    ) -> TrainingResults:
        """
        Executes Algorithm 2:
        Input: Processed training data (X_train, Y_train), List of initialized models M
        Output: Dictionary of trained models and their prediction probabilities
        """
        results = TrainingResults()
        y_train_arr = np.asarray(y_train)
        X_train_df = X_train if isinstance(X_train, pd.DataFrame) else pd.DataFrame(X_train)

        # 1: Initialize results dictionary R (handled by TrainingResults)
        # 2: For each model m in M do:
        for model in models:
            self.logger.info(f"--- Training Model: {model.name} ---")
            start_time = time.time()

            # 3: Initialize k-fold cross-validation
            skf = StratifiedKFold(
                n_splits=self.cv_folds,
                shuffle=True,
                random_state=self.random_state,
            )
            cv_log_losses: List[float] = []
            trained_fold_models: List[BaseModelWrapper] = []

            # 4: For each fold (train_fold, val_fold) do:
            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_df, y_train_arr), 1):
                X_tr, y_tr = X_train_df.iloc[train_idx], y_train_arr[train_idx]
                X_val, y_val = X_train_df.iloc[val_idx], y_train_arr[val_idx]

                # 5: Train m on train_fold (using fresh clone of model wrapper)
                fold_model = copy.deepcopy(model)
                fold_model.fit(X_tr, y_tr)
                trained_fold_models.append(fold_model)

                # 6: Calculate log-loss L on val_fold using Equation (9)
                val_probs = fold_model.predict_proba(X_val)
                # Compute log loss (cross-entropy)
                fold_loss = log_loss(y_val, val_probs, labels=np.unique(y_train_arr))
                cv_log_losses.append(fold_loss)
                self.logger.info(f"  [{model.name}] Fold {fold_idx}/{self.cv_folds} Log-Loss: {fold_loss:.5f}")

            mean_loss = np.mean(cv_log_losses)
            std_loss = np.std(cv_log_losses)
            self.logger.info(f"  [{model.name}] CV Log-Loss: {mean_loss:.5f} (+/- {std_loss:.5f})")

            # 8: Train final model m on full (X_train, Y_train)
            self.logger.info(f"  [{model.name}] Fitting final model on full training set...")
            model.fit(X_train_df, y_train_arr)

            # 9: Extract prediction probabilities P = m.predict_proba(X_test)
            test_probs = None
            if X_test is not None:
                self.logger.info(f"  [{model.name}] Computing test prediction probabilities...")
                test_probs = model.predict_proba(X_test)

            elapsed_time = time.time() - start_time

            # Optionally save model
            if save_dir:
                save_path = Path(save_dir) / f"{model.name}.joblib"
                model.save(save_path)
                self.logger.info(f"  [{model.name}] Saved model checkpoint to {save_path}")

            # 10: Store m and P in R
            results.add(
                model_name=model.name,
                model=model,
                prediction_probabilities=test_probs,
                cv_log_losses=cv_log_losses,
                fit_time_seconds=elapsed_time,
                fold_models=trained_fold_models,
            )

        # 12: Return R
        return results
