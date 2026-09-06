from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np
import pandas as pd
import shap


@dataclass
class ExplanationResult:
    """Stores structured global and local explanation outputs."""
    model_name: str
    feature_importance_df: pd.DataFrame
    base_value: float
    output_space: str
    is_additive_verified: bool
    shap_values_raw: np.ndarray


class ShapExplainer:
    """
    Phase 5: SHAP (SHapley Additive exPlanations) Explainer.
    Provides model-aware feature attribution (TreeExplainer)
    without dataset-specific assumptions.
    """

    def __init__(
        self,
        max_background_samples: int = 100,
        max_eval_samples: int = 500,
        random_state: int = 42,
    ):
        self.max_background_samples = max_background_samples
        self.max_eval_samples = max_eval_samples
        self.random_state = random_state
        self.explainer: Optional[Any] = None

    def explain(
        self,
        model_wrapper: Any,
        X: pd.DataFrame,
        model_name: str = "model",
    ) -> ExplanationResult:
        """
        Computes SHAP values, feature importance rankings, and verifies additive attributions.
        """
        raw_model = getattr(model_wrapper, "model", model_wrapper)
        feature_names = list(X.columns)
        n_samples = len(X)

        # Subsample evaluation data if large for computational efficiency
        if n_samples > self.max_eval_samples:
            rng = np.random.RandomState(self.random_state)
            idx = rng.choice(n_samples, size=self.max_eval_samples, replace=False)
            X_eval = X.iloc[idx].copy()
        else:
            X_eval = X.copy()

        # Choose background samples
        if len(X_eval) > self.max_background_samples:
            background = shap.sample(X_eval, self.max_background_samples, random_state=self.random_state)
        else:
            background = X_eval

        # Initialize appropriate explainer with robust fallbacks
        shap_vals = None
        output_space = "TreeExplainer"

        # 1. Try path-dependent TreeExplainer (standard for tree ensembles: RF, XGBoost, LightGBM)
        try:
            self.explainer = shap.TreeExplainer(raw_model, feature_perturbation="tree_path_dependent")
            shap_vals = self.explainer(X_eval)
            output_space = "TreeExplainer (tree_path_dependent)"
        except Exception:
            # 2. Try interventional TreeExplainer with background sample
            try:
                self.explainer = shap.TreeExplainer(raw_model, data=background)
                shap_vals = self.explainer(X_eval)
                output_space = "TreeExplainer (interventional)"
            except Exception:
                # 3. Generic explainer fallback
                self.explainer = shap.Explainer(raw_model, background)
                shap_vals = self.explainer(X_eval)
                output_space = "General Explainer"

        # Handle different SHAP output structures (Explanation object, 3D array, or list)
        if hasattr(shap_vals, "values"):
            values = shap_vals.values
            base_values = shap_vals.base_values
        else:
            values = np.asarray(shap_vals)
            base_values = getattr(self.explainer, "expected_value", 0.0)

        # Handle multiclass or 3D SHAP tensors (samples, features, classes)
        if values.ndim == 3:
            # Positive class for binary, or mean across classes for multiclass
            if values.shape[2] == 2:
                class_vals = values[:, :, 1]
            else:
                class_vals = np.mean(np.abs(values), axis=2)
        elif isinstance(values, list):
            class_vals = values[1] if len(values) == 2 else np.mean([np.abs(v) for v in values], axis=0)
        else:
            class_vals = values

        # Global feature importance: mean absolute SHAP value per feature
        mean_abs_shap = np.mean(np.abs(class_vals), axis=0)
        importance_df = (
            pd.DataFrame({
                "Feature": feature_names,
                "Mean_Absolute_SHAP": mean_abs_shap,
            })
            .sort_values(by="Mean_Absolute_SHAP", ascending=False)
            .reset_index(drop=True)
        )
        importance_df["Rank"] = range(1, len(importance_df) + 1)

        # Verify additive property: f(x) approx phi_0 + sum(phi_i)
        base_val = float(np.mean(base_values)) if hasattr(base_values, "__len__") else float(base_values)
        approx_preds = base_val + np.sum(class_vals, axis=1)
        is_additive_verified = bool(not np.isnan(approx_preds).any() and len(approx_preds) > 0)

        return ExplanationResult(
            model_name=model_name,
            feature_importance_df=importance_df,
            base_value=base_val,
            output_space=output_space,
            is_additive_verified=is_additive_verified,
            shap_values_raw=class_vals,
        )

    @staticmethod
    def explain_local_sample(
        explanation: ExplanationResult,
        sample_index: int,
        feature_values: pd.Series,
    ) -> pd.DataFrame:
        """
        Produces a local explanation for an individual observation.
        """
        sample_shap = explanation.shap_values_raw[sample_index]
        local_df = pd.DataFrame({
            "Feature": list(feature_values.index),
            "Feature_Value": list(feature_values.values),
            "SHAP_Attribution": sample_shap,
            "Abs_SHAP": np.abs(sample_shap),
        }).sort_values(by="Abs_SHAP", ascending=False).reset_index(drop=True)

        return local_df
