from typing import Any, Dict, Optional
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from src.modeling.base import BaseModelWrapper


class RandomForestWrapper(BaseModelWrapper):
    """
    Random Forest Classifier wrapper (Equation 7: majority vote of decision trees).
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(name="random_forest", params=params)

    def _initialize_model(self) -> None:
        default_params: Dict[str, Any] = {
            "n_estimators": 100,
            "max_depth": 15,
            "random_state": 42,
            "n_jobs": -1,
        }
        merged_params = {**default_params, **self.params}
        self.model = RandomForestClassifier(**merged_params)


class XGBoostWrapper(BaseModelWrapper):
    """
    XGBoost Classifier wrapper (Equation 8 & 9: regularized gradient boosting minimizing log-loss).
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(name="xgboost", params=params)

    def _initialize_model(self) -> None:
        default_params: Dict[str, Any] = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 42,
            "eval_metric": "logloss",
            "n_jobs": -1,
        }
        merged_params = {**default_params, **self.params}
        self.model = xgb.XGBClassifier(**merged_params)


class LightGBMWrapper(BaseModelWrapper):
    """
    LightGBM Classifier wrapper (Equation 8 & 9: fast histogram-based gradient boosting).
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(name="lightgbm", params=params)

    def _initialize_model(self) -> None:
        default_params: Dict[str, Any] = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        merged_params = {**default_params, **self.params}
        self.model = lgb.LGBMClassifier(**merged_params)


def get_model(name: str, params: Optional[Dict[str, Any]] = None) -> BaseModelWrapper:
    """Factory function to instantiate models by string identifier."""
    name_clean = name.lower().strip()
    if name_clean in ("random_forest", "rf"):
        return RandomForestWrapper(params=params)
    elif name_clean in ("xgboost", "xgb"):
        return XGBoostWrapper(params=params)
    elif name_clean in ("lightgbm", "lgb"):
        return LightGBMWrapper(params=params)
    else:
        raise ValueError(f"Unsupported model name: '{name}'. Supported: 'random_forest', 'xgboost', 'lightgbm'")
