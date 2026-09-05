from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union
import joblib
import numpy as np
import pandas as pd


class BaseModelWrapper(ABC):
    """
    Standardized abstract base class for tree-based ensemble models in the pipeline.
    Ensures all models implement fit, predict, predict_proba, save, and load.
    """

    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        self.name = name
        self.params = params or {}
        self.model: Any = None
        self.is_fitted: bool = False
        self._initialize_model()

    @abstractmethod
    def _initialize_model(self) -> None:
        """Initializes the underlying model with self.params."""
        pass

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> "BaseModelWrapper":
        """Trains the model on features X and labels y."""
        X_val = X.values if isinstance(X, pd.DataFrame) else X
        y_val = y.values if isinstance(y, pd.Series) else y
        self.model.fit(X_val, y_val)
        self.is_fitted = True
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Generates class predictions."""
        if not self.is_fitted:
            raise RuntimeError(f"Model {self.name} is not fitted.")
        X_val = X.values if isinstance(X, pd.DataFrame) else X
        return self.model.predict(X_val)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Generates predicted class probabilities."""
        if not self.is_fitted:
            raise RuntimeError(f"Model {self.name} is not fitted.")
        X_val = X.values if isinstance(X, pd.DataFrame) else X
        return self.model.predict_proba(X_val)

    def save(self, file_path: Union[str, Path]) -> None:
        """Serializes the fitted model wrapper to disk."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, file_path)

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "BaseModelWrapper":
        """Deserializes a saved model wrapper from disk."""
        return joblib.load(Path(file_path))
