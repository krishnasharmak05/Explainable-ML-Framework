from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd


class DataPreprocessor:
    """
    Module 1: Data Preprocessing and Normalization
    
    Implements:
    - Infinite value handling (capping and converting to NaN)
    - Equation (1): Missing value imputation:
        x'_{i,j} = x_{i,j} if observed, else mu_j (mean or median)
    - Equation (2): Logarithmic transformation for highly skewed features:
        X_{log} = log(X + 1)
    - Equation (3): Min-Max scaling to standard range [0, 1]:
        X_{norm} = (X - X_min) / (X_max - X_min)
    - Target label encoding (Binary Benign vs Attack, or Multi-class)
    """

    def __init__(
        self,
        imputation_strategy: str = "mean",
        skewness_threshold: float = 1.0,
        scaling_range: Tuple[float, float] = (0.0, 1.0),
        drop_columns: Optional[List[str]] = None,
        classification_mode: str = "binary",
        benign_label: str = "BENIGN",
    ):
        if imputation_strategy not in ("mean", "median"):
            raise ValueError(f"Unknown imputation strategy: {imputation_strategy}")
        if classification_mode not in ("binary", "multiclass"):
            raise ValueError(f"Classification mode must be 'binary' or 'multiclass', got {classification_mode}")

        self.imputation_strategy = imputation_strategy
        self.skewness_threshold = skewness_threshold
        self.scaling_range = scaling_range
        self.drop_columns = drop_columns or []
        self.classification_mode = classification_mode
        self.benign_label = benign_label.upper().strip()

        # State learned during fit()
        self.imputation_values_: Dict[str, float] = {}
        self.skewed_columns_: List[str] = []
        self.min_values_: Dict[str, float] = {}
        self.max_values_: Dict[str, float] = {}
        self.feature_names_: List[str] = []
        self.label_mapping_: Dict[str, int] = {}
        self.inverse_label_mapping_: Dict[int, str] = {}
        self.is_fitted_: bool = False

    def clean_raw_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans column names by trimming whitespaces and removes drop_columns.
        """
        cleaned = df.copy()
        cleaned.columns = cleaned.columns.astype(str).str.strip()
        cols_to_drop = [c for c in self.drop_columns if c in cleaned.columns]
        if cols_to_drop:
            cleaned = cleaned.drop(columns=cols_to_drop)
        return cleaned

    def encode_labels(self, y: pd.Series, fit: bool = False) -> np.ndarray:
        """
        Encodes labels for either binary (BENIGN -> 0, Attack -> 1) or multiclass detection.
        """
        y_str = y.astype(str).str.strip()

        if self.classification_mode == "binary":
            # 0 for Benign, 1 for any malicious attack
            encoded = (y_str.str.upper() != self.benign_label).astype(int).values
            if fit:
                self.label_mapping_ = {self.benign_label: 0, "ATTACK": 1}
                self.inverse_label_mapping_ = {0: self.benign_label, 1: "ATTACK"}
            return encoded
        else:
            # Multiclass mode: BENIGN -> 0, other attack classes sorted alphabetically
            if fit:
                unique_classes = sorted(y_str.unique().tolist())
                # Ensure BENIGN is always index 0
                classes_order = []
                for c in unique_classes:
                    if c.upper() == self.benign_label:
                        classes_order.insert(0, c)
                    else:
                        classes_order.append(c)

                self.label_mapping_ = {cls_name: idx for idx, cls_name in enumerate(classes_order)}
                self.inverse_label_mapping_ = {idx: cls_name for cls_name, idx in self.label_mapping_.items()}

            return y_str.map(self.label_mapping_).fillna(1).astype(int).values

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "DataPreprocessor":
        """
        Fits preprocessing parameters strictly on training features X.
        """
        X_clean = self.clean_raw_dataframe(X)

        # Convert infinities to NaN
        X_numeric = X_clean.select_dtypes(include=[np.number]).copy()
        X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan)
        self.feature_names_ = list(X_numeric.columns)

        # 1. Missing Value Imputation (Equation 1)
        if self.imputation_strategy == "mean":
            self.imputation_values_ = X_numeric.mean().to_dict()
        else:
            self.imputation_values_ = X_numeric.median().to_dict()

        # Impute temporarily on training copy to assess skewness & scaling parameters
        X_imputed = X_numeric.fillna(self.imputation_values_)

        # 2. Skewness Detection for Log-transform (Equation 2)
        self.skewed_columns_ = []
        for col in self.feature_names_:
            # Calculate sample skewness
            skew_val = X_imputed[col].skew()
            if not np.isnan(skew_val) and abs(skew_val) > self.skewness_threshold:
                self.skewed_columns_.append(col)

        # Apply log(X + 1) to skewed columns to compute proper min/max for scaling
        X_transformed = X_imputed.copy()
        for col in self.skewed_columns_:
            # Network features are typically non-negative; ensure non-negativity
            col_vals = X_transformed[col]
            min_val = col_vals.min()
            shift = abs(min_val) if min_val < 0 else 0.0
            X_transformed[col] = np.log1p(np.maximum(col_vals + shift, 0.0))

        # 3. Min-Max Normalization (Equation 3)
        self.min_values_ = X_transformed.min().to_dict()
        self.max_values_ = X_transformed.max().to_dict()

        if y is not None:
            self.encode_labels(y, fit=True)

        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms features X using parameters learned during fit():
        - Infinite values capped / converted to NaN
        - Imputation (Eq 1)
        - Log transformation (Eq 2)
        - Min-Max scaling (Eq 3)
        """
        if not self.is_fitted_:
            raise RuntimeError("DataPreprocessor must be fitted before transforming data.")

        X_clean = self.clean_raw_dataframe(X)
        
        # Ensure only known numeric features are processed
        X_numeric = X_clean[self.feature_names_].copy()
        X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan)

        # Imputation (Equation 1)
        X_imputed = X_numeric.fillna(self.imputation_values_)

        # Logarithmic Transformation (Equation 2)
        X_transformed = X_imputed.copy()
        for col in self.skewed_columns_:
            col_vals = X_transformed[col]
            min_val = self.min_values_.get(col, 0.0)
            shift = abs(min_val) if min_val < 0 else 0.0
            X_transformed[col] = np.log1p(np.maximum(col_vals + shift, 0.0))

        # Min-Max Scaling (Equation 3)
        range_min, range_max = self.scaling_range
        scaled_df = pd.DataFrame(index=X_transformed.index)

        for col in self.feature_names_:
            col_min = self.min_values_.get(col, 0.0)
            col_max = self.max_values_.get(col, 1.0)
            denominator = col_max - col_min

            if denominator == 0 or np.isnan(denominator):
                # Constant feature across training set
                scaled_col = np.zeros(len(X_transformed))
            else:
                scaled_col = (X_transformed[col] - col_min) / denominator
                if (range_min, range_max) != (0.0, 1.0):
                    scaled_col = scaled_col * (range_max - range_min) + range_min

            # Clip within range to protect against unseen extremes in test data
            scaled_df[col] = np.clip(scaled_col, range_min, range_max)

        return scaled_df

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fits preprocessor and transforms input features."""
        return self.fit(X, y).transform(X)

    def save(self, file_path: Union[str, Path]) -> None:
        """Saves fitted preprocessor state using joblib."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, file_path)

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "DataPreprocessor":
        """Loads fitted preprocessor state from disk."""
        return joblib.load(Path(file_path))
