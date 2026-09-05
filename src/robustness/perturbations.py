from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def apply_feature_noise(
    X: pd.DataFrame,
    noise_level: float = 0.05,
    columns: Optional[List[str]] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Injects zero-mean Gaussian noise scaled by each feature's empirical standard deviation.
    X_noisy = X + N(0, (noise_level * std(X))^2)
    """
    X_perturbed = X.copy()
    rng = np.random.RandomState(random_state)
    target_cols = columns or list(X.columns)

    for col in target_cols:
        col_std = float(X[col].std())
        if col_std == 0.0 or np.isnan(col_std):
            col_std = 1.0

        noise = rng.normal(loc=0.0, scale=noise_level * col_std, size=len(X))
        X_perturbed[col] = X_perturbed[col] + noise

    return X_perturbed


def apply_random_missingness(
    X: pd.DataFrame,
    missing_rate: float = 0.10,
    columns: Optional[List[str]] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Randomly injects NaN missing values into a configurable fraction of entries.
    """
    X_perturbed = X.copy()
    rng = np.random.RandomState(random_state)
    target_cols = columns or list(X.columns)

    for col in target_cols:
        mask = rng.binomial(n=1, p=missing_rate, size=len(X)).astype(bool)
        X_perturbed.loc[mask, col] = np.nan

    return X_perturbed


def apply_feature_corruption(
    X: pd.DataFrame,
    corrupt_columns: List[str],
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Permutes / scrambles values of specified features to break their predictive association.
    """
    X_perturbed = X.copy()
    rng = np.random.RandomState(random_state)

    for col in corrupt_columns:
        if col in X_perturbed.columns:
            permuted = rng.permutation(X_perturbed[col].values)
            X_perturbed[col] = permuted

    return X_perturbed


def subsample_training_data(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    fraction: float = 0.5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Subsamples training data to evaluate model performance degradation under reduced data.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"Fraction must be between (0, 1], got {fraction}")

    n_samples = len(X_train)
    subset_size = max(2, int(n_samples * fraction))

    rng = np.random.RandomState(random_state)
    indices = rng.choice(n_samples, size=subset_size, replace=False)

    return X_train.iloc[indices].copy(), y_train[indices].copy()
