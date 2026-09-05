from pathlib import Path
from typing import Dict, List, Optional, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif


class MIFeatureSelector:
    """
    Module 2: Feature Selection using Mutual Information (Algorithm 1)
    
    Implements:
    - Equation (4): Target entropy H(Y)
    - Equation (5): Mutual Information I(X; Y)
    - Equation (6): Feature subset optimization S* = argmax_S I(S; Y)
    - Algorithm 1:
        1: Initialize S* = empty set
        2: For each feature f in F do:
        3:     Calculate probability distributions and compute I(f; Y)
        4:     Store I(f; Y) in score_map
        5: End For
        6: Sort score_map in descending order
        7: Select top k features from sorted score_map
        8: Add selected features to S*
        9: Return S*
    """

    def __init__(
        self,
        top_k: int = 15,
        n_neighbors: int = 3,
        sample_size: Optional[int] = 50000,
        random_state: int = 42,
    ):
        self.top_k = top_k
        self.n_neighbors = n_neighbors
        self.sample_size = sample_size
        self.random_state = random_state

        # State learned during fit()
        self.score_map_: Dict[str, float] = {}
        self.selected_features_: List[str] = []
        self.is_fitted_: bool = False

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> "MIFeatureSelector":
        """
        Computes Mutual Information scores for all features and selects top k (Algorithm 1).
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        y_arr = np.asarray(y)
        n_samples = len(X)

        # Subsample for computational efficiency on very large datasets if requested
        if self.sample_size and n_samples > self.sample_size:
            rng = np.random.RandomState(self.random_state)
            indices = rng.choice(n_samples, size=self.sample_size, replace=False)
            X_sub = X.iloc[indices]
            y_sub = y_arr[indices]
        else:
            X_sub = X
            y_sub = y_arr

        # Algorithm 1: Lines 2-5: Compute Mutual Information I(f; Y) for each feature
        feature_names = list(X.columns)
        mi_scores = mutual_info_classif(
            X_sub.values,
            y_sub,
            n_neighbors=self.n_neighbors,
            random_state=self.random_state,
        )

        # Store in score_map dictionary
        self.score_map_ = {
            name: float(score) for name, score in zip(feature_names, mi_scores)
        }

        # Algorithm 1: Line 7: Sort score_map in descending order based on I(f; Y)
        sorted_scores = sorted(
            self.score_map_.items(), key=lambda item: item[1], reverse=True
        )

        # Algorithm 1: Lines 8-9: Select top k features
        k = min(self.top_k, len(sorted_scores))
        self.selected_features_ = [feat for feat, _ in sorted_scores[:k]]

        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Filters input dataframe X to only retain selected features S*.
        """
        if not self.is_fitted_:
            raise RuntimeError("MIFeatureSelector must be fitted before calling transform.")

        # Reindex ensures only selected features in consistent order
        return X[self.selected_features_].copy()

    def fit_transform(
        self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]
    ) -> pd.DataFrame:
        """Fits selector and filters features."""
        return self.fit(X, y).transform(X)

    def get_selected_features(self) -> List[str]:
        """Returns the list of top-k selected feature names S*."""
        if not self.is_fitted_:
            raise RuntimeError("Selector is not fitted yet.")
        return list(self.selected_features_)

    def get_feature_scores(self) -> pd.DataFrame:
        """
        Returns a sorted DataFrame containing all features and their Mutual Information scores.
        """
        if not self.is_fitted_:
            raise RuntimeError("Selector is not fitted yet.")

        df_scores = (
            pd.DataFrame(
                list(self.score_map_.items()),
                columns=["Feature", "Mutual_Information"],
            )
            .sort_values(by="Mutual_Information", ascending=False)
            .reset_index(drop=True)
        )
        df_scores["Selected"] = df_scores["Feature"].isin(self.selected_features_)
        return df_scores

    def save_scores(self, file_path: Union[str, Path]) -> None:
        """Exports the feature scores to a CSV file."""
        df_scores = self.get_feature_scores()
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        df_scores.to_csv(file_path, index=False)

    def save(self, file_path: Union[str, Path]) -> None:
        """Saves fitted selector state using joblib."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, file_path)

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "MIFeatureSelector":
        """Loads fitted selector from disk."""
        return joblib.load(Path(file_path))
