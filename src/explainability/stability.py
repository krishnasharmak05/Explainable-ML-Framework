from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


@dataclass
class StabilityResults:
    """Stores explanation stability metrics and consensus statistics across CV folds."""
    model_name: str
    top_k: int
    n_folds: int
    mean_spearman_rho: float
    std_spearman_rho: float
    mean_kendall_tau: float
    std_kendall_tau: float
    mean_jaccard_similarity: float
    std_jaccard_similarity: float
    pairwise_spearman_matrix: pd.DataFrame
    pairwise_kendall_matrix: pd.DataFrame
    pairwise_jaccard_matrix: pd.DataFrame
    top_k_consensus_features: pd.DataFrame
    feature_rank_distribution: pd.DataFrame

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "top_k": self.top_k,
            "n_folds": self.n_folds,
            "mean_spearman_rho": self.mean_spearman_rho,
            "std_spearman_rho": self.std_spearman_rho,
            "mean_kendall_tau": self.mean_kendall_tau,
            "std_kendall_tau": self.std_kendall_tau,
            "mean_jaccard_similarity": self.mean_jaccard_similarity,
            "std_jaccard_similarity": self.std_jaccard_similarity,
        }


class ExplanationStabilityAnalyzer:
    """
    Phase 5: Explanation Stability Analysis.
    Measures consistency of feature attributions across cross-validation folds.
    """

    def __init__(self, top_k: int = 10):
        self.top_k = top_k

    def analyze_stability(
        self,
        fold_importances: List[pd.DataFrame],
        model_name: str = "model",
    ) -> StabilityResults:
        """
        Calculates Spearman, Kendall, and Jaccard stability metrics
        across fold-level feature importance dataframes.
        Each dataframe must contain 'Feature' and 'Mean_Absolute_SHAP' (or 'Rank').
        """
        n_folds = len(fold_importances)
        if n_folds < 2:
            raise ValueError(f"Need at least 2 folds to compute stability, got {n_folds}")

        # Ensure all dfs are sorted and ranked
        ranked_dfs = []
        for df in fold_importances:
            sorted_df = df.sort_values(by="Mean_Absolute_SHAP", ascending=False).reset_index(drop=True)
            sorted_df["Rank"] = range(1, len(sorted_df) + 1)
            ranked_dfs.append(sorted_df)

        all_features = sorted(list(ranked_dfs[0]["Feature"].unique()))

        # Build pairwise matrices
        spearman_mat = np.zeros((n_folds, n_folds))
        kendall_mat = np.zeros((n_folds, n_folds))
        jaccard_mat = np.zeros((n_folds, n_folds))

        spearman_vals = []
        kendall_vals = []
        jaccard_vals = []

        for i in range(n_folds):
            df_i = ranked_dfs[i].set_index("Feature")
            top_k_i = set(ranked_dfs[i]["Feature"].iloc[:self.top_k])

            for j in range(n_folds):
                df_j = ranked_dfs[j].set_index("Feature")
                top_k_j = set(ranked_dfs[j]["Feature"].iloc[:self.top_k])

                if i == j:
                    spearman_mat[i, j] = 1.0
                    kendall_mat[i, j] = 1.0
                    jaccard_mat[i, j] = 1.0
                else:
                    # Ranks aligned across all features
                    ranks_i = [df_i.loc[f, "Rank"] for f in all_features]
                    ranks_j = [df_j.loc[f, "Rank"] for f in all_features]

                    rho, _ = spearmanr(ranks_i, ranks_j)
                    tau, _ = kendalltau(ranks_i, ranks_j)

                    # Jaccard index: |A inter B| / |A union B|
                    intersection = len(top_k_i.intersection(top_k_j))
                    union = len(top_k_i.union(top_k_j))
                    jaccard = intersection / union if union > 0 else 0.0

                    spearman_mat[i, j] = rho
                    kendall_mat[i, j] = tau
                    jaccard_mat[i, j] = jaccard

                    if i < j:
                        spearman_vals.append(rho)
                        kendall_vals.append(tau)
                        jaccard_vals.append(jaccard)

        # Top-k Consensus
        feature_counts: Dict[str, int] = {}
        for df in ranked_dfs:
            for feat in df["Feature"].iloc[:self.top_k]:
                feature_counts[feat] = feature_counts.get(feat, 0) + 1

        consensus_df = (
            pd.DataFrame([
                {"Feature": f, "Appearance_In_Top_K": count, "Selection_Frequency": count / n_folds}
                for f, count in feature_counts.items()
            ])
            .sort_values(by="Appearance_In_Top_K", ascending=False)
            .reset_index(drop=True)
        )

        # Feature Rank Distribution across folds
        rank_records = []
        for feat in all_features:
            ranks = [df.set_index("Feature").loc[feat, "Rank"] for df in ranked_dfs]
            rank_records.append({
                "Feature": feat,
                "Mean_Rank": float(np.mean(ranks)),
                "Std_Rank": float(np.std(ranks)),
                "Min_Rank": int(np.min(ranks)),
                "Max_Rank": int(np.max(ranks)),
            })
        rank_dist_df = pd.DataFrame(rank_records).sort_values(by="Mean_Rank").reset_index(drop=True)

        fold_labels = [f"Fold_{i+1}" for i in range(n_folds)]

        return StabilityResults(
            model_name=model_name,
            top_k=self.top_k,
            n_folds=n_folds,
            mean_spearman_rho=float(np.mean(spearman_vals)),
            std_spearman_rho=float(np.std(spearman_vals)),
            mean_kendall_tau=float(np.mean(kendall_vals)),
            std_kendall_tau=float(np.std(kendall_vals)),
            mean_jaccard_similarity=float(np.mean(jaccard_vals)),
            std_jaccard_similarity=float(np.std(jaccard_vals)),
            pairwise_spearman_matrix=pd.DataFrame(spearman_mat, index=fold_labels, columns=fold_labels),
            pairwise_kendall_matrix=pd.DataFrame(kendall_mat, index=fold_labels, columns=fold_labels),
            pairwise_jaccard_matrix=pd.DataFrame(jaccard_mat, index=fold_labels, columns=fold_labels),
            top_k_consensus_features=consensus_df,
            feature_rank_distribution=rank_dist_df,
        )
