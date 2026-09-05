from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import chi2, f, friedmanchisquare, rankdata


@dataclass
class FriedmanResult:
    statistic_chi2: float
    p_value_chi2: float
    iman_davenport_f: float
    p_value_f: float
    average_ranks: Dict[str, float]
    is_significant: bool
    alpha: float


# Studentized range critical values for Nemenyi test at alpha=0.05
# Approximate q_alpha values for K = 2..10 models
NEMENYI_Q_ALPHA_005 = {
    2: 1.960,
    3: 2.343,
    4: 2.569,
    5: 2.728,
    6: 2.850,
    7: 2.949,
    8: 3.031,
    9: 3.102,
    10: 3.164,
}


def run_friedman_test(
    scores_dict: Dict[str, List[float]],
    higher_is_better: bool = False,  # e.g. False for log-loss, True for F1/accuracy
    alpha: float = 0.05,
) -> FriedmanResult:
    """
    Performs Friedman omnibus test and Iman-Davenport F-test across models and folds.
    """
    model_names = list(scores_dict.keys())
    k = len(model_names)
    n = len(scores_dict[model_names[0]])

    if k < 2:
        raise ValueError(f"Friedman test requires at least 2 models, got {k}")
    if n < 2:
        raise ValueError(f"Friedman test requires at least 2 folds/evaluations, got {n}")

    # Build matrix: N folds x K models
    matrix = np.zeros((n, k))
    for j, name in enumerate(model_names):
        matrix[:, j] = scores_dict[name]

    # Rank each fold across models
    # rankdata: 1 is smallest value.
    # If higher_is_better is True, multiply matrix by -1 so higher score gets rank 1.
    ranked_matrix = np.zeros_like(matrix)
    for i in range(n):
        row = -matrix[i, :] if higher_is_better else matrix[i, :]
        ranked_matrix[i, :] = rankdata(row)

    avg_ranks = np.mean(ranked_matrix, axis=0)
    avg_rank_dict = {model_names[j]: float(avg_ranks[j]) for j in range(k)}

    # Standard Chi2 statistic
    chi2_stat = float((12.0 * n / (k * (k + 1.0))) * (np.sum(avg_ranks ** 2) - (k * (k + 1.0) ** 2) / 4.0))
    p_val_chi2 = float(1.0 - chi2.cdf(chi2_stat, df=k - 1)) if 'chi2' in globals() else float(friedmanchisquare(*[matrix[:, j] for j in range(k)]).pvalue)

    # Iman-Davenport F-statistic correction
    denom = (n * (k - 1.0) - chi2_stat)
    if denom > 0:
        f_stat = float(((n - 1.0) * chi2_stat) / denom)
        df1 = k - 1
        df2 = (k - 1) * (n - 1)
        p_val_f = float(1.0 - f.cdf(f_stat, df1, df2))
    else:
        f_stat = float("inf")
        p_val_f = 0.0

    is_sig = bool(p_val_f < alpha or p_val_chi2 < alpha)

    return FriedmanResult(
        statistic_chi2=chi2_stat,
        p_value_chi2=p_val_chi2,
        iman_davenport_f=f_stat,
        p_value_f=p_val_f,
        average_ranks=avg_rank_dict,
        is_significant=is_sig,
        alpha=alpha,
    )


def run_nemenyi_posthoc(
    friedman_result: FriedmanResult,
    n_folds: int,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Computes pairwise rank differences and Critical Difference (CD)
    for Nemenyi post-hoc procedure.
    """
    avg_ranks = friedman_result.average_ranks
    models = list(avg_ranks.keys())
    k = len(models)

    q_val = NEMENYI_Q_ALPHA_005.get(k, 3.2)
    # CD = q_alpha * sqrt(k * (k + 1) / (6 * n))
    cd = float(q_val * np.sqrt((k * (k + 1.0)) / (6.0 * n_folds)))

    records = []
    for i in range(k):
        for j in range(i + 1, k):
            m1, m2 = models[i], models[j]
            rank_diff = abs(avg_ranks[m1] - avg_ranks[m2])
            is_sig = rank_diff > cd
            records.append({
                "Model_1": m1,
                "Model_2": m2,
                "Rank_1": f"{avg_ranks[m1]:.3f}",
                "Rank_2": f"{avg_ranks[m2]:.3f}",
                "Absolute_Rank_Difference": f"{rank_diff:.3f}",
                "Critical_Difference_CD": f"{cd:.3f}",
                "Statistically_Significant": is_sig,
            })

    return pd.DataFrame(records)
