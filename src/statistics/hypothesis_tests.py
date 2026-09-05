from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.stats import binom, chi2, wilcoxon


@dataclass
class PairedComparisonResult:
    test_name: str
    model_a: str
    model_b: str
    statistic: float
    p_value: float
    is_statistically_significant: bool
    significance_threshold: float
    effect_size: float
    effect_size_name: str
    contingency_counts: Optional[Dict[str, int]] = None
    interpretation: str = ""


def run_mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    model_a_name: str = "Model_A",
    model_b_name: str = "Model_B",
    alpha: float = 0.05,
) -> PairedComparisonResult:
    """
    Performs McNemar's test for paired classification predictions on identical observations.
    Uses exact binomial test when discordant pairs (b + c) < 25,
    and Edwards continuity-corrected chi-squared test otherwise.
    """
    y_true_arr = np.asarray(y_true)
    y_a = np.asarray(y_pred_a)
    y_b = np.asarray(y_pred_b)

    correct_a = (y_a == y_true_arr)
    correct_b = (y_b == y_true_arr)

    # Contingency Table:
    # a: both correct
    # b: A correct, B incorrect
    # c: A incorrect, B correct
    # d: both incorrect
    a = int(np.sum(correct_a & correct_b))
    b = int(np.sum(correct_a & (~correct_b)))
    c = int(np.sum((~correct_a) & correct_b))
    d = int(np.sum((~correct_a) & (~correct_b)))

    discordant = b + c

    if discordant == 0:
        stat = 0.0
        p_val = 1.0
        odds_ratio = 1.0
    elif discordant < 25:
        # Exact two-sided binomial test with p = 0.5
        stat = float(min(b, c))
        p_val = float(2.0 * binom.cdf(min(b, c), discordant, 0.5))
        p_val = min(1.0, p_val)
        odds_ratio = float(b / c) if c > 0 else (float(b) if b > 0 else 1.0)
    else:
        # Chi-squared with continuity correction
        stat = float((abs(b - c) - 1.0) ** 2 / discordant)
        p_val = float(1.0 - chi2.cdf(stat, df=1))
        odds_ratio = float(b / c) if c > 0 else (float(b) if b > 0 else 1.0)

    sig = bool(p_val < alpha)
    interp = (
        f"Significant difference (p={p_val:.4f} < {alpha}) between {model_a_name} and {model_b_name}."
        if sig
        else f"No statistically significant difference (p={p_val:.4f} >= {alpha}) between {model_a_name} and {model_b_name}."
    )

    return PairedComparisonResult(
        test_name="McNemar Test",
        model_a=model_a_name,
        model_b=model_b_name,
        statistic=stat,
        p_value=p_val,
        is_statistically_significant=sig,
        significance_threshold=alpha,
        effect_size=odds_ratio,
        effect_size_name="Odds Ratio (b/c)",
        contingency_counts={"both_correct": a, "a_only": b, "b_only": c, "both_incorrect": d},
        interpretation=interp,
    )


def run_wilcoxon_test(
    scores_a: Union[List[float], np.ndarray],
    scores_b: Union[List[float], np.ndarray],
    model_a_name: str = "Model_A",
    model_b_name: str = "Model_B",
    alpha: float = 0.05,
) -> PairedComparisonResult:
    """
    Performs Wilcoxon signed-rank test on paired cross-validation fold scores.
    Computes rank-biserial correlation as effect size.
    """
    a_arr = np.asarray(scores_a)
    b_arr = np.asarray(scores_b)

    diffs = a_arr - b_arr
    non_zero = diffs[diffs != 0]

    if len(non_zero) == 0:
        return PairedComparisonResult(
            test_name="Wilcoxon Signed-Rank Test",
            model_a=model_a_name,
            model_b=model_b_name,
            statistic=0.0,
            p_value=1.0,
            is_statistically_significant=False,
            significance_threshold=alpha,
            effect_size=0.0,
            effect_size_name="Rank-Biserial Correlation",
            interpretation=f"Identical scores across folds between {model_a_name} and {model_b_name}.",
        )

    res = wilcoxon(a_arr, b_arr, zero_method="wilcox")
    stat = float(res.statistic)
    p_val = float(res.pvalue)

    # Rank-biserial correlation effect size r = 4 * |W - W_mean| / (N*(N+1))
    n = len(non_zero)
    w_mean = n * (n + 1) / 4.0
    w_max = n * (n + 1) / 2.0
    r_rb = float((stat - w_mean) / w_max) if w_max > 0 else 0.0

    sig = bool(p_val < alpha)
    interp = (
        f"Significant score distribution difference (p={p_val:.4f} < {alpha}) between {model_a_name} and {model_b_name}."
        if sig
        else f"No significant score difference (p={p_val:.4f} >= {alpha}) between {model_a_name} and {model_b_name}."
    )

    return PairedComparisonResult(
        test_name="Wilcoxon Signed-Rank Test",
        model_a=model_a_name,
        model_b=model_b_name,
        statistic=stat,
        p_value=p_val,
        is_statistically_significant=sig,
        significance_threshold=alpha,
        effect_size=abs(r_rb),
        effect_size_name="Rank-Biserial Correlation (|r|)",
        interpretation=interp,
    )
