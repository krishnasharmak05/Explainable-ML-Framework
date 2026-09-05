import numpy as np
import pandas as pd
import pytest
from src.statistics.bootstrap import compute_bootstrap_ci, bootstrap_results_to_dataframe
from src.statistics.hypothesis_tests import run_mcnemar_test, run_wilcoxon_test
from src.statistics.omnibus import run_friedman_test, run_nemenyi_posthoc


def test_bootstrap_ci():
    rng = np.random.RandomState(42)
    n = 100
    y_true = rng.choice([0, 1], size=n, p=[0.7, 0.3])
    y_prob = np.clip(y_true * 0.7 + rng.uniform(0.1, 0.4, size=n), 0.0, 1.0)
    y_pred = (y_prob > 0.5).astype(int)

    results = compute_bootstrap_ci(
        y_true, y_pred, y_prob, n_bootstraps=50, confidence_level=0.95, random_state=42
    )

    assert "accuracy" in results
    assert "f1" in results
    assert "mcc" in results

    acc_res = results["accuracy"]
    assert 0.0 <= acc_res.ci_lower <= acc_res.ci_upper <= 1.0
    assert 0.0 <= acc_res.point_estimate <= 1.0

    df = bootstrap_results_to_dataframe(results)
    assert len(df) == len(results)


def test_mcnemar_test():
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 0] * 5)
    # Model A is mostly correct
    y_pred_a = y_true.copy()
    y_pred_a[0] = 0
    # Model B makes several mistakes
    y_pred_b = y_true.copy()
    y_pred_b[1:8] = 1 - y_pred_b[1:8]

    res = run_mcnemar_test(y_true, y_pred_a, y_pred_b, "Model_A", "Model_B")

    assert res.test_name == "McNemar Test"
    assert res.statistic >= 0.0
    assert 0.0 <= res.p_value <= 1.0
    assert res.effect_size >= 0.0
    assert "both_correct" in res.contingency_counts


def test_wilcoxon_test():
    scores_a = [0.95, 0.94, 0.96, 0.93, 0.95, 0.97, 0.94, 0.96]
    scores_b = [0.85, 0.83, 0.87, 0.82, 0.84, 0.86, 0.81, 0.85]

    res = run_wilcoxon_test(scores_a, scores_b, "Model_A", "Model_B")

    assert res.test_name == "Wilcoxon Signed-Rank Test"
    assert 0.0 <= res.p_value <= 1.0
    assert res.is_statistically_significant
    assert res.effect_size > 0.0


def test_friedman_and_nemenyi():
    # 3 models evaluated across 5 folds
    scores = {
        "Model_1": [0.10, 0.12, 0.09, 0.11, 0.10],
        "Model_2": [0.15, 0.14, 0.16, 0.15, 0.14],
        "Model_3": [0.30, 0.28, 0.32, 0.29, 0.31],
    }

    friedman_res = run_friedman_test(scores, higher_is_better=False, alpha=0.05)

    assert friedman_res.statistic_chi2 > 0.0
    assert 0.0 <= friedman_res.p_value_chi2 <= 1.0
    assert friedman_res.is_significant
    assert len(friedman_res.average_ranks) == 3

    posthoc_df = run_nemenyi_posthoc(friedman_res, n_folds=5)
    assert len(posthoc_df) == 3  # (3 choose 2) pairs
    assert "Critical_Difference_CD" in posthoc_df.columns
    assert "Statistically_Significant" in posthoc_df.columns
