from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from src.calibration.metrics import compute_brier_score


@dataclass
class BootstrapResult:
    metric_name: str
    point_estimate: float
    bootstrap_mean: float
    bootstrap_std: float
    ci_lower: float
    ci_upper: float
    confidence_level: float


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> Dict[str, BootstrapResult]:
    """
    Computes non-parametric percentile bootstrap confidence intervals
    for classification evaluation metrics.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    y_prob_arr = np.asarray(y_prob)

    n_samples = len(y_true_arr)
    rng = np.random.RandomState(random_state)
    is_binary = len(np.unique(y_true_arr)) <= 2
    avg_mode = "binary" if is_binary else "weighted"

    # Base point estimates
    def calculate_metrics(yt, yp, ypr):
        acc = accuracy_score(yt, yp)
        bacc = balanced_accuracy_score(yt, yp)
        prec = precision_score(yt, yp, average=avg_mode, zero_division=0)
        rec = recall_score(yt, yp, average=avg_mode, zero_division=0)
        f1 = f1_score(yt, yp, average=avg_mode, zero_division=0)
        mcc = matthews_corrcoef(yt, yp)

        try:
            if is_binary:
                probs_pos = ypr[:, 1] if ypr.ndim == 2 else ypr.ravel()
                auc = roc_auc_score(yt, probs_pos)
            else:
                auc = roc_auc_score(yt, ypr, multi_class="ovr")
        except Exception:
            auc = np.nan

        brier = compute_brier_score(yt, ypr)

        return {
            "accuracy": acc,
            "balanced_accuracy": bacc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "mcc": mcc,
            "roc_auc": auc,
            "brier_score": brier,
        }

    base_metrics = calculate_metrics(y_true_arr, y_pred_arr, y_prob_arr)
    metric_names = list(base_metrics.keys())
    bootstrap_samples: Dict[str, List[float]] = {m: [] for m in metric_names}

    alpha = 1.0 - confidence_level
    lower_pct = (alpha / 2.0) * 100.0
    upper_pct = (1.0 - alpha / 2.0) * 100.0

    for _ in range(n_bootstraps):
        boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
        yt_b = y_true_arr[boot_idx]
        yp_b = y_pred_arr[boot_idx]
        ypr_b = y_prob_arr[boot_idx]

        # Skip degenerate resample where only 1 class exists
        if len(np.unique(yt_b)) < 2:
            continue

        b_mets = calculate_metrics(yt_b, yp_b, ypr_b)
        for m in metric_names:
            if not np.isnan(b_mets[m]):
                bootstrap_samples[m].append(b_mets[m])

    results = {}
    for m in metric_names:
        vals = np.array(bootstrap_samples[m])
        if len(vals) > 0:
            ci_low = float(np.percentile(vals, lower_pct))
            ci_high = float(np.percentile(vals, upper_pct))
            b_mean = float(np.mean(vals))
            b_std = float(np.std(vals))
        else:
            ci_low = ci_high = b_mean = b_std = base_metrics[m]

        results[m] = BootstrapResult(
            metric_name=m,
            point_estimate=float(base_metrics[m]),
            bootstrap_mean=b_mean,
            bootstrap_std=b_std,
            ci_lower=ci_low,
            ci_upper=ci_high,
            confidence_level=confidence_level,
        )

    return results


def bootstrap_results_to_dataframe(results: Dict[str, BootstrapResult]) -> pd.DataFrame:
    """Formats bootstrap results dictionary into a readable DataFrame."""
    rows = []
    for m, r in results.items():
        rows.append({
            "Metric": r.metric_name,
            "Point Estimate": f"{r.point_estimate:.4f}",
            "Bootstrap Mean": f"{r.bootstrap_mean:.4f}",
            "Bootstrap Std": f"{r.bootstrap_std:.4f}",
            f"{int(r.confidence_level*100)}% CI Lower": f"{r.ci_lower:.4f}",
            f"{int(r.confidence_level*100)}% CI Upper": f"{r.ci_upper:.4f}",
        })
    return pd.DataFrame(rows)
