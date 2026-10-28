from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import brier_score_loss


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Computes Brier Score for binary or multi-class predictions.
    Lower score indicates better calibrated and more accurate probabilities.
    """
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)

    if y_prob_arr.ndim == 1 or (y_prob_arr.ndim == 2 and y_prob_arr.shape[1] == 1):
        # Binary with 1D probability array
        probs = y_prob_arr.ravel()
        return float(brier_score_loss(y_true_arr, probs))

    elif y_prob_arr.ndim == 2 and y_prob_arr.shape[1] == 2:
        # Binary with 2D probability array [p0, p1]
        probs = y_prob_arr[:, 1]
        return float(brier_score_loss(y_true_arr, probs))

    else:
        # Multi-class Brier score: (1/N) * sum_i sum_c (p_ic - y_ic)^2
        n_samples, n_classes = y_prob_arr.shape
        # One-hot encode y_true
        y_one_hot = np.zeros((n_samples, n_classes), dtype=float)
        for i, val in enumerate(y_true_arr):
            if 0 <= int(val) < n_classes:
                y_one_hot[i, int(val)] = 1.0
        return float(np.mean(np.sum((y_prob_arr - y_one_hot) ** 2, axis=1)))


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Computes Expected Calibration Error (ECE) across M bins.
    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)
    n_samples = len(y_true_arr)

    if n_samples == 0:
        return 0.0

    if y_prob_arr.ndim == 1 or (y_prob_arr.ndim == 2 and y_prob_arr.shape[1] == 1):
        confidences = y_prob_arr.ravel()
        accuracies = (y_true_arr == 1).astype(float)
    elif y_prob_arr.ndim == 2 and y_prob_arr.shape[1] == 2:
        confidences = y_prob_arr[:, 1]
        accuracies = (y_true_arr == 1).astype(float)
    else:
        # Multi-class confidence is max predicted probability
        confidences = np.max(y_prob_arr, axis=1)
        predictions = np.argmax(y_prob_arr, axis=1)
        accuracies = (predictions == y_true_arr).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_size / n_samples) * abs(bin_acc - bin_conf)

    return float(ece)


def compute_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, np.ndarray]:
    """
    Computes empirical accuracy and average confidence across bins for reliability curves.
    Supports both binary and multi-class predictions.
    """
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)

    unique_classes = np.unique(y_true_arr)
    is_binary = len(unique_classes) <= 2 and (y_prob_arr.ndim == 1 or y_prob_arr.shape[1] <= 2)

    if is_binary:
        if set(unique_classes) != {0, 1}:
            pos_label = unique_classes[-1]
            correct_or_pos = (y_true_arr == pos_label).astype(float)
        else:
            correct_or_pos = (y_true_arr == 1).astype(float)

        if y_prob_arr.ndim == 2 and y_prob_arr.shape[1] == 2:
            probs = y_prob_arr[:, 1]
        else:
            probs = y_prob_arr.ravel()
    else:
        probs = np.max(y_prob_arr, axis=1)
        preds = np.argmax(y_prob_arr, axis=1)
        correct_or_pos = (preds == y_true_arr).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            in_bin = (probs >= bin_lower) & (probs <= bin_upper)
        else:
            in_bin = (probs >= bin_lower) & (probs < bin_upper)

        if np.any(in_bin):
            bin_accs.append(float(np.mean(correct_or_pos[in_bin])))
            bin_confs.append(float(np.mean(probs[in_bin])))
            bin_counts.append(int(np.sum(in_bin)))
        else:
            bin_accs.append(0.0)
            bin_confs.append((bin_lower + bin_upper) / 2.0)
            bin_counts.append(0)

    return {
        "mean_predicted_value": np.array(bin_confs),
        "fraction_of_positives": np.array(bin_accs),
        "bin_counts": np.array(bin_counts),
    }
