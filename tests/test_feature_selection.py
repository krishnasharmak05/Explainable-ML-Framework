import numpy as np
import pandas as pd
import pytest
from src.feature_selection.mutual_info import MIFeatureSelector


def test_mutual_info_selection():
    rng = np.random.RandomState(42)
    n_samples = 400

    # Informative features
    y = rng.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    x_informative_1 = y * 3.0 + rng.normal(0, 0.5, size=n_samples)
    x_informative_2 = y * 5.0 + rng.normal(0, 0.5, size=n_samples)

    # Noise features
    x_noise_1 = rng.uniform(0, 10, size=n_samples)
    x_noise_2 = rng.normal(0, 1, size=n_samples)
    x_noise_3 = rng.exponential(1, size=n_samples)

    df = pd.DataFrame({
        "Info1": x_informative_1,
        "Info2": x_informative_2,
        "Noise1": x_noise_1,
        "Noise2": x_noise_2,
        "Noise3": x_noise_3,
    })

    selector = MIFeatureSelector(top_k=2, random_state=42)
    X_selected = selector.fit_transform(df, y)

    # Selected features should be Info1 and Info2
    selected_cols = selector.get_selected_features()
    assert len(selected_cols) == 2
    assert "Info1" in selected_cols
    assert "Info2" in selected_cols
    assert X_selected.shape == (n_samples, 2)

    # Check scores dataframe
    scores_df = selector.get_feature_scores()
    assert len(scores_df) == 5
    assert scores_df.iloc[0]["Mutual_Information"] >= scores_df.iloc[1]["Mutual_Information"]


def test_selector_save_load(tmp_path):
    df = pd.DataFrame({"A": [1, 2, 3, 4], "B": [4, 3, 2, 1]})
    y = np.array([0, 0, 1, 1])

    selector = MIFeatureSelector(top_k=1, random_state=42)
    selector.fit(df, y)

    save_path = tmp_path / "selector.joblib"
    selector.save(save_path)

    loaded = MIFeatureSelector.load(save_path)
    assert loaded.get_selected_features() == selector.get_selected_features()
