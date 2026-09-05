import numpy as np
import pandas as pd
import pytest
from src.preprocessing.cleaner import DataPreprocessor


def test_missing_and_infinite_value_handling():
    # Construct DataFrame with infinities and NaNs
    df = pd.DataFrame({
        "Flow Duration": [100.0, 200.0, np.inf, 400.0, np.nan],
        "Flow Bytes/s": [-np.inf, 50.0, 60.0, 70.0, 80.0],
        "Packets": [1.0, 2.0, 3.0, 4.0, 5.0],
    })

    preprocessor = DataPreprocessor(imputation_strategy="mean")
    transformed = preprocessor.fit_transform(df)

    # Verify no NaN or Inf remain
    assert not transformed.isna().any().any()
    assert not np.isinf(transformed.values).any()
    # Verify all scaled within [0, 1]
    assert (transformed.values >= 0.0).all()
    assert (transformed.values <= 1.0).all()


def test_log_transformation():
    # Create feature with high skewness
    rng = np.random.RandomState(42)
    skewed_data = rng.exponential(scale=100.0, size=1000)
    normal_data = rng.normal(loc=50.0, scale=5.0, size=1000)

    df = pd.DataFrame({
        "Skewed": skewed_data,
        "Normal": normal_data,
    })

    preprocessor = DataPreprocessor(skewness_threshold=1.0)
    preprocessor.fit(df)

    assert "Skewed" in preprocessor.skewed_columns_
    # Normal data should have skewness close to 0, not exceeding threshold
    assert "Normal" not in preprocessor.skewed_columns_


def test_label_encoding():
    y_raw = pd.Series(["BENIGN", "DoS Hulk", "BENIGN", "PortScan", "DDoS"])

    # Binary
    prep_bin = DataPreprocessor(classification_mode="binary")
    y_bin = prep_bin.encode_labels(y_raw, fit=True)
    assert list(y_bin) == [0, 1, 0, 1, 1]

    # Multiclass
    prep_multi = DataPreprocessor(classification_mode="multiclass")
    y_multi = prep_multi.encode_labels(y_raw, fit=True)
    assert y_multi[0] == 0  # BENIGN is class 0
    assert len(set(y_multi)) == 4


def test_preprocessor_save_load(tmp_path):
    df = pd.DataFrame({"A": [10.0, 20.0, 30.0], "B": [1.0, 2.0, 3.0]})
    preprocessor = DataPreprocessor()
    preprocessor.fit(df)

    save_path = tmp_path / "preprocessor.joblib"
    preprocessor.save(save_path)

    loaded = DataPreprocessor.load(save_path)
    transformed_orig = preprocessor.transform(df)
    transformed_loaded = loaded.transform(df)

    pd.testing.assert_frame_equal(transformed_orig, transformed_loaded)
