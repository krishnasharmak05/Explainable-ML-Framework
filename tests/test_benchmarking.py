import numpy as np
import pandas as pd
import pytest
from src.benchmarking.resource_tracker import EfficiencyReport, ResourceTracker
from src.modeling.ensemble import RandomForestWrapper


def test_resource_tracker_inference():
    rng = np.random.RandomState(42)
    X_train = pd.DataFrame(rng.normal(size=(50, 4)), columns=[f"f{i}" for i in range(4)])
    y_train = rng.choice([0, 1], size=50)
    X_test = pd.DataFrame(rng.normal(size=(20, 4)), columns=[f"f{i}" for i in range(4)])

    model = RandomForestWrapper(params={"n_estimators": 5, "random_state": 42})
    model.fit(X_train, y_train)

    perf = ResourceTracker.measure_inference_performance(model, X_test, n_repeats=2)

    assert "inference_time_sec" in perf
    assert "latency_ms" in perf
    assert "throughput" in perf
    assert perf["latency_ms"] >= 0.0
    assert perf["throughput"] >= 0.0


def test_resource_tracker_memory():
    tracker = ResourceTracker()
    tracker.start_memory_tracking()
    # Allocate temporary array
    arr = np.ones((500, 500), dtype=float)
    peak_mb = tracker.stop_memory_tracking()
    del arr

    assert peak_mb > 0.0


def test_efficiency_report_generation(tmp_path):
    dummy_model_file = tmp_path / "model.joblib"
    dummy_model_file.write_bytes(b"dummy model content 12345")

    report = ResourceTracker.create_efficiency_report(
        model_name="rf",
        preprocessing_time_sec=0.12,
        feature_selection_time_sec=0.45,
        training_time_sec=1.23,
        inference_time_sec=0.05,
        inference_latency_ms=0.5,
        inference_throughput=2000.0,
        peak_memory_mb=128.5,
        model_file_path=dummy_model_file,
        n_raw_features=25,
        n_selected_features=10,
        n_train_samples=800,
        n_test_samples=200,
    )

    assert report.model_name == "rf"
    assert report.model_size_bytes == 25
    df = ResourceTracker.reports_to_dataframe([report])
    assert len(df) == 1
    assert df.iloc[0]["n_selected_features"] == 10
