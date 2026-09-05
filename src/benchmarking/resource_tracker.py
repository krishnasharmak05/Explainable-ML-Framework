from dataclasses import asdict, dataclass
import os
from pathlib import Path
import time
import tracemalloc
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import psutil


@dataclass
class EfficiencyReport:
    """Dataclass holding computational efficiency benchmarks for a model."""
    model_name: str
    preprocessing_time_sec: float
    feature_selection_time_sec: float
    training_time_sec: float
    inference_time_sec: float
    inference_latency_ms_per_sample: float
    inference_throughput_samples_per_sec: float
    peak_memory_mb: float
    model_size_bytes: int
    model_size_kb: float
    n_raw_features: int
    n_selected_features: int
    n_train_samples: int
    n_test_samples: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ResourceTracker:
    """
    Phase 4: Computational Benchmarking & Resource Tracking.
    Measures timing, memory consumption, latency, throughput, and disk footprint.
    """

    def __init__(self):
        self._process = psutil.Process(os.getpid())
        self._start_time: Optional[float] = None
        self._initial_memory: float = 0.0
        self._peak_memory: float = 0.0

    def start_memory_tracking(self) -> None:
        """Starts monitoring memory usage via tracemalloc."""
        tracemalloc.start()
        self._initial_memory = self._process.memory_info().rss / (1024 * 1024)
        self._peak_memory = self._initial_memory

    def stop_memory_tracking(self) -> float:
        """
        Stops memory monitoring and returns peak memory in MB.
        Uses tracemalloc peak with fallback to process RSS.
        """
        current_rss = self._process.memory_info().rss / (1024 * 1024)
        if tracemalloc.is_tracing():
            _, peak_traced = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_mb = max(current_rss, self._initial_memory + (peak_traced / (1024 * 1024)))
        else:
            peak_mb = current_rss

        self._peak_memory = max(self._peak_memory, peak_mb)
        return float(self._peak_memory)

    @staticmethod
    def measure_inference_performance(
        model: Any,
        X_test: Union[pd.DataFrame, Any],
        n_repeats: int = 3,
    ) -> Dict[str, float]:
        """
        Accurately measures total inference time, per-sample latency, and throughput.
        """
        n_samples = len(X_test)
        if n_samples == 0:
            return {
                "inference_time_sec": 0.0,
                "latency_ms": 0.0,
                "throughput": 0.0,
            }

        # Warm-up pass
        _ = model.predict(X_test[:min(10, n_samples)])

        # Timed passes
        latencies = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            _ = model.predict(X_test)
            latencies.append(time.perf_counter() - t0)

        mean_inference_time = float(sum(latencies) / len(latencies))
        latency_ms_per_sample = (mean_inference_time / n_samples) * 1000.0
        throughput = n_samples / mean_inference_time if mean_inference_time > 0 else 0.0

        return {
            "inference_time_sec": mean_inference_time,
            "latency_ms": latency_ms_per_sample,
            "throughput": throughput,
        }

    @staticmethod
    def get_file_size(file_path: Union[str, Path]) -> int:
        """Returns file size in bytes, or 0 if file does not exist."""
        path = Path(file_path)
        return path.stat().st_size if path.exists() else 0

    @classmethod
    def create_efficiency_report(
        cls,
        model_name: str,
        preprocessing_time_sec: float,
        feature_selection_time_sec: float,
        training_time_sec: float,
        inference_time_sec: float,
        inference_latency_ms: float,
        inference_throughput: float,
        peak_memory_mb: float,
        model_file_path: Optional[Union[str, Path]],
        n_raw_features: int,
        n_selected_features: int,
        n_train_samples: int,
        n_test_samples: int,
    ) -> EfficiencyReport:
        """Constructs an EfficiencyReport object."""
        model_size_bytes = cls.get_file_size(model_file_path) if model_file_path else 0
        return EfficiencyReport(
            model_name=model_name,
            preprocessing_time_sec=preprocessing_time_sec,
            feature_selection_time_sec=feature_selection_time_sec,
            training_time_sec=training_time_sec,
            inference_time_sec=inference_time_sec,
            inference_latency_ms_per_sample=inference_latency_ms,
            inference_throughput_samples_per_sec=inference_throughput,
            peak_memory_mb=peak_memory_mb,
            model_size_bytes=model_size_bytes,
            model_size_kb=model_size_bytes / 1024.0,
            n_raw_features=n_raw_features,
            n_selected_features=n_selected_features,
            n_train_samples=n_train_samples,
            n_test_samples=n_test_samples,
        )

    @staticmethod
    def reports_to_dataframe(reports: List[EfficiencyReport]) -> pd.DataFrame:
        """Converts a list of EfficiencyReport items into a summary DataFrame."""
        return pd.DataFrame([r.to_dict() for r in reports])
