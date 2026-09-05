from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml


@dataclass
class DataConfig:
    dataset_path: str = "./dataset/"
    target_column: str = "Label"
    classification_mode: str = "binary"  # "binary" or "multiclass"
    benign_label: str = "BENIGN"
    test_size: float = 0.2
    random_state: int = 42


@dataclass
class PreprocessingConfig:
    imputation_strategy: str = "mean"  # "mean" or "median" (Equation 1)
    skewness_threshold: float = 1.0     # Skewness threshold for log(X+1) (Equation 2)
    scaling_range: Tuple[float, float] = (0.0, 1.0)  # Min-Max range (Equation 3)
    drop_columns: List[str] = field(default_factory=list)


@dataclass
class FeatureSelectionConfig:
    top_k: int = 15                     # Top k features selected via MI (Algorithm 1)
    n_neighbors: int = 3
    sample_size: Optional[int] = 50000  # Subsample size for MI computation if large


@dataclass
class ModelingConfig:
    cv_folds: int = 5                   # K-fold cross validation (Algorithm 2)
    active_models: List[str] = field(
        default_factory=lambda: ["random_forest", "xgboost", "lightgbm"]
    )
    random_forest: Dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 100,
            "max_depth": 15,
            "random_state": 42,
            "n_jobs": -1,
        }
    )
    xgboost: Dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 42,
            "eval_metric": "logloss",
            "n_jobs": -1,
        }
    )
    lightgbm: Dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
    )


@dataclass
class BenchmarkingConfig:
    enabled: bool = True
    n_repeats: int = 3


@dataclass
class CalibrationConfig:
    enabled: bool = True
    methods: List[str] = field(default_factory=lambda: ["sigmoid", "isotonic"])
    n_bins: int = 10


@dataclass
class StabilityConfig:
    enabled: bool = True
    top_k: int = 10


@dataclass
class ExplainabilityConfig:
    enabled: bool = True
    max_eval_samples: int = 300
    stability: StabilityConfig = field(default_factory=StabilityConfig)


@dataclass
class RobustnessConfig:
    enabled: bool = True
    noise_levels: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.10, 0.20])
    missingness_rates: List[float] = field(default_factory=lambda: [0.05, 0.10, 0.20])


@dataclass
class OutputConfig:
    output_dir: str = "./outputs"
    models_dir: str = "./outputs/models"
    reports_dir: str = "./outputs/reports"


@dataclass
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    feature_selection: FeatureSelectionConfig = field(default_factory=FeatureSelectionConfig)
    modeling: ModelingConfig = field(default_factory=ModelingConfig)
    benchmarking: BenchmarkingConfig = field(default_factory=BenchmarkingConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    explainability: ExplainabilityConfig = field(default_factory=ExplainabilityConfig)
    robustness: RobustnessConfig = field(default_factory=RobustnessConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        path = Path(path)
        if not path.exists():
            return cls()

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        exp_raw = raw.get("explainability", {})
        stab_raw = exp_raw.get("stability", {}) if isinstance(exp_raw, dict) else {}

        return cls(
            data=DataConfig(**raw.get("data", {})),
            preprocessing=PreprocessingConfig(
                imputation_strategy=raw.get("preprocessing", {}).get("imputation_strategy", "mean"),
                skewness_threshold=raw.get("preprocessing", {}).get("skewness_threshold", 1.0),
                scaling_range=tuple(raw.get("preprocessing", {}).get("scaling_range", [0.0, 1.0])),
                drop_columns=raw.get("preprocessing", {}).get("drop_columns", []),
            ),
            feature_selection=FeatureSelectionConfig(**raw.get("feature_selection", {})),
            modeling=ModelingConfig(
                cv_folds=raw.get("modeling", {}).get("cv_folds", 5),
                active_models=raw.get("modeling", {}).get("active_models", ["random_forest", "xgboost", "lightgbm"]),
                random_forest=raw.get("modeling", {}).get("random_forest", {}),
                xgboost=raw.get("modeling", {}).get("xgboost", {}),
                lightgbm=raw.get("modeling", {}).get("lightgbm", {}),
            ),
            benchmarking=BenchmarkingConfig(**raw.get("benchmarking", {})),
            calibration=CalibrationConfig(**raw.get("calibration", {})),
            explainability=ExplainabilityConfig(
                enabled=exp_raw.get("enabled", True),
                max_eval_samples=exp_raw.get("max_eval_samples", 300),
                stability=StabilityConfig(**stab_raw),
            ),
            robustness=RobustnessConfig(**raw.get("robustness", {})),
            output=OutputConfig(**raw.get("output", {})),
        )
