import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

from src.config import PipelineConfig
from src.sample_data import generate_cicids2017_sample
from src.utils import load_dataset, setup_logger


def cmd_validate(args):
    """Inspects and validates an arbitrary tabular dataset without modifying it."""
    cfg = PipelineConfig.from_yaml(args.config)
    data_path = Path(args.data_path or cfg.data.dataset_path)

    logger = setup_logger("validate")
    logger.info(f"Validating dataset at: {data_path}")

    if not data_path.exists() or (data_path.is_dir() and not list(data_path.glob("*.csv"))):
        logger.error(f"Dataset path '{data_path}' does not exist or has no CSV files.")
        sys.exit(1)

    df = load_dataset(data_path, logger=logger)
    target_col = cfg.data.target_column

    matched_cols = [c for c in df.columns if c.lower() == target_col.lower()]
    if not matched_cols:
        logger.error(f"Target column '{target_col}' not found. Available columns: {list(df.columns)}")
        sys.exit(1)
    target_col = matched_cols[0]

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns

    inf_count = int(np.isinf(df[numeric_cols].values).sum()) if len(numeric_cols) > 0 else 0
    nan_count = int(df.isna().sum().sum())
    dup_count = int(df.duplicated().sum())

    print("\n" + "=" * 60)
    print("DATASET VALIDATION REPORT")
    print("=" * 60)
    print(f"Total Rows:            {len(df):,}")
    print(f"Total Columns:         {len(df.columns):,}")
    print(f"Numerical Features:    {len(numeric_cols):,}")
    print(f"Categorical Features:  {len(categorical_cols):,}")
    print(f"Infinite Values:       {inf_count:,}")
    print(f"Missing Values (NaN):  {nan_count:,}")
    print(f"Duplicate Rows:        {dup_count:,}")
    print(f"Target Column:         '{target_col}'")
    print(f"Class Distribution:\n{df[target_col].value_counts(dropna=False)}")
    print("=" * 60 + "\n")


def cmd_run(args):
    """Executes the complete pipeline."""
    from run_pipeline import run as run_main
    run_main()


def main():
    parser = argparse.ArgumentParser(
        prog="researchml",
        description="Dataset-Agnostic Explainable Machine Learning Research Framework",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # validate
    p_val = subparsers.add_parser("validate", help="Validate and inspect dataset hygiene")
    p_val.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    p_val.add_argument("--data-path", type=str, default=None, help="Path to dataset")

    # run
    p_run = subparsers.add_parser("run", help="Run full pipeline (Phases 1-8)")
    p_run.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    p_run.add_argument("--data-path", type=str, default=None, help="Override dataset path")
    p_run.add_argument("--k-features", type=int, default=None, help="Override top-k features")
    p_run.add_argument("--mode", type=str, choices=["binary", "multiclass"], default=None)
    p_run.add_argument("--generate-sample", action="store_true", help="Generate sample dataset if empty")

    args, unknown = parser.parse_known_args()

    if args.command == "validate":
        cmd_validate(args)
    elif args.command == "run" or args.command is None:
        cmd_run(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
