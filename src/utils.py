import logging
import os
from pathlib import Path
import random
from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def setup_logger(name: str = "ids_pipeline", log_dir: Optional[str] = None) -> logging.Logger:
    """Configures a standardized console and file logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Optional File Handler
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(Path(log_dir) / f"{name}.log", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def set_seed(seed: int = 42) -> None:
    """Sets random seeds across standard libraries for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_dataset(path: Union[str, Path], logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Loads CIC-IDS2017 CSV file(s) from a file path or directory.
    Strips leading and trailing whitespace from column names.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset path not found: {path}")

    csv_files: List[Path] = []
    if path.is_file() and path.suffix.lower() == ".csv":
        csv_files.append(path)
    elif path.is_dir():
        csv_files = sorted(list(path.glob("*.csv")))
        if not csv_files:
            raise FileNotFoundError(f"No .csv files found in directory: {path}")
    else:
        raise ValueError(f"Path must be a CSV file or directory containing CSV files: {path}")

    dfs = []
    for f in csv_files:
        if logger:
            logger.info(f"Loading CSV: {f.name} ({f.stat().st_size / (1024 * 1024):.2f} MB)...")
        # CIC-IDS2017 CSVs may contain latin-1 or utf-8 characters
        try:
            df = pd.read_csv(f, encoding="utf-8", low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(f, encoding="latin-1", low_memory=False)
        
        # Standardize column headers: strip whitespace
        df.columns = df.columns.str.strip()
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    if logger:
        logger.info(f"Loaded dataset total shape: {combined.shape[0]:,} rows x {combined.shape[1]:,} columns")

    return combined
