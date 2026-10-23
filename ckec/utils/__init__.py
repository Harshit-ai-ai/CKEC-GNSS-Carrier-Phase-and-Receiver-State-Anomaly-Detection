"""
ckec.utils — Shared helpers: configuration loading, logging setup, I/O utilities.
"""

from __future__ import annotations

import os
import sys
import yaml
import random
import logging
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO", log_file: str | None = None) -> logging.Logger:
    """Configure the root logger.  Returns the root logger."""
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format=fmt, handlers=handlers, force=True)
    return logging.getLogger("ckec")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load the YAML experiment configuration file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path.resolve()}")
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


def get_path(cfg: dict[str, Any], key: str) -> Path:
    """Return a Path object for a key inside cfg['paths'], creating it if needed."""
    p = Path(cfg["paths"][key])
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set random seeds for NumPy and Python random for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_csv(df: "pd.DataFrame", path: str | Path, **kwargs) -> None:  # type: ignore[name-defined]
    """Save a DataFrame to CSV, creating parent directories as needed."""
    import pandas as pd  # lazy import
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, **kwargs)


def load_csv(path: str | Path, **kwargs) -> "pd.DataFrame":  # type: ignore[name-defined]
    """Load a CSV file into a DataFrame."""
    import pandas as pd
    return pd.read_csv(path, **kwargs)
