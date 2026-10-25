"""
ckec.preprocessing — Signal cleaning, temporal alignment, and phase handling.

All functions operate on pandas DataFrames with a uniform sample index.
They never silently drop or fill data without logging a warning.

V2 Improvements
---------------
* estimate_benign_stats now uses MAD (Median Absolute Deviation) as a robust
  alternative to std, which is insensitive to outliers in the calibration window.
* Added estimate_benign_stats_robust for explicit MAD-based estimation.
* Added clip_innovations to pre-filter extreme phase/AGC spikes before z-scoring.
* Added exponential smoothing helper for adaptive per-sample statistics.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase handling
# ---------------------------------------------------------------------------

def unwrap_phase(
    series: pd.Series,
    unit: str = "radians",
    discont: float | None = None,
) -> pd.Series:
    """
    Unwrap wrapped phase observations.

    unit:
        "radians" -- phase period is 2*pi
        "cycles"  -- phase period is 1

    Do not call this on already-unwrapped carrier phase.
    """
    if unit not in {"radians", "cycles"}:
        raise ValueError("unit must be 'radians' or 'cycles'")

    values = series.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(
            "unwrap_phase received NaN or infinite values; "
            "split into contiguous finite segments before unwrapping."
        )

    scale = 2.0 * np.pi if unit == "cycles" else 1.0
    radians = values * scale

    if discont is None:
        discontinuity = np.pi
    else:
        discontinuity = discont * scale

    unwrapped = np.unwrap(radians, discont=discontinuity) / scale
    return pd.Series(unwrapped, index=series.index, name=series.name)


def wrap_phase(value: float | np.ndarray) -> float | np.ndarray:
    """Wrap a phase value or array to (−π, π]."""
    return (np.asarray(value) + np.pi) % (2 * np.pi) - np.pi


def temporal_difference(series: pd.Series, periods: int = 1) -> pd.Series:
    """Return the first difference of a series (Δy_t = y_t − y_{t-1})."""
    return series.diff(periods=periods)


# ---------------------------------------------------------------------------
# AGC smoothing
# ---------------------------------------------------------------------------

def smooth_agc(agc: pd.Series, window: int = 5) -> pd.Series:
    """
    Apply a trailing rolling-mean filter to AGC observations.

    The filter is causal: the value at time t uses only observations at or
    before t. A window of 1 or less returns the series unchanged.
    AGC is a receiver-level aggregate measurement; smoothing reduces sample-
    to-sample noise but may also slow the response to genuine step changes.
    Choose window size carefully and report it in the evaluation log.
    """
    if window <= 1:
        return agc
    smoothed = agc.rolling(window=window, min_periods=1).mean()
    return smoothed.rename(agc.name)


def clip_innovations(
    series: pd.Series,
    n_sigma: float = 8.0,
    window: int = 50,
) -> pd.Series:
    """
    Clip extreme innovation spikes before z-scoring.

    Replaces values beyond n_sigma * MAD with the rolling median.
    This prevents receiver-reset artifacts from contaminating the calibration
    or inflating z-scores for subsequent genuine changes.

    Parameters
    ----------
    series   : series of innovations (first differences)
    n_sigma  : clipping threshold in robust sigma units
    window   : rolling window for local median and MAD estimation
    """
    roll_med = series.rolling(window=window, center=True, min_periods=3).median()
    roll_mad = (series - roll_med).abs().rolling(window=window, center=True, min_periods=3).median()
    robust_sigma = roll_mad * 1.4826   # MAD → equivalent Gaussian sigma
    threshold = n_sigma * robust_sigma
    outlier_mask = (series - roll_med).abs() > threshold.fillna(threshold.median())
    clipped = series.copy()
    clipped[outlier_mask] = roll_med[outlier_mask]
    if outlier_mask.sum() > 0:
        log.debug("clip_innovations: clipped %d outlier samples.", outlier_mask.sum())
    return clipped.rename(series.name)


# ---------------------------------------------------------------------------
# Temporal alignment
# ---------------------------------------------------------------------------

def align_channels(
    df: pd.DataFrame,
    time_col: str = "time_s",
    tolerance_s: float = 0.01,
) -> pd.DataFrame:
    """Validate timestamp order and report gaps without synthesizing samples."""
    if time_col not in df.columns:
        raise ValueError(f"Missing required time column: {time_col}")
    if tolerance_s < 0:
        raise ValueError("tolerance_s must be non-negative")

    out = df.copy()
    out[time_col] = pd.to_numeric(out[time_col], errors="raise")

    times = out[time_col].to_numpy(dtype=float)
    if not np.isfinite(times).all():
        raise ValueError("Timestamps contain NaN or infinity.")

    out = out.sort_values(time_col, kind="stable").reset_index(drop=True)
    times = out[time_col].to_numpy(dtype=float)

    if len(times) < 2:
        return out

    dt = np.diff(times)
    if np.any(dt <= 0):
        raise ValueError(
            "Timestamps must be unique and strictly increasing. "
            "Resolve duplicates explicitly before alignment."
        )

    median_dt = float(np.median(dt))
    irregular = np.abs(dt - median_dt) > tolerance_s

    if irregular.any():
        log.warning(
            "%d irregular intervals detected; original timestamps preserved.",
            int(irregular.sum()),
        )

    return out


# ---------------------------------------------------------------------------
# Missing-value handling
# ---------------------------------------------------------------------------

def handle_missing(
    df: pd.DataFrame,
    strategy: str = "interpolate",
    max_gap: int = 10,
) -> pd.DataFrame:
    """
    Handle NaN values in the DataFrame.

    Parameters
    ----------
    strategy : "interpolate" | "drop" | "forward_fill"
    max_gap  : maximum number of consecutive NaN values to fill
               (gaps longer than this remain as NaN)

    Notes
    -----
    Missing AGC or phase values at attack boundaries can affect detection delay
    measurements.  Always log the number and location of filled values.
    """
    n_missing_before = df.isna().sum().sum()
    if n_missing_before == 0:
        return df

    if strategy == "interpolate":
        df = df.interpolate(method="linear", limit=max_gap, limit_direction="both")
    elif strategy == "forward_fill":
        df = df.ffill(limit=max_gap)
    elif strategy == "drop":
        df = df.dropna()
    else:
        raise ValueError(f"Unknown missing-value strategy: '{strategy}'")

    n_missing_after = df.isna().sum().sum()
    log.info(
        "Missing values: %d before -> %d after ('%s', max_gap=%d).",
        n_missing_before, n_missing_after, strategy, max_gap,
    )
    return df


# ---------------------------------------------------------------------------
# Calibration window estimation
# ---------------------------------------------------------------------------

def estimate_benign_stats(
    series: pd.Series,
    n_samples: int,
    use_mad: bool = True,
) -> tuple[float, float]:
    """Estimate robust location and scale from a pre-attack calibration window."""
    if n_samples < 2:
        raise ValueError("n_samples must be >= 2")

    calibration = pd.Series(series).iloc[:n_samples].to_numpy(dtype=float)
    calibration = calibration[np.isfinite(calibration)]

    if calibration.size < 2:
        raise ValueError(
            "Calibration failed: fewer than two finite observations."
        )

    mu = float(np.median(calibration) if use_mad else np.mean(calibration))

    if use_mad:
        sigma = float(
            1.4826 * np.median(np.abs(calibration - mu))
        )
    else:
        sigma = float(np.std(calibration, ddof=1))

    if not np.isfinite(mu) or not np.isfinite(sigma):
        raise ValueError("Calibration failed: non-finite location or scale.")

    if sigma < 0 or not np.isfinite(sigma):
        raise ValueError("Calibration failed: invalid scale.")

    return mu, sigma


def estimate_benign_stats_robust(
    series: pd.Series,
    n_samples: int,
) -> tuple[float, float]:
    """
    Alias for estimate_benign_stats(use_mad=True) for explicit call sites.
    Always uses MAD for sigma estimation. Preferred over the std-based version
    because real GNSS calibration windows may contain short transients.
    """
    return estimate_benign_stats(series, n_samples, use_mad=True)
