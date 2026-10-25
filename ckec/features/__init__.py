"""
ckec.features — Feature extraction for all CKEC detectors.

Modules
-------
Phase features       Carrier-phase temporal differences and normalised z-scores
Power features       Hardware AGC, correlator magnitude, and C/N0 — each named
                     to reflect their distinct physical origin
Geometry features    Predicted inter-antenna phase differences and residuals
                     (synthetic/deferred only — requires antenna baseline)
Cross-modal feature  Rolling Pearson correlation between phase and power innovations
New (V2) features    Multi-scale cross-modal correlation, innovation entropy,
                     and CUSUM statistic for gradual spoofing detection

Design notes
------------
* Every feature function is pure: takes arrays/Series, returns arrays/Series.
* No feature function reads from disk or modifies global state.
* Secondary-modality functions (delta_agc, delta_corr, delta_cn0) are named and
  documented separately. They must NOT be used interchangeably — each targets a
  distinct physical measurement.
* Geometry features require antenna_baseline_m and satellite_unit_vector inputs.
  Do NOT call them when those inputs are unavailable (i.e., on any open real dataset).
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from ckec.preprocessing import (
    temporal_difference,
    wrap_phase,
    estimate_benign_stats,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Phase features
# ---------------------------------------------------------------------------

def delta_phase(
    phase: pd.Series,
    periods: int = 1,
    unit: str = "cycles",
) -> pd.Series:
    """Difference carrier phase, preserving the input unit."""
    if unit not in {"cycles", "radians"}:
        raise ValueError("unit must be 'cycles' or 'radians'")
    return temporal_difference(phase, periods=periods).rename(
        f"delta_phi_{unit}"
    )


def normalised_phase(
    d_phi: pd.Series,
    mu: float,
    sigma: float,
    epsilon: float = 1e-4,
) -> pd.Series:
    """
    Z-score of phase-change relative to benign calibration statistics.

    Z_{phi,t} = (Delta_phi_t - mu_phi) / (sigma_phi + epsilon)
    """
    return ((d_phi - mu) / (sigma + epsilon)).rename("z_phi")


# ---------------------------------------------------------------------------
# Secondary modality features — named by physical origin
# ---------------------------------------------------------------------------

def delta_agc(agc: pd.Series, periods: int = 1) -> pd.Series:
    """
    First difference of HARDWARE AGC observations (UBX-MON-RF agcCnt, 0-8191).
    
    Valid ONLY for Jammertest-2024 dataset which exports true front-end AGC.
    Do NOT pass correlator magnitude or C/N0 into this function.
    """
    return temporal_difference(agc, periods=periods).rename("delta_agc")


def normalised_agc(
    d_agc: pd.Series,
    mu: float,
    sigma: float,
    epsilon: float = 1e-4,
) -> pd.Series:
    """Z-score of hardware AGC change relative to benign calibration statistics."""
    return ((d_agc - mu) / (sigma + epsilon)).rename("z_agc")


def delta_corr(corr_mag: pd.Series, periods: int = 1) -> pd.Series:
    """
    First difference of prompt CORRELATOR MAGNITUDE (sqrt(I_P^2 + Q_P^2)).
    
    Valid ONLY for FGI-SpoofRepo where FGI-GSRx tracking results are available.
    This is post-correlation despread amplitude — NOT hardware AGC.
    """
    return temporal_difference(corr_mag, periods=periods).rename("delta_corr")


def normalised_corr(
    d_corr: pd.Series,
    mu: float,
    sigma: float,
    epsilon: float = 1e-4,
) -> pd.Series:
    """Z-score of correlator magnitude change relative to benign calibration statistics."""
    return ((d_corr - mu) / (sigma + epsilon)).rename("z_corr")


def delta_cn0(cn0: pd.Series, periods: int = 1) -> pd.Series:
    """
    First difference of CARRIER-TO-NOISE DENSITY (C/N0 in dB-Hz).
    
    Valid for Yunnan-2024 (primary) and Jammertest-2024 (secondary comparison).
    C/N0 measures signal power relative to noise bandwidth density.
    It is NOT hardware AGC and NOT correlator magnitude.
    """
    return temporal_difference(cn0, periods=periods).rename("delta_cn0")


def normalised_cn0(
    d_cn0: pd.Series,
    mu: float,
    sigma: float,
    epsilon: float = 1e-4,
) -> pd.Series:
    """Z-score of C/N0 change relative to benign calibration statistics."""
    return ((d_cn0 - mu) / (sigma + epsilon)).rename("z_cn0")


# ---------------------------------------------------------------------------
# Raw ratio (D1 exploratory baseline — documents instability)
# ---------------------------------------------------------------------------

def raw_ratio(
    d_phi: pd.Series,
    d_secondary: pd.Series,
    epsilon: float = 1e-4,
) -> pd.Series:
    """
    J_t = |Delta_phi_t| / (|Delta_M_t| + epsilon)

    where M is the secondary modality series (hardware AGC, correlator magnitude,
    or C/N0 — specified by caller).

    This is an EXPLORATORY BASELINE used to document numerical instability
    when the denominator |Delta_M_t| approaches zero. Do not report this as
    a reliable spoofing detector or as evidence of the proposed mechanism.
    """
    ratio = d_phi.abs() / (d_secondary.abs() + epsilon)
    return ratio.rename("raw_ratio")


# ---------------------------------------------------------------------------
# Geometry-aware features (D5 — synthetic / deferred)
# ---------------------------------------------------------------------------

def predict_inter_antenna_phase(
    satellite_unit_vector: np.ndarray,
    antenna_baseline_m: np.ndarray,
    carrier_wavelength_m: float = 0.1903,
) -> np.ndarray:
    """
    Predict inter-antenna carrier-phase difference from satellite geometry.

    phi_pred = (2 * pi / lambda) * dot(baseline, unit_vector)

    BLOCKED on all available real datasets (FGI-SpoofRepo, Jammertest-2024,
    Yunnan-2024) because all use single-antenna architectures. For use only
    in synthetic simulations with simulated baselines.
    """
    sv = np.asarray(satellite_unit_vector)
    b = np.asarray(antenna_baseline_m)
    projection = sv @ b
    phi_pred = (2 * np.pi / carrier_wavelength_m) * projection
    return phi_pred


def geometry_residual(
    phi_obs: pd.Series,
    phi_pred: np.ndarray,
) -> pd.Series:
    """
    r_{phi,t} = wrap( phi_obs_t - phi_pred_t )

    Wrapping accounts for 2π periodicity. Large residuals may indicate spoofing
    but also arise from calibration errors, multipath, or attitude errors.
    BLOCKED on real datasets — antenna baseline unavailable.
    """
    residual = wrap_phase(phi_obs.to_numpy() - phi_pred)
    return pd.Series(residual, index=phi_obs.index, name="geo_residual")


def normalised_residual(
    residual: pd.Series,
    mu: float,
    sigma: float,
    epsilon: float = 1e-4,
) -> pd.Series:
    """Z-score of geometry residual relative to benign calibration statistics."""
    return ((residual - mu) / (sigma + epsilon)).rename("z_residual")


# ---------------------------------------------------------------------------
# Cross-modal consistency feature (candidate CKEC mechanism)
# ---------------------------------------------------------------------------

def cross_modal_consistency(
    z_phi: pd.Series,
    z_secondary: pd.Series,
    window: int = 10,
) -> pd.Series:
    """
    C_t — candidate cross-modal consistency feature.

    Computes the rolling Pearson correlation between normalised phase innovations
    and normalised secondary-modality innovations over a window of `window` samples.

    Research hypothesis: during a spoofing pull-off, the secondary-modality
    (AGC / correlator / C/N0) response decouples from the carrier-phase innovation
    in a characteristic way, causing C_t to collapse or change sign.

    This is a CANDIDATE feature — NOT a proven detector. It must be compared
    against D2, D3(a/b/c), and D4 on real data before any novelty claim is made.

    Parameters
    ----------
    z_phi       : normalised phase-change z-scores
    z_secondary : normalised secondary-modality z-scores (agc, corr, or cn0)
    window      : rolling window length in samples

    Returns
    -------
    rolling Pearson correlation (NaN for the first window−1 samples)
    """
    if window < 2:
        raise ValueError("window must be >= 2 for correlation computation.")
    combined = pd.DataFrame({"z_phi": z_phi, "z_sec": z_secondary})
    corr = combined["z_phi"].rolling(window=window).corr(combined["z_sec"])
    return corr.rename("cross_modal_corr")


def cross_modal_consistency_multiscale(
    z_phi: pd.Series,
    z_secondary: pd.Series,
    windows: tuple[int, int, int] = (5, 10, 20),
) -> pd.Series:
    """
    Multi-scale cross-modal correlation diagnostic.

    Returns the signed correlation at the scale with the smallest absolute
    correlation. This makes a near-zero correlation at one scale visible
    even if another scale has a strong negative correlation.

    This is a candidate feature, not a validated spoofing detector.
    """
    for w in windows:
        if w < 2:
            raise ValueError(f"All windows must be >= 2; got {w}.")

    combined = pd.DataFrame({
        "z_phi": z_phi,
        "z_sec": z_secondary,
    })

    corrs = pd.concat(
        [
            combined["z_phi"].rolling(window=w).corr(combined["z_sec"])
            for w in windows
        ],
        axis=1,
    )
    corrs.columns = [f"corr_{w}" for w in windows]

    # Ignore missing scales when choosing the minimum absolute correlation.
    abs_corrs = corrs.abs()
    selected_scale = abs_corrs.fillna(np.inf).to_numpy().argmin(axis=1)

    values = corrs.to_numpy()
    rows = np.arange(len(corrs))
    result = values[rows, selected_scale]

    # Return NaN when every scale is unavailable.
    all_missing = corrs.isna().all(axis=1).to_numpy()
    result[all_missing] = np.nan

    return pd.Series(
        result,
        index=z_phi.index,
        name="cross_modal_corr_ms",
    )


def innovation_entropy(
    z_phi: pd.Series,
    window: int = 20,
    n_bins: int = 8,
) -> pd.Series:
    """
    Rolling normalised entropy of phase-innovation z-scores.

    Under authentic tracking, phase innovations are Gaussian and entropy
    is near-maximum for the estimated distribution. During spoofing, the
    innovation distribution becomes skewed or bimodal (transition artifact),
    which is detected as a transient entropy change.

    Returns a rolling entropy estimate (bits), normalised by log2(n_bins)
    so the range is [0, 1]. Values near 0 indicate heavy distribution
    concentration — a signature of a systematic phase ramp.

    Parameters
    ----------
    z_phi  : normalised phase innovation z-scores
    window : rolling window length
    n_bins : number of histogram bins
    """
    vals = z_phi.to_numpy(dtype=float)
    n = len(vals)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        chunk = vals[i - window + 1 : i + 1]
        chunk = chunk[~np.isnan(chunk)]
        if len(chunk) < 4:
            continue
        counts, _ = np.histogram(chunk, bins=n_bins)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs))
        out[i] = entropy / np.log2(n_bins)   # normalise to [0, 1]
    return pd.Series(out, index=z_phi.index, name="innovation_entropy")


def cusum_statistic(
    z_scores: pd.Series,
    k: float = 0.5,
    window: int = 150,
) -> pd.Series:
    """Two-sided cumulative-evidence score over a finite trailing window."""
    if window < 1:
        raise ValueError("window must be >= 1.")
    if k < 0:
        raise ValueError("k must be >= 0.")

    vals = z_scores.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan, dtype=float)

    for i in range(len(vals)):
        start = max(0, i - window + 1)
        chunk = vals[start:i + 1]

        # Use the finite suffix ending at this sample.
        if not np.isfinite(chunk[-1]):
            continue

        bad = np.flatnonzero(~np.isfinite(chunk))
        if len(bad):
            chunk = chunk[bad[-1] + 1:]

        if len(chunk) == 0:
            continue

        scores = []
        for increments in (chunk - k, -chunk - k):
            prefix = np.concatenate(([0.0], np.cumsum(increments)))
            # Largest contiguous sum ending at the current sample.
            scores.append(float(prefix[-1] - np.min(prefix[:-1])))

        out[i] = max(0.0, *scores)

    return pd.Series(out, index=z_scores.index, name="cusum_stat")


# ---------------------------------------------------------------------------
# Feature matrix assembly
# ---------------------------------------------------------------------------

def build_feature_matrix(
    z_phi: pd.Series,
    z_agc: pd.Series | None = None,
    z_corr: pd.Series | None = None,
    z_cn0: pd.Series | None = None,
    z_residual: pd.Series | None = None,
    cross_modal: pd.Series | None = None,
    cross_modal_ms: pd.Series | None = None,
    entropy: pd.Series | None = None,
    cusum: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Assemble available normalised features into a single DataFrame.

    Only z_phi is required. Provide exactly ONE secondary modality per experiment:
      - z_agc   for Jammertest-2024 (hardware AGC)
      - z_corr  for FGI-SpoofRepo   (correlator magnitude)
      - z_cn0   for Yunnan-2024 / Jammertest (C/N0 density)

    Providing multiple secondary modalities in the same DataFrame is permitted
    for cross-dataset ablation studies, but each must be analysed separately.

    V2 additions (optional):
      - cross_modal_ms : multi-scale cross-modal correlation
      - entropy        : rolling normalised phase innovation entropy
      - cusum          : CUSUM statistic for gradual mean-shift detection
    """
    features: dict[str, pd.Series] = {"z_phi": z_phi}
    if z_agc is not None:
        features["z_agc"] = z_agc
    if z_corr is not None:
        features["z_corr"] = z_corr
    if z_cn0 is not None:
        features["z_cn0"] = z_cn0
    if z_residual is not None:
        features["z_residual"] = z_residual
    if cross_modal is not None:
        features["cross_modal_corr"] = cross_modal
    if cross_modal_ms is not None:
        features["cross_modal_corr_ms"] = cross_modal_ms
    if entropy is not None:
        features["innovation_entropy"] = entropy
    if cusum is not None:
        features["cusum_stat"] = cusum
    return pd.DataFrame(features)
