"""
ckec.detectors — All detector implementations for the CKEC comparison study.

Detectors
---------
D1  RawRatioDetector           — |Δφ| / (|ΔM| + ε)  exploratory baseline; documents instability.
D2  PhaseOnlyDetector          — threshold on |z_phi|
D3a AGCOnlyDetector            — threshold on |z_agc| (TRUE hardware AGC — Jammertest-2024 only)
D3b CorrMagDetector            — threshold on |z_corr| (sqrt(I_P^2+Q_P^2) — FGI-SpoofRepo)
D3c CN0Detector                — threshold on |z_cn0|  (C/N0 density — Yunnan-2024 / Jammertest)
D4  JointDetector              — weighted sum of z_phi^2 + z_M^2  (M = any available secondary)
D5  GeometryDetector           — threshold on |z_residual|  (requires geometry inputs — synthetic only)
D6  FullCKECDetector           — all supported features (weights from validation data)

Design rules
------------
* All detectors share a common interface: fit(X_val, y_val) → set threshold,
  predict(X) → binary label,  score(X) → continuous detection score.
* Thresholds are set on validation data, NEVER on test data.
* D3a/D3b/D3c evaluate DISTINCT physical measurements — they must NOT be used
  interchangeably or reported as equivalent.
* A detector that cannot be fitted (wrong input columns) raises RuntimeError
  with a clear message rather than silently producing wrong scores.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseDetector(ABC):
    """Abstract base for all CKEC detectors."""

    name: str = "base"

    def __init__(self, epsilon: float = 1e-4) -> None:
        self.epsilon = epsilon
        self._threshold: float | None = None

    @abstractmethod
    def score(self, X: pd.DataFrame) -> pd.Series:
        """Return a continuous anomaly score for each sample in X."""

    def fit(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series | np.ndarray,
        target_far: float = 0.01,
    ) -> "BaseDetector":
        """
        Calibrate detection threshold on validation data to achieve `target_far`
        false-alarm rate on benign samples.

        Parameters
        ----------
        X_val      : validation feature matrix
        y_val      : binary labels (0 = benign, 1 = spoofing)
        target_far : desired false-alarm rate (proportion of benign samples flagged)

        Returns self for method chaining.
        """
        scores = self.score(X_val)
        score_values = np.asarray(scores, dtype=float).reshape(-1)
        y_arr = np.asarray(y_val).reshape(-1)
        if not np.isin(y_arr, [0, 1]).all():
            raise ValueError("Calibration labels must be binary (0 or 1).")

        if len(score_values) != len(y_arr):
            raise ValueError(
                "Calibration scores and labels must have equal lengths."
            )

        if not np.isfinite(target_far) or not 0.0 < target_far < 1.0:
            raise ValueError("target_far must be strictly between 0 and 1.")

        benign_mask = y_arr == 0
        benign_values = score_values[benign_mask]
        finite_benign = benign_values[np.isfinite(benign_values)]

        if len(finite_benign) != len(benign_values):
            raise RuntimeError(
                "Calibration failed: benign scores contain non-finite values."
            )

        if len(finite_benign) == 0:
            raise RuntimeError(
                "No benign samples in validation set — cannot calibrate threshold."
            )

        self._threshold = float(
            np.quantile(finite_benign, 1.0 - target_far)
        )

        if not np.isfinite(self._threshold):
            self._threshold = None
            raise RuntimeError("Calibration produced a non-finite threshold.")

        log.info(
            "%s: threshold=%.6g, finite benign calibration n=%d, target FAR=%.3f",
            self.name,
            self._threshold,
            len(finite_benign),
            target_far,
        )
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Return binary predictions (1 = spoofing detected, 0 = benign)."""
        if self._threshold is None:
            raise RuntimeError(
                f"Detector '{self.name}' has not been fitted.  "
                "Call .fit(X_val, y_val) first to calibrate the threshold."
            )
        scores = self.score(X)
        return (scores > self._threshold).astype(int).rename("predicted_label")

    def _require_columns(self, X: pd.DataFrame, *cols: str) -> None:
        """Raise RuntimeError if any required column is missing from X."""
        missing = [c for c in cols if c not in X.columns]
        if missing:
            raise RuntimeError(
                f"Detector '{self.name}' requires columns {missing} "
                f"but they are missing from the feature matrix.  "
                f"Available columns: {list(X.columns)}"
            )


# ---------------------------------------------------------------------------
# D1 — Raw ratio (exploratory baseline — documents instability)
# ---------------------------------------------------------------------------

class RawRatioDetector(BaseDetector):
    """
    D1: J_t = |Δφ_t| / (|ΔM_t| + ε)

    where M is the secondary physical measurement column (delta_agc, delta_corr,
    or delta_cn0 depending on which dataset is used).

    WARNING: this statistic is numerically unstable when |ΔM_t| ≈ 0.
    Use ONLY as an exploratory baseline to document this instability — not as a
    production detector or as evidence of the proposed coupling mechanism.
    """
    name = "D1_raw_ratio"

    def __init__(self, secondary_col: str = "delta_agc", **kwargs) -> None:
        super().__init__(**kwargs)
        self.secondary_col = secondary_col

    def score(self, X: pd.DataFrame) -> pd.Series:
        self._require_columns(X, "delta_phi", self.secondary_col)
        s = X["delta_phi"].abs() / (X[self.secondary_col].abs() + self.epsilon)
        return s.rename(self.name)


# ---------------------------------------------------------------------------
# D2 — Phase-only baseline
# ---------------------------------------------------------------------------

class PhaseOnlyDetector(BaseDetector):
    """
    D2: score = |z_phi_t|

    Establishes a defensible carrier-phase innovation baseline. Documents how
    much detection information exists in phase alone, before adding any second
    modality.
    """
    name = "D2_phase_only"

    def score(self, X: pd.DataFrame) -> pd.Series:
        self._require_columns(X, "z_phi")
        return X["z_phi"].abs().rename(self.name)


# ---------------------------------------------------------------------------
# D3a — Hardware AGC-only (Jammertest-2024: UBX-MON-RF.agcCnt)
# ---------------------------------------------------------------------------

class AGCOnlyDetector(BaseDetector):
    """
    D3a: score = |z_agc_t|

    Evaluates TRUE front-end hardware AGC from UBX-MON-RF (agcCnt, 0–8191).
    Valid ONLY for Jammertest-2024 dataset which exports actual hardware AGC.

    Do NOT use this detector with correlator magnitude or C/N0 data — those
    are distinct physical measurements evaluated by D3b and D3c respectively.
    """
    name = "D3a_agc_only"

    def score(self, X: pd.DataFrame) -> pd.Series:
        self._require_columns(X, "z_agc")
        return X["z_agc"].abs().rename(self.name)


# ---------------------------------------------------------------------------
# D3b — Prompt correlator magnitude-only (FGI-SpoofRepo: sqrt(I_P^2 + Q_P^2))
# ---------------------------------------------------------------------------

class CorrMagDetector(BaseDetector):
    """
    D3b: score = |z_corr_t|

    Evaluates prompt correlator magnitude sqrt(I_P^2 + Q_P^2) from FGI-GSRx
    tracking results. This is post-correlation despread power — it is NOT
    front-end hardware AGC.

    Valid for FGI-SpoofRepo where FGI-GSRx tracking results are available.
    Do NOT conflate with D3a (hardware AGC) or D3c (C/N0 density).
    """
    name = "D3b_corr_mag"

    def score(self, X: pd.DataFrame) -> pd.Series:
        self._require_columns(X, "z_corr")
        return X["z_corr"].abs().rename(self.name)


# ---------------------------------------------------------------------------
# D3c — Carrier-to-Noise density-only (Yunnan-2024 / Jammertest: C/N0 in dB-Hz)
# ---------------------------------------------------------------------------

class CN0Detector(BaseDetector):
    """
    D3c: score = |z_cn0_t|

    Evaluates the carrier-to-noise density ratio (C/N0, measured in dB-Hz)
    from RINEX S1C/S5I observables (Jammertest) or UBX-NAV-SAT.cno (Yunnan).

    C/N0 measures carrier power relative to noise density. It is NOT hardware
    AGC (D3a) and NOT correlator prompt magnitude (D3b).

    Valid for Yunnan-2024 (primary) and Jammertest-2024 (secondary comparison).
    """
    name = "D3c_cn0_only"

    def score(self, X: pd.DataFrame) -> pd.Series:
        self._require_columns(X, "z_cn0")
        return X["z_cn0"].abs().rename(self.name)


# ---------------------------------------------------------------------------
# D4 — Joint statistical detector (available secondary modality)
# ---------------------------------------------------------------------------

class JointDetector(BaseDetector):
    """
    D4: S_t = w_phi * z_phi_t^2 + w_M * z_M_t^2

    Tests whether combining phase innovations with a genuinely available secondary
    modality M provides a reproducible, statistically significant improvement over
    both D2 and D3(a/b/c) at a fixed false-alarm rate.

    The secondary_col parameter specifies which column to use as z_M:
      - 'z_agc'  → comparing against D3a (hardware AGC; Jammertest-2024)
      - 'z_corr' → comparing against D3b (correlator magnitude; FGI-SpoofRepo)
      - 'z_cn0'  → comparing against D3c (C/N0 density; Yunnan-2024)

    V2: Default weights changed from (1.0, 1.0) to (1.2, 0.8) based on empirical
    tuning across synthetic benchmarks — phase innovations carry slightly more
    spoofing information than secondary modality in abrupt transitions.
    The research question is whether D4 > max(D2, D3_best) at FAR=1%.
    """
    name = "D4_joint"

    def __init__(
        self,
        secondary_col: str = "z_agc",
        w_phi: float = 1.2,
        w_secondary: float = 0.8,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.secondary_col = secondary_col
        self.w_phi = w_phi
        self.w_secondary = w_secondary

    def score(self, X: pd.DataFrame) -> pd.Series:
        self._require_columns(X, "z_phi", self.secondary_col)
        s = self.w_phi * X["z_phi"] ** 2 + self.w_secondary * X[self.secondary_col] ** 2
        return s.rename(self.name)


# ---------------------------------------------------------------------------
# D5 — Geometry-aware detector (synthetic simulations only)
# ---------------------------------------------------------------------------

class GeometryDetector(BaseDetector):
    """
    D5: score = |z_residual_t|

    Requires pre-computed geometry residuals between observed and predicted
    inter-antenna phase differences. BLOCKED on all available open datasets
    (FGI-SpoofRepo, Jammertest-2024, Yunnan-2024) due to missing antenna baselines.

    Use ONLY in synthetic simulations — never report synthetic D5 results as
    real-data spoofing detection evidence.
    """
    name = "D5_geometry"

    def score(self, X: pd.DataFrame) -> pd.Series:
        self._require_columns(X, "z_residual")
        return X["z_residual"].abs().rename(self.name)


# ---------------------------------------------------------------------------
# D6 — Full CKEC detector
# ---------------------------------------------------------------------------

class FullCKECDetector(BaseDetector):
    """
    D6: S_t = w_phi * z_phi^2 + w_M * z_M^2 + w_r * z_residual^2 + w_c * (1 - |C_t|)
             + w_e * (1 - entropy_t)

    Full CKEC detector combining all available features. V2 additions:
    - Multi-scale cross-modal correlation term (|C_ms_t| — minimum across 3 windows)
    - Innovation entropy term: low entropy signals systematic phase drift (spoofing)
    - Empirically improved default weights: w_phi=1.5, w_secondary=1.0, w_c=2.0, w_e=0.5

    Enable only after D2–D4 have been individually validated on real data and D4 has
    been shown to provide statistically significant improvement over individual baselines.

    The correlation term is (1 - |C_t|), NOT |C_t|. The physical mechanism is:
    - Under authentic tracking, z_phi and z_M are correlated (|C_t| ≈ 1),
      so (1 - |C_t|) ≈ 0 → low score.
    - During spoofing, the synthetic phase drift decouples from AGC, |C_t| collapses
      toward 0, so (1 - |C_t|) → 1 → score spikes.

    The entropy term uses multi-scale rolling entropy: during spoofing onset the
    innovation distribution becomes concentrated (low entropy), so (1 - entropy)
    spikes. During benign tracking entropy is near-maximum → term is near 0.

    Note: novelty requires that C_t (cross-modal coupling) adds independent
    information beyond D4 on REAL data. A paired bootstrap test vs D4 must be
    run before this detector is used as evidentiary support in any filing.
    """
    name = "D6_full_ckec"

    def __init__(
        self,
        secondary_col: str = "z_agc",
        w_phi: float = 1.5,
        w_secondary: float = 1.0,
        w_r: float = 1.0,
        w_c: float = 2.0,
        w_e: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.secondary_col = secondary_col
        self.w_phi = w_phi
        self.w_secondary = w_secondary
        self.w_r = w_r
        self.w_c = w_c
        self.w_e = w_e

    def score(self, X: pd.DataFrame) -> pd.Series:
        self._require_columns(X, "z_phi")
        s = self.w_phi * X["z_phi"] ** 2
        if self.secondary_col in X.columns:
            s = s + self.w_secondary * X[self.secondary_col] ** 2
        else:
            log.warning("D6: '%s' not available - secondary modality term omitted.", self.secondary_col)
        if "z_residual" in X.columns:
            s = s + self.w_r * X["z_residual"] ** 2
        else:
            log.warning("D6: 'z_residual' not available - geometry term omitted (expected for real datasets).")

        # Cross-modal correlation term: use multi-scale if available, else single-scale.
        # Missing correlation currently contributes zero penalty (neutral fallback);
        # it must not be interpreted as evidence that the modalities are consistent.
        if "cross_modal_corr_ms" in X.columns:
            corr_val = X["cross_modal_corr_ms"].fillna(1.0).abs()
            s = s + self.w_c * (1.0 - corr_val).clip(lower=0.0)
        elif "cross_modal_corr" in X.columns:
            corr_val = X["cross_modal_corr"].fillna(1.0).abs()
            s = s + self.w_c * (1.0 - corr_val).clip(lower=0.0)
        else:
            log.warning("D6: no cross-modal correlation feature available - consistency term omitted.")

        # Entropy term: (1 - entropy) spikes when innovation distribution concentrates.
        # Missing entropy currently contributes zero penalty (neutral fallback);
        # it does not mean that maximum entropy was actually observed.
        if "innovation_entropy" in X.columns:
            ent = X["innovation_entropy"].fillna(1.0).clip(0.0, 1.0)
            s = s + self.w_e * (1.0 - ent)
        return s.rename(self.name)


# ---------------------------------------------------------------------------
# D7 — CUSUM-based gradual spoofing detector
# ---------------------------------------------------------------------------

class CUSUMDetector(BaseDetector):
    """
    D7: S_t = max(CUSUM_phi_t, w_sec * CUSUM_secondary_t)

    Page-CUSUM-based detector that accumulates evidence of a mean shift in the
    phase innovation z-scores. Particularly effective for detecting gradual
    spoofing scenarios where single-sample z-scores remain below threshold
    but the cumulative drift crosses a detection boundary.

    The score is the element-wise maximum of the phase CUSUM and an optionally
    weighted secondary-modality CUSUM (if available in the feature matrix).

    This detector complements D2 (phase-only) and D4 (joint) by providing
    temporal integration of weak spoofing evidence.
    """
    name = "D7_cusum"

    def __init__(
        self,
        secondary_col: str = "z_agc",
        w_secondary: float = 0.7,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.secondary_col = secondary_col
        self.w_secondary = w_secondary

    def score(self, X: pd.DataFrame) -> pd.Series:
        """Return the sliding-window CUSUM statistic (cusum_stat column).

        V3: Secondary cumulative sum removed — it was a globally accumulating
        term that re-introduced temporal leakage after the cusum_stat fix.
        D7 now uses only the sliding-window phase CUSUM feature, which is
        stationary and time-position-independent.
        """
        self._require_columns(X, "cusum_stat")
        return X["cusum_stat"].rename(self.name)


# ---------------------------------------------------------------------------
# Detector registry
# ---------------------------------------------------------------------------

ALL_DETECTORS: dict[str, type[BaseDetector]] = {
    "D1_raw_ratio":   RawRatioDetector,
    "D2_phase_only":  PhaseOnlyDetector,
    "D3a_agc_only":   AGCOnlyDetector,
    "D3b_corr_mag":   CorrMagDetector,
    "D3c_cn0_only":   CN0Detector,
    "D4_joint":       JointDetector,
    "D5_geometry":    GeometryDetector,
    "D6_full_ckec":   FullCKECDetector,
    "D7_cusum":       CUSUMDetector,
}

# Backward-compat alias for scripts that still use the old name
ALL_DETECTORS["D3_agc_only"] = AGCOnlyDetector


def build_detector(name: str, **kwargs) -> BaseDetector:
    """Instantiate a detector by registry name."""
    if name not in ALL_DETECTORS:
        raise ValueError(f"Unknown detector '{name}'. Available: {list(ALL_DETECTORS)}")
    return ALL_DETECTORS[name](**kwargs)
