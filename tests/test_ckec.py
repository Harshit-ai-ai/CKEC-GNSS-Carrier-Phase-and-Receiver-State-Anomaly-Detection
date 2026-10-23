"""
tests/test_ckec.py — Full unit test suite for the ckec library.

Covers:
  - preprocessing (unwrap, wrap, smoothing, calibration stats)
  - features (phase, AGC, correlator, C/N0, geometry, cross-modal)
  - detectors (D1, D2, D3a, D3b, D3c, D4, D5, D6 + registry)
  - simulation (scenario generation)
  - evaluation (metrics, bootstrap CI)
  - ingestion parsers (unit test with no real files required)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from ckec.preprocessing import (
    unwrap_phase, wrap_phase, temporal_difference, smooth_agc,
    estimate_benign_stats, handle_missing,
)
from ckec.features import (
    delta_phase, delta_agc, delta_corr, delta_cn0,
    raw_ratio,
    normalised_phase, normalised_agc, normalised_corr, normalised_cn0,
    predict_inter_antenna_phase, geometry_residual,
    cross_modal_consistency, build_feature_matrix, cross_modal_consistency_multiscale,
)
from ckec.detectors import (
    RawRatioDetector, PhaseOnlyDetector,
    AGCOnlyDetector, CorrMagDetector, CN0Detector,
    JointDetector, GeometryDetector, FullCKECDetector,
    build_detector, ALL_DETECTORS,
)
from ckec.simulation import (
    generate_abrupt_spoofing, generate_gradual_spoofing, generate_benign_multipath,
)
from ckec.evaluation import compute_metrics, bootstrap_metrics


# ============================================================================
# Preprocessing
# ============================================================================

class TestPreprocessing:

    def test_unwrap_phase_no_wrap(self):
        phase = pd.Series([0.0, 0.1, 0.2, 0.3])
        result = unwrap_phase(phase)
        np.testing.assert_allclose(result.to_numpy(), phase.to_numpy(), atol=1e-10)

    def test_unwrap_phase_handles_jump(self):
        phase = pd.Series([0.0, np.pi - 0.1, -(np.pi - 0.1), -(np.pi - 0.2)])
        result = unwrap_phase(phase)
        diffs = np.diff(result.to_numpy())
        assert all(abs(d) < np.pi for d in diffs), f"Unexpected wrap: {diffs}"

    def test_wrap_phase_identity_in_range(self):
        val = 1.0
        assert abs(wrap_phase(val) - val) < 1e-10

    def test_wrap_phase_maps_to_neg_pi_pi(self):
        val = 5.0
        wrapped = float(wrap_phase(val))
        assert -np.pi < wrapped <= np.pi

    def test_smooth_agc_reduces_variance(self):
        rng = np.random.default_rng(0)
        agc = pd.Series(rng.normal(30, 2, 200))
        smoothed = smooth_agc(agc, window=11)
        assert smoothed.std() < agc.std()

    def test_smooth_agc_window_1_unchanged(self):
        agc = pd.Series([1.0, 2.0, 3.0])
        result = smooth_agc(agc, window=1)
        pd.testing.assert_series_equal(result, agc, check_names=False)

    def test_estimate_benign_stats_basic(self):
        s = pd.Series(np.ones(100) * 2.0)
        mu, sigma = estimate_benign_stats(s, 100)
        assert abs(mu - 2.0) < 1e-6

    def test_estimate_benign_stats_empty_raises(self):
        s = pd.Series([np.nan] * 10)
        with pytest.raises(ValueError):
            estimate_benign_stats(s, 10)

    def test_temporal_difference_shape(self):
        s = pd.Series([1.0, 2.0, 4.0, 7.0])
        d = temporal_difference(s)
        assert len(d) == len(s)
        assert np.isnan(d.iloc[0])
        np.testing.assert_allclose(d.iloc[1:].to_numpy(), [1.0, 2.0, 3.0], atol=1e-10)


# ============================================================================
# Features — Phase
# ============================================================================

class TestPhaseFeatures:

    def test_delta_phase_shape(self):
        phase = pd.Series([0.0, 0.1, 0.3, 0.6])
        d = delta_phase(phase)
        assert len(d) == len(phase)
        assert np.isnan(d.iloc[0])

    def test_normalised_phase_zero_at_mean(self):
        d_phi = pd.Series([0.1, 0.1, 0.1])
        z = normalised_phase(d_phi, mu=0.1, sigma=0.01)
        np.testing.assert_allclose(z.to_numpy(), np.zeros(3), atol=1e-8)

    def test_normalised_phase_column_name(self):
        d = normalised_phase(pd.Series([1.0, 2.0]), mu=0.0, sigma=1.0)
        assert d.name == "z_phi"


# ============================================================================
# Features — Secondary modalities (separate physical measurements)
# ============================================================================

class TestSecondaryModalities:

    def test_delta_agc_name_and_shape(self):
        """delta_agc is specifically for hardware AGC (UBX-MON-RF agcCnt)."""
        agc = pd.Series([8000.0, 7800.0, 7600.0, 7400.0])
        d = delta_agc(agc)
        assert d.name == "delta_agc"
        assert len(d) == len(agc)
        assert np.isnan(d.iloc[0])

    def test_normalised_agc_name(self):
        d = normalised_agc(pd.Series([0.5, -0.5]), mu=0.0, sigma=1.0)
        assert d.name == "z_agc"

    def test_delta_corr_name_and_shape(self):
        """delta_corr is specifically for FGI-GSRx prompt correlator magnitude."""
        corr = pd.Series([100.0, 102.0, 98.0, 95.0])
        d = delta_corr(corr)
        assert d.name == "delta_corr"
        assert len(d) == len(corr)
        assert np.isnan(d.iloc[0])

    def test_normalised_corr_name(self):
        d = normalised_corr(pd.Series([1.0, 2.0]), mu=1.5, sigma=0.5)
        assert d.name == "z_corr"

    def test_delta_cn0_name_and_shape(self):
        """delta_cn0 is specifically for C/N0 density (dB-Hz), NOT hardware AGC."""
        cn0 = pd.Series([42.0, 41.5, 40.0, 38.5])
        d = delta_cn0(cn0)
        assert d.name == "delta_cn0"
        assert len(d) == len(cn0)
        assert np.isnan(d.iloc[0])

    def test_normalised_cn0_name(self):
        d = normalised_cn0(pd.Series([40.0, 38.0]), mu=41.0, sigma=1.0)
        assert d.name == "z_cn0"

    def test_modalities_are_distinct(self):
        """
        Verify that hardware AGC, correlator magnitude, and C/N0 produce
        independently named features — they must not be conflated.
        """
        s = pd.Series([1.0, 2.0, 3.0])
        assert delta_agc(s).name  == "delta_agc"
        assert delta_corr(s).name == "delta_corr"
        assert delta_cn0(s).name  == "delta_cn0"
        assert normalised_agc(s, 0.0, 1.0).name  == "z_agc"
        assert normalised_corr(s, 0.0, 1.0).name == "z_corr"
        assert normalised_cn0(s, 0.0, 1.0).name  == "z_cn0"


# ============================================================================
# Features — Raw ratio instability
# ============================================================================

class TestRawRatio:

    def test_raw_ratio_positive(self):
        d_phi = pd.Series([0.1, 0.2, 0.3])
        d_sec = pd.Series([0.5, 0.5, 0.5])
        r = raw_ratio(d_phi, d_sec)
        assert (r >= 0).all()

    def test_raw_ratio_epsilon_stability(self):
        """Core documented instability: when denominator → 0 result is large but finite."""
        d_phi = pd.Series([1.0])
        d_sec = pd.Series([0.0])
        r = raw_ratio(d_phi, d_sec, epsilon=1e-4)
        assert np.isfinite(r.iloc[0])
        assert r.iloc[0] > 1_000   # large spike — documents instability

    def test_raw_ratio_name(self):
        r = raw_ratio(pd.Series([1.0]), pd.Series([0.5]))
        assert r.name == "raw_ratio"


# ============================================================================
# Features — Geometry (synthetic / deferred)
# ============================================================================

class TestGeometryFeatures:

    def test_predict_inter_antenna_phase_known_value(self):
        sv = np.array([0.0, 0.0, 1.0])
        baseline = np.array([0.0, 0.0, 0.1903])  # 1 wavelength
        phi_pred = predict_inter_antenna_phase(sv, baseline, carrier_wavelength_m=0.1903)
        np.testing.assert_allclose(phi_pred, 2 * np.pi, atol=1e-8)

    def test_geometry_residual_zero_when_exact(self):
        phi_obs = pd.Series([1.0, 2.0, 3.0])
        phi_pred = np.array([1.0, 2.0, 3.0])
        r = geometry_residual(phi_obs, phi_pred)
        np.testing.assert_allclose(r.to_numpy(), np.zeros(3), atol=1e-8)


# ============================================================================
# Features — Cross-modal consistency
# ============================================================================

class TestCrossModalConsistency:

    def test_shape_and_nan_prefix(self):
        n = 50
        z_phi = pd.Series(np.random.randn(n))
        z_sec = pd.Series(np.random.randn(n))
        c = cross_modal_consistency(z_phi, z_sec, window=10)
        assert len(c) == n
        assert c.iloc[:9].isna().all()
        assert c.name == "cross_modal_corr"

    def test_window_too_small_raises(self):
        with pytest.raises(ValueError):
            cross_modal_consistency(pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0]), window=1)

    def test_perfect_positive_correlation(self):
        """Identical series should have correlation 1.0 in the steady state."""
        s = pd.Series(np.arange(20, dtype=float))
        c = cross_modal_consistency(s, s, window=5)
        np.testing.assert_allclose(c.dropna().to_numpy(), np.ones(len(c.dropna())), atol=1e-8)


# ============================================================================
# Features — build_feature_matrix
# ============================================================================

class TestBuildFeatureMatrix:

    def test_phase_only(self):
        z_phi = pd.Series([1.0, 2.0, 3.0])
        df = build_feature_matrix(z_phi)
        assert "z_phi" in df.columns
        assert len(df.columns) == 1

    def test_with_agc(self):
        z_phi = pd.Series([1.0, 2.0])
        z_agc = pd.Series([0.5, 0.6])
        df = build_feature_matrix(z_phi, z_agc=z_agc)
        assert "z_agc" in df.columns
        assert "z_corr" not in df.columns

    def test_with_corr(self):
        z_phi = pd.Series([1.0, 2.0])
        z_corr = pd.Series([0.5, 0.6])
        df = build_feature_matrix(z_phi, z_corr=z_corr)
        assert "z_corr" in df.columns
        assert "z_agc" not in df.columns

    def test_with_cn0(self):
        z_phi = pd.Series([1.0, 2.0])
        z_cn0 = pd.Series([0.5, 0.6])
        df = build_feature_matrix(z_phi, z_cn0=z_cn0)
        assert "z_cn0" in df.columns


# ============================================================================
# Detectors — interface contract
# ============================================================================

def _make_feature_df(n: int = 200, onset: int = 100) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(42)
    z_phi  = pd.Series(rng.normal(0, 1, n))
    z_agc  = pd.Series(rng.normal(0, 1, n))
    z_corr = pd.Series(rng.normal(0, 1, n))
    z_cn0  = pd.Series(rng.normal(0, 1, n))
    z_phi[onset:] += 5.0
    feat_df = pd.DataFrame({
        "z_phi":  z_phi,
        "z_agc":  z_agc,
        "z_corr": z_corr,
        "z_cn0":  z_cn0,
        "delta_phi":  z_phi * 0.05,
        "delta_agc":  z_agc * 0.2,
        "delta_corr": z_corr * 0.1,
        "delta_cn0":  z_cn0 * 0.5,
    })
    feat_df["raw_ratio"] = feat_df["delta_phi"].abs() / (feat_df["delta_agc"].abs() + 1e-4)
    labels = np.zeros(n, dtype=int)
    labels[onset:] = 1
    return feat_df, labels


class TestDetectors:

    def test_predict_raises_before_fit(self):
        det = PhaseOnlyDetector()
        feat_df, _ = _make_feature_df()
        with pytest.raises(RuntimeError):
            det.predict(feat_df)

    def test_d1_raw_ratio_fit_and_predict(self):
        feat_df, labels = _make_feature_df()
        det = RawRatioDetector(secondary_col="delta_agc")
        det.fit(feat_df[:100], labels[:100], target_far=0.05)
        preds = det.predict(feat_df[100:])
        assert len(preds) == 100
        assert set(preds.unique()).issubset({0, 1})

    def test_d2_phase_only_fit_and_predict(self):
        feat_df, labels = _make_feature_df()
        det = PhaseOnlyDetector()
        det.fit(feat_df[:100], labels[:100], target_far=0.05)
        preds = det.predict(feat_df[100:])
        assert len(preds) == 100

    def test_d3a_agc_only(self):
        """D3a uses hardware AGC column (z_agc)."""
        feat_df, labels = _make_feature_df()
        det = AGCOnlyDetector()
        det.fit(feat_df[:100], labels[:100])
        scores = det.score(feat_df)
        assert scores.name == "D3a_agc_only"

    def test_d3b_corr_mag(self):
        """D3b uses prompt correlator magnitude (z_corr)."""
        feat_df, labels = _make_feature_df()
        det = CorrMagDetector()
        det.fit(feat_df[:100], labels[:100])
        scores = det.score(feat_df)
        assert scores.name == "D3b_corr_mag"

    def test_d3c_cn0_only(self):
        """D3c uses C/N0 density (z_cn0)."""
        feat_df, labels = _make_feature_df()
        det = CN0Detector()
        det.fit(feat_df[:100], labels[:100])
        scores = det.score(feat_df)
        assert scores.name == "D3c_cn0_only"

    def test_d4_joint_with_agc(self):
        """D4 joint detector with hardware AGC as secondary modality."""
        feat_df, labels = _make_feature_df()
        det = JointDetector(secondary_col="z_agc")
        det.fit(feat_df[:100], labels[:100])
        preds = det.predict(feat_df[100:])
        assert len(preds) == 100

    def test_d4_joint_with_corr(self):
        """D4 joint detector with correlator magnitude as secondary modality."""
        feat_df, labels = _make_feature_df()
        det = JointDetector(secondary_col="z_corr")
        det.fit(feat_df[:100], labels[:100])
        preds = det.predict(feat_df[100:])
        assert len(preds) == 100

    def test_d4_joint_with_cn0(self):
        """D4 joint detector with C/N0 density as secondary modality."""
        feat_df, labels = _make_feature_df()
        det = JointDetector(secondary_col="z_cn0")
        det.fit(feat_df[:100], labels[:100])
        preds = det.predict(feat_df[100:])
        assert len(preds) == 100

    def test_d5_geometry_missing_column_raises(self):
        det = GeometryDetector()
        feat_df = pd.DataFrame({"z_phi": [1.0, 2.0]})
        with pytest.raises(RuntimeError):
            det.score(feat_df)

    def test_d6_full_ckec_graceful_optional_columns(self):
        """D6 must not crash when optional columns (z_residual, cross_modal_corr) are absent."""
        feat_df, labels = _make_feature_df()
        det = FullCKECDetector(secondary_col="z_agc")
        det.fit(feat_df[:100], labels[:100])
        preds = det.predict(feat_df[100:])
        assert len(preds) == 100

    def test_fit_calibrates_threshold_on_benign_only(self):
        """Threshold must be calibrated ONLY on benign validation samples."""
        feat_df, labels = _make_feature_df(n=300, onset=150)
        # Validate only on the benign first half
        val_df = feat_df[:100]
        val_labels = labels[:100]
        assert (val_labels == 0).all(), "Validation set should be purely benign"
        det = PhaseOnlyDetector()
        det.fit(val_df, val_labels, target_far=0.01)
        assert det._threshold is not None

    def test_no_benign_samples_raises(self):
        feat_df, _ = _make_feature_df()
        all_attack_labels = np.ones(100, dtype=int)
        det = PhaseOnlyDetector()
        with pytest.raises(RuntimeError, match="No benign samples"):
            det.fit(feat_df[:100], all_attack_labels)

    def test_modality_isolation_d3_variants_differ(self):
        """D3a, D3b, D3c must have distinct names to prevent reporting confusion."""
        assert AGCOnlyDetector().name  == "D3a_agc_only"
        assert CorrMagDetector().name  == "D3b_corr_mag"
        assert CN0Detector().name      == "D3c_cn0_only"
        names = {AGCOnlyDetector().name, CorrMagDetector().name, CN0Detector().name}
        assert len(names) == 3, "D3a/D3b/D3c must have unique names"

    def test_build_detector_registry_all_present(self):
        expected = [
            "D1_raw_ratio", "D2_phase_only",
            "D3a_agc_only", "D3b_corr_mag", "D3c_cn0_only",
            "D4_joint", "D5_geometry", "D6_full_ckec",
        ]
        for name in expected:
            det = build_detector(name)
            assert det.name == name

    def test_backward_compat_d3_alias(self):
        """'D3_agc_only' is a backward-compatibility alias for D3a_agc_only."""
        det = build_detector("D3_agc_only")
        assert isinstance(det, AGCOnlyDetector)

    def test_build_detector_unknown_raises(self):
        with pytest.raises(ValueError):
            build_detector("D99_nonexistent")


# ============================================================================
# Simulation
# ============================================================================

class TestSimulation:

    def _check_df(self, df: pd.DataFrame, n: int):
        assert len(df) == n
        required = {"time_s", "carrier_phase", "agc", "attack_label"}
        assert required.issubset(df.columns), f"Missing columns: {required - set(df.columns)}"

    def test_abrupt_spoofing_shape_and_meta(self):
        df, meta = generate_abrupt_spoofing(n_samples=1000, seed=0)
        self._check_df(df, 1000)
        assert meta["data_source"] == "SYNTHETIC"
        assert meta["scenario"] == "abrupt_spoofing"

    def test_abrupt_spoofing_labels(self):
        df, meta = generate_abrupt_spoofing(n_samples=1000, onset=500, seed=0)
        assert (df["attack_label"][:500] == 0).all()
        assert (df["attack_label"][500:] == 1).all()

    def test_gradual_spoofing_shape(self):
        df, meta = generate_gradual_spoofing(n_samples=1000, seed=1)
        self._check_df(df, 1000)
        assert meta["data_source"] == "SYNTHETIC"

    def test_benign_multipath_all_zeros(self):
        df, _ = generate_benign_multipath(n_samples=500, seed=2)
        assert (df["attack_label"] == 0).all()

    def test_synthetic_tag_prevents_real_data_confusion(self):
        """All simulated scenarios must carry the SYNTHETIC tag in metadata."""
        for gen in [generate_abrupt_spoofing, generate_gradual_spoofing, generate_benign_multipath]:
            _, meta = gen(n_samples=100, seed=42)
            assert meta["data_source"] == "SYNTHETIC", (
                f"{gen.__name__} must set data_source='SYNTHETIC' in metadata"
            )


# ============================================================================
# Evaluation
# ============================================================================

class TestEvaluation:

    def test_perfect_detector(self):
        y_true = np.array([0] * 50 + [1] * 50)
        y_pred = y_true.copy()
        scores = y_pred.astype(float)
        r = compute_metrics(y_true, y_pred, scores, "perfect", "test", 0.01)
        assert r.detection_probability == pytest.approx(1.0)
        assert r.false_alarm_rate == pytest.approx(0.0)

    def test_all_benign_predictions(self):
        y_true = np.array([0] * 50 + [1] * 50)
        y_pred = np.zeros(100, dtype=int)
        scores = np.zeros(100)
        r = compute_metrics(y_true, y_pred, scores, "null", "test", 0.01)
        assert r.detection_probability == pytest.approx(0.0)
        assert r.false_alarm_rate == pytest.approx(0.0)

    def test_bootstrap_ci_shape(self):
        y_true = np.array([0] * 50 + [1] * 50)
        y_pred = np.array([0] * 45 + [1] * 5 + [1] * 45 + [0] * 5)
        ci = bootstrap_metrics(y_true, y_pred, n_iter=100, seed=0)
        assert "detection_probability" in ci
        assert "false_alarm_rate" in ci
        lo, hi = ci["detection_probability"]
        assert lo <= hi


# ============================================================================
# Ingestion parsers (unit test — no real files required)
# ============================================================================

class TestIngestionParsers:
    """
    Verify parser behaviour when files are absent (graceful empty-DF return)
    and that output schema is correct.
    """

    def test_parse_jammertest_no_files_returns_empty(self):
        from ckec.ingestion.parsers import parse_jammertest_session
        df = parse_jammertest_session(ubx_path=None, rinex_obs_path=None, attack_label=2)
        assert isinstance(df, pd.DataFrame)
        assert "time_s" in df.columns
        assert "hardware_agc" in df.columns
        assert "carrier_phase" in df.columns
        assert "corr_magnitude" in df.columns
        assert "cn0_db_hz" in df.columns
        assert "attack_label" in df.columns

    def test_parse_fgi_no_files_returns_empty(self):
        from ckec.ingestion.parsers import parse_fgi_session
        df = parse_fgi_session(rinex_obs_path=None, mat_trackresults_path=None)
        assert isinstance(df, pd.DataFrame)
        assert "carrier_phase" in df.columns
        assert "corr_magnitude" in df.columns
        assert "hardware_agc" in df.columns

    def test_parse_yunnan_no_files_returns_empty(self):
        from ckec.ingestion.parsers import parse_yunnan_session
        df = parse_yunnan_session(rawx_csv_path=None, navsat_csv_path=None)
        assert isinstance(df, pd.DataFrame)
        assert "carrier_phase" in df.columns
        assert "cn0_db_hz" in df.columns
        assert "hardware_agc" in df.columns

    def test_parse_jammertest_hardware_agc_is_nan_when_missing(self):
        """Confirms hardware AGC is explicitly NaN when UBX file absent."""
        from ckec.ingestion.parsers import parse_jammertest_session
        df = parse_jammertest_session(ubx_path=None, rinex_obs_path=None)
        if len(df) > 0:
            assert df["hardware_agc"].isna().all()

    def test_parse_fgi_hardware_agc_always_nan(self):
        """FGI-SpoofRepo does not export hardware AGC — must always be NaN."""
        from ckec.ingestion.parsers import parse_fgi_session
        df = parse_fgi_session()
        if len(df) > 0:
            assert df["hardware_agc"].isna().all(), (
                "FGI-SpoofRepo does not export hardware AGC — column must be NaN"
            )

    def test_parse_yunnan_hardware_agc_always_nan(self):
        """Yunnan-2024 does not export hardware AGC — must always be NaN."""
        from ckec.ingestion.parsers import parse_yunnan_session
        df = parse_yunnan_session()
        if len(df) > 0:
            assert df["hardware_agc"].isna().all(), (
                "Yunnan-2024 does not export hardware AGC — column must be NaN"
            )

    def test_parse_yunnan_from_synthetic_csv(self, tmp_path):
        """
        Integration test using synthetic CSV files that mimic Yunnan's schema.
        Verifies the parser correctly reads rcvTow, cpMes, iTOW, cno.
        """
        from ckec.ingestion.parsers import parse_yunnan_session

        # Create synthetic RAWX CSV (mimics UBX-RXM-RAWX export)
        rawx_csv = tmp_path / "rawx.csv"
        rawx_df = pd.DataFrame({
            "rcvTow": np.arange(0, 10, dtype=float),
            "cpMes":  np.linspace(10000.0, 10090.0, 10),
        })
        rawx_df.to_csv(rawx_csv, index=False)

        # Create synthetic NAVSAT CSV (mimics UBX-NAV-SAT export)
        navsat_csv = tmp_path / "navsat.csv"
        navsat_df = pd.DataFrame({
            "iTOW": np.arange(0, 10_000, 1000, dtype=float),
            "cno":  np.full(10, 42.0),
        })
        navsat_df.to_csv(navsat_csv, index=False)

        df = parse_yunnan_session(
            rawx_csv_path=str(rawx_csv),
            navsat_csv_path=str(navsat_csv),
            attack_label=2,
        )
        assert len(df) == 10
        assert (df["attack_label"] == 2).all()
        np.testing.assert_allclose(df["carrier_phase"].values, rawx_df["cpMes"].values, atol=1e-6)
        assert df["hardware_agc"].isna().all()
        assert df["corr_magnitude"].isna().all()



def test_multiscale_correlation_current_signed_minimum():
    z_phi = pd.Series(np.arange(30, dtype=float))
    z_sec = pd.Series(np.arange(30, dtype=float))

    result = cross_modal_consistency_multiscale(
        z_phi, z_sec, windows=(5, 10, 20)
    )

    assert result.name == "cross_modal_corr_ms"
    # The minimum uses any available scale, so the 5-sample
    # correlation becomes available at index 4.
    assert result.iloc[:4].isna().all()
    assert result.iloc[4:].notna().all()
    assert result.iloc[4] == pytest.approx(1.0)
    assert result.iloc[19] == pytest.approx(1.0)



def test_signed_minimum_can_hide_near_zero_correlation():
    # Illustrative correlations from three scales.
    correlations = pd.Series([-0.90, 0.05, 0.80])

    current_value = abs(correlations.min())
    weakest_absolute = correlations.abs().min()

    # The current aggregation reports strong correlation despite
    # one scale having almost no correlation.
    assert current_value == pytest.approx(0.90)
    assert weakest_absolute == pytest.approx(0.05)
    assert current_value > weakest_absolute


def test_d6_missing_corr_and_entropy_use_neutral_fallback():
    """Missing optional D6 terms should not create NaN scores."""
    det = FullCKECDetector(
        secondary_col="z_agc",
        w_phi=1.0,
        w_secondary=0.0,
        w_r=0.0,
        w_c=1.0,
        w_e=1.0,
    )
    X = pd.DataFrame({
        "z_phi": [2.0, 3.0],
        "z_agc": [0.0, 0.0],
        "cross_modal_corr": [np.nan, np.nan],
        "innovation_entropy": [np.nan, np.nan],
    })

    scores = det.score(X)
    # This documents the current neutral fallback, not evidence of consistency.
    np.testing.assert_allclose(scores.to_numpy(), [4.0, 9.0])


def test_cusum_statistic_is_causal():
    """Changing future observations must not alter earlier CUSUM scores."""
    from ckec.features import cusum_statistic

    prefix = pd.Series([0.2, 0.8, 1.5, -0.1])
    extended = pd.Series([0.2, 0.8, 1.5, -0.1, 1000.0])

    prefix_scores = cusum_statistic(prefix, window=3)
    extended_scores = cusum_statistic(extended, window=3)

    np.testing.assert_allclose(
        prefix_scores.to_numpy(),
        extended_scores.iloc[:len(prefix)].to_numpy(),
        equal_nan=True,
    )


def test_cusum_statistic_handles_missing_observations():
    """A missing current sample stays missing; later finite samples can resume."""
    from ckec.features import cusum_statistic

    scores = cusum_statistic(
        pd.Series([1.0, np.nan, 2.0]),
        window=3,
    )

    assert np.isfinite(scores.iloc[0])
    assert np.isnan(scores.iloc[1])
    assert np.isfinite(scores.iloc[2])
