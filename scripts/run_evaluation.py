"""
scripts/run_evaluation.py
=========================
Comprehensive evaluation runner and ablation analysis for CKEC-GNSS.

Evaluates detectors (D1, D2, D3a/b/c, D4, D6) on processed real datasets or
controlled synthetic scenarios with strict threshold calibration on validation splits.

Usage:
    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --data synthetic --ablation
    python scripts/run_evaluation.py --config config.yaml --target-far 0.01
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import click
import numpy as np
import pandas as pd

from ckec.utils import setup_logging, load_config, get_path, set_seed
from ckec.preprocessing import (
    unwrap_phase, smooth_agc, handle_missing, estimate_benign_stats,
    clip_innovations,
)
from ckec.features import (
    delta_phase, delta_agc, delta_corr, delta_cn0,
    normalised_phase, normalised_agc, normalised_corr, normalised_cn0,
    raw_ratio, cross_modal_consistency, cross_modal_consistency_multiscale,
    innovation_entropy, cusum_statistic, build_feature_matrix,
)
from ckec.detectors import (
    RawRatioDetector, PhaseOnlyDetector, AGCOnlyDetector, CorrMagDetector,
    CN0Detector, JointDetector, FullCKECDetector, CUSUMDetector,
)
from ckec.evaluation import (
    compute_metrics, bootstrap_metrics, detection_delay,
    results_to_dataframe, print_results_table, DetectorResult,
)
from ckec.simulation import generate_all_scenarios


# ---------------------------------------------------------------------------
# Feature Preparation
# ---------------------------------------------------------------------------

def prepare_features(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    window_corr: int = 20,
    epsilon: float = 1e-4,
) -> pd.DataFrame:
    """
    Extract all features from raw/synthetic scenario DataFrame.
    Automatically identifies available secondary modality (hardware_agc, agc,
    corr_magnitude, cn0_db_hz, or cn0).

    V2: Uses clip_innovations for outlier pre-filtering, MAD-based calibration,
    multi-scale cross-modal correlation, innovation entropy, and CUSUM statistic.
    """
    pp_cfg = cfg.get("preprocessing", {})
    smooth_w = pp_cfg.get("agc_smoothing_window", 5)
    strategy = pp_cfg.get("missing_sample_strategy", "interpolate")
    cal_seconds = pp_cfg.get("calibration_split_seconds", 60)

    df_prep = df.copy()

    # Determine time step
    dt = df_prep["time_s"].diff().median() if "time_s" in df_prep.columns else 0.1
    if not dt or np.isnan(dt):
        dt = 0.1
    cal_n = int(cal_seconds * (1.0 / dt))

    # Phase — unwrap, handle missing, then clip extreme innovations before z-score
    phase_unwrapped = unwrap_phase(df_prep["carrier_phase"])
    df_prep["carrier_phase"] = phase_unwrapped
    df_prep = handle_missing(df_prep, strategy=strategy, max_gap=10)
    d_phi_raw = delta_phase(df_prep["carrier_phase"])
    d_phi = clip_innovations(d_phi_raw, n_sigma=8.0, window=50)
    mu_phi, sigma_phi = estimate_benign_stats(d_phi, min(cal_n, max(20, len(d_phi) // 4)), use_mad=True)
    z_phi = normalised_phase(d_phi, mu_phi, sigma_phi, epsilon=epsilon)

    # Detect secondary modality (priority: hardware AGC > agc > corr_magnitude > cn0)
    z_agc, z_corr, z_cn0 = None, None, None
    secondary_col = "z_agc"
    d_sec = None

    if "hardware_agc" in df_prep.columns and not df_prep["hardware_agc"].isna().all():
        agc_smoothed = smooth_agc(df_prep["hardware_agc"], window=smooth_w)
        d_agc = clip_innovations(delta_agc(agc_smoothed), n_sigma=8.0)
        mu_agc, sigma_agc = estimate_benign_stats(d_agc, min(cal_n, max(20, len(d_agc) // 4)), use_mad=True)
        z_agc = normalised_agc(d_agc, mu_agc, sigma_agc, epsilon=epsilon)
        secondary_col = "z_agc"
        d_sec = d_agc
    elif "agc" in df_prep.columns and not df_prep["agc"].isna().all():
        agc_smoothed = smooth_agc(df_prep["agc"], window=smooth_w)
        d_agc = clip_innovations(delta_agc(agc_smoothed), n_sigma=8.0)
        mu_agc, sigma_agc = estimate_benign_stats(d_agc, min(cal_n, max(20, len(d_agc) // 4)), use_mad=True)
        z_agc = normalised_agc(d_agc, mu_agc, sigma_agc, epsilon=epsilon)
        secondary_col = "z_agc"
        d_sec = d_agc
    elif "corr_magnitude" in df_prep.columns and not df_prep["corr_magnitude"].isna().all():
        d_corr_s = clip_innovations(delta_corr(df_prep["corr_magnitude"]), n_sigma=8.0)
        mu_c, sigma_c = estimate_benign_stats(d_corr_s, min(cal_n, max(20, len(d_corr_s) // 4)), use_mad=True)
        z_corr = normalised_corr(d_corr_s, mu_c, sigma_c, epsilon=epsilon)
        secondary_col = "z_corr"
        d_sec = d_corr_s
    elif "cn0_db_hz" in df_prep.columns and not df_prep["cn0_db_hz"].isna().all():
        d_cn0_s = clip_innovations(delta_cn0(df_prep["cn0_db_hz"]), n_sigma=8.0)
        mu_cn, sigma_cn = estimate_benign_stats(d_cn0_s, min(cal_n, max(20, len(d_cn0_s) // 4)), use_mad=True)
        z_cn0 = normalised_cn0(d_cn0_s, mu_cn, sigma_cn, epsilon=epsilon)
        secondary_col = "z_cn0"
        d_sec = d_cn0_s

    # Active secondary z-score series (the one actually used)
    z_secondary_active = z_agc if z_agc is not None else (z_corr if z_corr is not None else z_cn0)

    # Cross-modal correlation — single-scale (for D6 fallback) and multi-scale
    if z_secondary_active is not None:
        c_t = cross_modal_consistency(z_phi, z_secondary_active, window=max(3, window_corr))
        c_t_ms = cross_modal_consistency_multiscale(
            z_phi, z_secondary_active, windows=(5, 10, min(20, window_corr))
        )
    else:
        c_t = pd.Series(0.0, index=z_phi.index, name="cross_modal_corr")
        c_t_ms = pd.Series(0.0, index=z_phi.index, name="cross_modal_corr_ms")

    # Innovation entropy (rolling normalised Shannon entropy of phase z-scores)
    ent = innovation_entropy(z_phi, window=max(10, window_corr // 2))

    # CUSUM statistic for gradual drift detection
    cusum = cusum_statistic(z_phi, k=0.5)

    # Build matrix
    feat_df = build_feature_matrix(
        z_phi=z_phi,
        z_agc=z_agc,
        z_corr=z_corr,
        z_cn0=z_cn0,
        cross_modal=c_t,
        cross_modal_ms=c_t_ms,
        entropy=ent,
        cusum=cusum,
    )
    feat_df["delta_phi"] = d_phi
    if d_sec is not None:
        feat_df["delta_sec"] = d_sec
        feat_df["raw_ratio"] = raw_ratio(d_phi, d_sec, epsilon=epsilon)
    else:
        feat_df["raw_ratio"] = np.nan

    feat_df["attack_label"] = df_prep["attack_label"].values
    if "time_s" in df_prep.columns:
        feat_df["time_s"] = df_prep["time_s"].values

    return feat_df.dropna(subset=["z_phi"])


# ---------------------------------------------------------------------------
# Train / Val / Test Split
# ---------------------------------------------------------------------------

def train_val_test_split(
    feat_df: pd.DataFrame,
    val_frac: float = 0.15,
    test_frac: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split time-series into train, val, and test splits.
    Ensures pre-attack benign observations are used for calibration without data snooping.
    """
    n = len(feat_df)
    benign_mask = feat_df["attack_label"] == 0

    if benign_mask.any() and (~benign_mask).any():
        first_attack = int(np.where(~benign_mask)[0][0])
        # Calibration section within benign portion
        calib_cutoff = max(int(first_attack * 0.8), int(n * 0.3))
        train = feat_df.iloc[: int(calib_cutoff * 0.6)]
        val = feat_df.iloc[int(calib_cutoff * 0.6) : calib_cutoff]
        test = feat_df.iloc[calib_cutoff:]
    else:
        n_test = int(n * test_frac)
        n_val = int(n * val_frac)
        n_train = n - n_val - n_test
        train = feat_df.iloc[:n_train]
        val = feat_df.iloc[n_train : n_train + n_val]
        test = feat_df.iloc[n_train + n_val :]

    return train, val, test


# ---------------------------------------------------------------------------
# Scenario Evaluator
# ---------------------------------------------------------------------------

def evaluate_detectors(
    scenario_name: str,
    feat_df: pd.DataFrame,
    target_far: float = 0.01,
    epsilon: float = 1e-4,
    secondary_col: str = "z_agc",
    bootstrap_iter: int = 500,
    seed: int = 42,
) -> list[DetectorResult]:
    """Fit detectors on validation split, evaluate on test split, and return metrics."""
    _, val_df, test_df = train_val_test_split(feat_df)

    y_val = val_df["attack_label"].to_numpy()
    y_test = test_df["attack_label"].to_numpy()

    # Detect onset in test split
    attack_indices = np.where(y_test == 1)[0]
    onset_idx = int(attack_indices[0]) if len(attack_indices) > 0 else 0

    # Build detector candidate list
    candidate_detectors = [
        RawRatioDetector(epsilon=epsilon),
        PhaseOnlyDetector(),
    ]
    if secondary_col == "z_agc" and "z_agc" in feat_df.columns:
        candidate_detectors.append(AGCOnlyDetector())
    elif secondary_col == "z_corr" and "z_corr" in feat_df.columns:
        candidate_detectors.append(CorrMagDetector())
    elif secondary_col == "z_cn0" and "z_cn0" in feat_df.columns:
        candidate_detectors.append(CN0Detector())

    if secondary_col in feat_df.columns:
        candidate_detectors.append(JointDetector(secondary_col=secondary_col))
        candidate_detectors.append(
            FullCKECDetector(secondary_col=secondary_col, w_phi=1.5, w_secondary=1.0,
                              w_c=2.0, w_e=0.5)
        )

    # D7 CUSUM detector (requires cusum_stat column)
    if "cusum_stat" in feat_df.columns:
        candidate_detectors.append(CUSUMDetector(secondary_col=secondary_col))

    results: list[DetectorResult] = []

    for det in candidate_detectors:
        try:
            t0 = time.perf_counter()
            det.fit(val_df, y_val, target_far=target_far)
            scores_test = det.score(test_df).to_numpy()
            y_pred_test = det.predict(test_df).to_numpy()
            runtime = time.perf_counter() - t0

            r = compute_metrics(
                y_true=y_test,
                y_pred=y_pred_test,
                scores=scores_test,
                detector_name=f"{det.name}|{scenario_name}",
                split="test",
                target_far=target_far,
            )
            r.runtime_seconds = runtime
            if len(attack_indices) > 0:
                r.detection_delay_samples = detection_delay(y_test, y_pred_test, onset_idx)

            # Bootstrap CI
            ci = bootstrap_metrics(y_test, y_pred_test, n_iter=bootstrap_iter, seed=seed)
            r.ci_detection_probability = ci["detection_probability"]
            r.ci_false_alarm_rate = ci["false_alarm_rate"]

        except Exception as exc:
            r = DetectorResult(
                detector_name=f"{det.name}|{scenario_name}",
                split="test",
                target_far=target_far,
                notes=str(exc),
            )
        results.append(r)

    return results


# ---------------------------------------------------------------------------
# Ablation Runner
# ---------------------------------------------------------------------------

def run_ablation_analysis(
    scenario_name: str,
    raw_df: pd.DataFrame,
    cfg: dict[str, Any],
    target_far: float = 0.01,
) -> pd.DataFrame:
    """
    Perform ablation over:
    1. Rolling window length W in [5, 10, 20, 30, 50]
    2. Cross-modal coupling weight w_c in [0.0, 0.5, 1.0, 2.0]
    """
    windows = [5, 10, 20, 30, 50]
    weights_c = [0.0, 0.5, 1.0, 2.0]

    records = []
    for w in windows:
        feat_df = prepare_features(raw_df, cfg, window_corr=w)
        _, val_df, test_df = train_val_test_split(feat_df)
        y_val = val_df["attack_label"].to_numpy()
        y_test = test_df["attack_label"].to_numpy()
        attack_idx = np.where(y_test == 1)[0]
        onset = int(attack_idx[0]) if len(attack_idx) > 0 else 0

        for wc in weights_c:
            det = FullCKECDetector(secondary_col="z_agc", w_phi=1.0, w_secondary=1.0, w_c=wc)
            det.fit(val_df, y_val, target_far=target_far)
            scores = det.score(test_df).to_numpy()
            preds = det.predict(test_df).to_numpy()

            metrics = compute_metrics(y_test, preds, scores, det.name, "test", target_far)
            delay = detection_delay(y_test, preds, onset) if len(attack_idx) > 0 else float("nan")

            records.append({
                "scenario": scenario_name,
                "window_samples": w,
                "weight_cross_modal": wc,
                "detection_probability": metrics.detection_probability,
                "false_alarm_rate": metrics.false_alarm_rate,
                "precision": metrics.precision,
                "f1": metrics.f1,
                "roc_auc": metrics.roc_auc,
                "pr_auc": metrics.pr_auc,
                "detection_delay": delay,
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

@click.command()
@click.option("--config", default="config.yaml", show_default=True, help="Path to config.yaml")
@click.option(
    "--data",
    default="synthetic",
    show_default=True,
    help="Data source ('synthetic', 'processed', or path to CSV/directory)",
)
@click.option("--target-far", default=None, type=float, help="Override target false alarm rate (e.g. 0.01)")
@click.option("--ablation", is_flag=True, default=False, help="Run cross-modal ablation studies")
@click.option("--output", default="reports/evaluation_results.csv", show_default=True, help="Output CSV path")
def main(
    config: str,
    data: str,
    target_far: float | None,
    ablation: bool,
    output: str,
) -> None:
    """Run comprehensive detector evaluation and ablation analysis."""
    cfg = load_config(config)
    log = setup_logging(level=cfg["logging"]["level"], log_file=cfg["logging"]["file"])
    log.info("=== CKEC-GNSS Evaluation & Benchmarking Runner ===")

    seed = cfg["experiment"].get("seed", 42)
    set_seed(seed)

    effective_target_far = target_far if target_far is not None else cfg["evaluation"]["target_false_alarm_rate"]
    epsilon = cfg["detectors"]["stabilisation_epsilon"]
    bootstrap_iter = cfg["evaluation"].get("ci_bootstrap_iterations", 500)

    all_results: list[DetectorResult] = []
    ablation_dfs: list[pd.DataFrame] = []

    if data == "synthetic":
        log.info("Generating synthetic benchmark scenarios across seeds...")
        seeds = cfg["simulation"]["random_seeds"]
        n_samples = cfg["simulation"]["n_samples"]
        all_scenarios = generate_all_scenarios(n_samples=n_samples, seeds=seeds)

        for scenario_name, runs in all_scenarios.items():
            for seed_idx, (raw_df, _) in enumerate(runs):
                s_name = f"{scenario_name}_seed{seeds[seed_idx]}"
                log.info("Evaluating scenario: %s", s_name)
                feat_df = prepare_features(raw_df, cfg, epsilon=epsilon)
                res = evaluate_detectors(
                    scenario_name=s_name,
                    feat_df=feat_df,
                    target_far=effective_target_far,
                    epsilon=epsilon,
                    secondary_col="z_agc",
                    bootstrap_iter=bootstrap_iter,
                    seed=seed,
                )
                all_results.extend(res)

                if ablation and scenario_name == "abrupt_spoofing" and seed_idx == 0:
                    log.info("Running ablation analysis on %s...", s_name)
                    abl_df = run_ablation_analysis(s_name, raw_df, cfg, effective_target_far)
                    ablation_dfs.append(abl_df)

    else:
        # Load from processed files
        data_path = Path(data)
        if not data_path.exists():
            data_path = Path(__file__).resolve().parent.parent / "data" / "processed"

        csv_files = list(data_path.glob("*.csv")) if data_path.is_dir() else ([data_path] if data_path.exists() else [])
        if not csv_files:
            log.warning("No data files found in %s. Please run ingestion or use --data synthetic.", data_path)
            return

        for f in csv_files:
            log.info("Evaluating real processed session: %s", f.name)
            raw_df = pd.read_csv(f)
            feat_df = prepare_features(raw_df, cfg, epsilon=epsilon)
            sec_col = "z_agc" if "hardware_agc" in raw_df.columns else ("z_corr" if "corr_magnitude" in raw_df.columns else "z_cn0")
            res = evaluate_detectors(
                scenario_name=f.stem,
                feat_df=feat_df,
                target_far=effective_target_far,
                epsilon=epsilon,
                secondary_col=sec_col,
                bootstrap_iter=bootstrap_iter,
                seed=seed,
            )
            all_results.extend(res)

    # Save results
    results_df = results_to_dataframe(all_results)
    out_p = Path(output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_p, index=False)
    log.info("Evaluation results saved to: %s", out_p)

    print("\n=== EVALUATION RESULTS SUMMARY ===")
    print_results_table(all_results)

    if ablation_dfs:
        ablation_summary = pd.concat(ablation_dfs, ignore_index=True)
        abl_out = out_p.parent / "ablation_results.csv"
        ablation_summary.to_csv(abl_out, index=False)
        log.info("Ablation results saved to: %s", abl_out)
        print("\n=== ABLATION RESULTS (Sample) ===")
        print(ablation_summary.head(10).to_string(index=False))

    log.info("Evaluation pipeline completed successfully.")


if __name__ == "__main__":
    main()
