"""
scripts/run_baselines.py

Generate synthetic scenarios and run all enabled detectors.
Produces evaluation metrics and saves results to reports/.

Run:
    python scripts/run_baselines.py                    # uses config.yaml
    python scripts/run_baselines.py --data synthetic   # synthetic only (default)
    python scripts/run_baselines.py --config path/to/config.yaml
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import click
import numpy as np
import pandas as pd

from ckec.utils import setup_logging, load_config, get_path, set_seed
from ckec.preprocessing import (
    unwrap_phase, smooth_agc, handle_missing, estimate_benign_stats,
)
from ckec.features import (
    delta_phase, delta_agc, raw_ratio,
    normalised_phase, normalised_agc,
    build_feature_matrix,
)
from ckec.detectors import (
    RawRatioDetector, PhaseOnlyDetector, AGCOnlyDetector, JointDetector,
)
from ckec.evaluation import (
    compute_metrics, bootstrap_metrics, results_to_dataframe, print_results_table,
    DetectorResult,
)
from ckec.simulation import generate_all_scenarios


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def prepare_features(df: pd.DataFrame, cfg: dict, epsilon: float) -> pd.DataFrame:
    """Preprocess a scenario DataFrame and compute all phase/AGC features."""
    pp_cfg = cfg["preprocessing"]

    phase = unwrap_phase(df["carrier_phase"])
    agc   = smooth_agc(df["agc"], window=pp_cfg["agc_smoothing_window"])

    df_prep = df.copy()
    df_prep["carrier_phase"] = phase
    df_prep["agc"] = agc
    df_prep = handle_missing(
        df_prep,
        strategy=pp_cfg["missing_sample_strategy"],
        max_gap=10,
    )

    d_phi = delta_phase(df_prep["carrier_phase"])
    d_agc_s = delta_agc(df_prep["agc"])

    # Calibration window — first N benign samples
    cal_n = int(pp_cfg["calibration_split_seconds"] *
                (1 / (df["time_s"].diff().median() or 0.1)))
    mu_phi, sigma_phi = estimate_benign_stats(d_phi, cal_n)
    mu_agc, sigma_agc = estimate_benign_stats(d_agc_s, cal_n)

    z_phi = normalised_phase(d_phi, mu_phi, sigma_phi, epsilon=epsilon)
    z_agc = normalised_agc(d_agc_s, mu_agc, sigma_agc, epsilon=epsilon)
    r     = raw_ratio(d_phi, d_agc_s, epsilon=epsilon)

    feat_df = build_feature_matrix(z_phi, z_agc)
    feat_df["delta_phi"] = d_phi
    feat_df["delta_agc"] = d_agc_s
    feat_df["raw_ratio"] = r
    feat_df["attack_label"] = df_prep["attack_label"].values

    return feat_df.dropna()


def train_val_test_split(
    feat_df: pd.DataFrame,
    val_frac: float = 0.1,
    test_frac: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data for evaluation.
    To allow proper calibration without data snooping:
    - Train/Val uses the pre-attack benign calibration segment (or early section).
    - If labels exist, ensure val has benign samples for threshold calibration.
    """
    n = len(feat_df)
    # Check if there is an attack onset
    benign_mask = feat_df["attack_label"] == 0
    if benign_mask.any() and (~benign_mask).any():
        # Scenarios with both benign and attack:
        # Use initial benign section for train/val calibration, test spans benign transition into attack
        first_attack_idx = np.where(~benign_mask)[0][0]
        calib_cutoff = max(int(first_attack_idx * 0.8), int(n * 0.4))
        train = feat_df.iloc[: int(calib_cutoff * 0.7)]
        val = feat_df.iloc[int(calib_cutoff * 0.7) : calib_cutoff]
        test = feat_df.iloc[calib_cutoff:]
    else:
        # Pure benign or pure attack
        n_test = int(n * test_frac)
        n_val = int(n * val_frac)
        n_train = n - n_val - n_test
        train = feat_df.iloc[:n_train]
        val = feat_df.iloc[n_train : n_train + n_val]
        test = feat_df.iloc[n_train + n_val :]
    return train, val, test


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate_scenario(
    scenario_name: str,
    df: pd.DataFrame,
    cfg: dict,
    target_far: float,
    epsilon: float,
) -> list[DetectorResult]:
    """Run all detectors on one synthetic scenario and return results."""
    feat_df = prepare_features(df, cfg, epsilon=epsilon)
    _, val_df, test_df = train_val_test_split(
        feat_df,
        val_frac=cfg["evaluation"]["val_fraction"],
        test_frac=cfg["evaluation"]["test_fraction"],
    )

    y_val  = val_df["attack_label"].to_numpy()
    y_test = test_df["attack_label"].to_numpy()

    detectors = [
        RawRatioDetector(epsilon=epsilon),
        PhaseOnlyDetector(),
        AGCOnlyDetector(),
        JointDetector(),
    ]

    results = []
    for det in detectors:
        try:
            t0 = time.perf_counter()
            det.fit(val_df, pd.Series(y_val), target_far=target_far)
            scores_test = det.score(test_df).to_numpy()
            y_pred_test = det.predict(test_df).to_numpy()
            runtime = time.perf_counter() - t0

            r = compute_metrics(
                y_test, y_pred_test, scores_test,
                detector_name=f"{det.name}|{scenario_name}",
                split="test",
                target_far=target_far,
            )
            r.runtime_seconds = runtime

            # Bootstrap CI
            ci = bootstrap_metrics(
                y_test, y_pred_test,
                n_iter=cfg["evaluation"]["ci_bootstrap_iterations"],
                seed=cfg["experiment"]["seed"],
            )
            r.ci_detection_probability = ci["detection_probability"]
            r.ci_false_alarm_rate      = ci["false_alarm_rate"]

        except RuntimeError as exc:
            r = DetectorResult(
                detector_name=f"{det.name}|{scenario_name}",
                split="test",
                target_far=target_far,
                notes=str(exc),
            )

        results.append(r)
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

@click.command()
@click.option("--config", default="config.yaml", show_default=True)
@click.option("--data", default="synthetic", show_default=True,
              type=click.Choice(["synthetic"]),
              help="Data source.  Only 'synthetic' is supported until real data is ingested.")
def main(config: str, data: str) -> None:
    """Run baseline detectors on synthetic GNSS spoofing scenarios."""
    cfg = load_config(config)
    log = setup_logging(
        level=cfg["logging"]["level"],
        log_file=cfg["logging"]["file"],
    )
    log.info("=== CKEC-GNSS Baseline Experiment ===")
    log.info("Data source: %s  |  Experiment: %s", data, cfg["experiment"]["name"])

    set_seed(cfg["experiment"]["seed"])

    reports_dir = get_path(cfg, "reports")
    target_far  = cfg["evaluation"]["target_false_alarm_rate"]
    epsilon     = cfg["detectors"]["stabilisation_epsilon"]
    seeds       = cfg["simulation"]["random_seeds"]
    n_samples   = cfg["simulation"]["n_samples"]

    # Generate all synthetic scenarios across all seeds
    log.info("Generating synthetic scenarios (seeds=%s) …", seeds)
    all_scenarios = generate_all_scenarios(n_samples=n_samples, seeds=seeds)

    all_results = []
    for scenario_name, runs in all_scenarios.items():
        for seed_idx, (df, meta) in enumerate(runs):
            log.info("  Evaluating: %s  (seed=%s)", scenario_name, seeds[seed_idx])
            res = evaluate_scenario(
                scenario_name=f"{scenario_name}_seed{seeds[seed_idx]}",
                df=df,
                cfg=cfg,
                target_far=target_far,
                epsilon=epsilon,
            )
            all_results.extend(res)

    # Save and print results
    results_df = results_to_dataframe(all_results)
    out_path = reports_dir / "baseline_results.csv"
    results_df.to_csv(out_path, index=False)
    log.info("Results saved to: %s", out_path)

    print("\n=== SUMMARY ===")
    print_results_table(all_results)

    log.info("Baseline experiment complete.")
    log.warning(
        "REMINDER: These are SYNTHETIC results.  "
        "They validate detector behaviour under controlled conditions "
        "but do NOT demonstrate real-world spoofing detection capability."
    )


if __name__ == "__main__":
    main()
