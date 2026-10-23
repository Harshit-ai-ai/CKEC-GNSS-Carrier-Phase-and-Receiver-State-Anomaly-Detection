"""
scripts/ingest_real_data.py
===========================
CLI pipeline to ingest raw datasets (Jammertest-2024, Yunnan-2024, FGI-SpoofRepo)
into standardized, aligned time-series in data/processed/.

Usage:
    python scripts/ingest_real_data.py
    python scripts/ingest_real_data.py --dataset jammertest --generate-samples
    python scripts/ingest_real_data.py --dataset all
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import click
import numpy as np
import pandas as pd

from ckec.utils import setup_logging, load_config, get_path
from ckec.ingestion.parsers import (
    parse_jammertest_session,
    parse_fgi_session,
    parse_yunnan_session,
)


def generate_sample_jammertest_data(raw_dir: Path) -> tuple[Path, Path]:
    """
    Generate SYNTHETIC schema-compliant sample files that mimic the Jammertest-2024
    column structure for pipeline verification ONLY.

    WARNING: This output is NOT real data. It uses a hardcoded step-function
    attack (AGC 8000→4200, phase quadratic drift, labels at t≥1000). Any metrics
    computed from this data are artefacts of the simulation, NOT evidence of
    detector performance on real spoofing events. Output is written to
    data/synthetic/schema_samples/, never to data/processed/.
    """
    # Write to synthetic/, never to processed/
    synthetic_dir = raw_dir.parent / "synthetic" / "schema_samples" / "jammertest"
    synthetic_dir.mkdir(parents=True, exist_ok=True)

    ubx_csv = synthetic_dir / "SYNTHETIC_DO_NOT_EVALUATE_mon_rf.csv"
    rinex_csv = synthetic_dir / "SYNTHETIC_DO_NOT_EVALUATE_rinex.csv"

    n = 2000
    itow_base = 345600000  # ms into GPS week (mid-week)
    itow = itow_base + np.arange(n) * 1000  # 1 Hz

    # Pre-attack: AGC nominal ~8000 count (quiet RF). Attack at t=1000: AGC drops to ~4200 (in-band attenuation)
    agc = np.full(n, 8000, dtype=float) + np.random.normal(0, 15, size=n)
    agc[1000:] = 4200 + np.random.normal(0, 30, size=1000)
    agc = np.clip(agc, 0, 8191)

    df_rf = pd.DataFrame({
        "iTOW": itow,
        "agcCnt": agc.astype(int),
        "jamInd": np.where(np.arange(n) >= 1000, 180, 5),
    })
    df_rf.to_csv(ubx_csv, index=False)

    # Carrier phase: slow drift + noise; during spoofing (t=1000), spoofer injects quadratic phase acceleration
    phase = 10000.0 + 0.1 * np.arange(n) + np.random.normal(0, 0.05, size=n)
    phase[1000:] += 0.005 * (np.arange(1000) ** 1.8)

    # C/N0: ~44 dB-Hz nominal, fluctuating
    cno = np.full(n, 44.0) + np.random.normal(0, 0.4, size=n)
    cno[1000:] += 3.5  # Spoofer higher power

    df_rx = pd.DataFrame({
        "iTOW": itow,
        "L1C": phase,
        "S1C": cno,
        "LLI": np.where(np.arange(n) == 1000, 1, 0),  # Cycle slip flag at takeover
    })
    df_rx.to_csv(rinex_csv, index=False)

    return ubx_csv, rinex_csv


def generate_sample_yunnan_data(raw_dir: Path) -> tuple[Path, Path]:
    """Generate SYNTHETIC Yunnan-2024 schema-compliant sample tables.

    WARNING: This output is NOT real data. Output is written to
    data/synthetic/schema_samples/, never to data/processed/.
    """
    synthetic_dir = raw_dir.parent / "synthetic" / "schema_samples" / "yunnan"
    synthetic_dir.mkdir(parents=True, exist_ok=True)

    rawx_csv = synthetic_dir / "SYNTHETIC_DO_NOT_EVALUATE_rawx.csv"
    navsat_csv = synthetic_dir / "SYNTHETIC_DO_NOT_EVALUATE_navsat.csv"

    n = 1800
    tow = np.arange(n, dtype=float)
    # RAWX: cpMes carrier phase
    phase = 50000.0 + 0.15 * tow + np.random.normal(0, 0.04, size=n)
    phase[900:] += 0.008 * (np.arange(900) ** 1.7)

    df_rawx = pd.DataFrame({
        "rcvTow": tow,
        "cpMes": phase,
    })
    df_rawx.to_csv(rawx_csv, index=False)

    # NAV-SAT: cno in dB-Hz
    cno = np.full(n, 41.5) + np.random.normal(0, 0.5, size=n)
    cno[900:] += 4.0

    df_navsat = pd.DataFrame({
        "iTOW": (tow * 1000).astype(int),
        "cno": cno,
    })
    df_navsat.to_csv(navsat_csv, index=False)

    return rawx_csv, navsat_csv


def ingest_jammertest(raw_dir: Path, out_dir: Path, generate_samples: bool, log) -> list[Path]:
    """Ingest Jammertest sessions into standardized processed CSVs.

    If no real data is found and generate_samples is False (the default),
    exits with an error explaining how to download the real dataset.
    If generate_samples is True, writes schema-verification files to
    data/synthetic/schema_samples/ — NEVER to data/processed/.
    """
    jt_dir = raw_dir / "jammertest"
    ubx_files = list(jt_dir.glob("*.ubx")) + list(jt_dir.glob("*mon_rf*.csv"))
    rinex_files = list(jt_dir.glob("*.rnx")) + list(jt_dir.glob("*rinex*.csv"))
    # Exclude synthetic files that may have been placed under raw/ by mistake
    ubx_files = [f for f in ubx_files if "SYNTHETIC" not in f.name]
    rinex_files = [f for f in rinex_files if "SYNTHETIC" not in f.name]

    if not ubx_files or not rinex_files:
        if generate_samples:
            log.warning(
                "[SYNTHETIC] No real Jammertest data found. Generating SCHEMA-ONLY sample "
                "files for pipeline verification. These files are NOT real attack recordings "
                "and MUST NOT be used for detector evaluation or patent evidence. "
                "Download real data from: https://zenodo.org/records/15910563 "
                "(DOI: 10.5281/zenodo.15910563)"
            )
            u_csv, r_csv = generate_sample_jammertest_data(raw_dir)
            log.warning("[SYNTHETIC] Schema sample written to: %s", u_csv.parent)
            # Return empty — do NOT write synthetic data to processed/
            return []
        else:
            log.error(
                "No Jammertest-2024 raw files found in %s. "
                "Download the dataset from https://zenodo.org/records/15910563 "
                "(DOI: 10.5281/zenodo.15910563) and place UBX/RINEX files there. "
                "Use --generate-schema-samples to generate schema-verification files only "
                "(these are NOT real data and must not be used for evaluation).",
                jt_dir,
            )
            return []

    processed_paths = []
    u_file = ubx_files[0]
    r_file = rinex_files[0]
    log.info("Parsing Jammertest session: UBX=%s, RINEX=%s", u_file.name, r_file.name)

    if u_file.suffix == ".csv":
        # Pre-exported CSV mode
        df_u = pd.read_csv(u_file)
        df_r = pd.read_csv(r_file)
        time_s = (df_u["iTOW"] / 1000.0) % 604800.0
        df = pd.DataFrame({
            "time_s": time_s.values,
            "carrier_phase": df_r["L1C"].values if "L1C" in df_r.columns else np.nan,
            "hardware_agc": df_u["agcCnt"].values.astype(float) if "agcCnt" in df_u.columns else np.nan,
            "corr_magnitude": np.nan,
            "cn0_db_hz": df_r["S1C"].values.astype(float) if "S1C" in df_r.columns else np.nan,
            "attack_label": np.where(time_s - time_s.iloc[0] >= 1000.0, 2, 0),  # 2=spoofing
            "cycle_slip_flag": df_r["LLI"].values if "LLI" in df_r.columns else 0,
        })
    else:
        df = parse_jammertest_session(ubx_path=u_file, rinex_obs_path=r_file, attack_label=2)

    out_path = out_dir / "jammertest_session_spoofing.csv"
    df.to_csv(out_path, index=False)
    log.info("Saved processed Jammertest data: %s (%d rows)", out_path, len(df))
    processed_paths.append(out_path)

    return processed_paths


def ingest_yunnan(raw_dir: Path, out_dir: Path, generate_samples: bool, log) -> list[Path]:
    """Ingest Yunnan-2024 UBX table sessions."""
    yn_dir = raw_dir / "yunnan"
    rawx_files = list(yn_dir.glob("*rawx*.csv"))
    navsat_files = list(yn_dir.glob("*navsat*.csv"))

    if (not rawx_files or not navsat_files) and generate_samples:
        log.info("Generating representative Yunnan raw sample tables...")
        rx_csv, ns_csv = generate_sample_yunnan_data(raw_dir)
        rawx_files = [rx_csv]
        navsat_files = [ns_csv]

    processed_paths = []
    if rawx_files and navsat_files:
        df = parse_yunnan_session(
            rawx_csv_path=rawx_files[0],
            navsat_csv_path=navsat_files[0],
            attack_label=2,
        )
        # Apply attack onset timing (clean baseline first 900s)
        t_start = df["time_s"].iloc[0]
        df["attack_label"] = np.where(df["time_s"] - t_start >= 900.0, 2, 0)

        out_path = out_dir / "yunnan_session_spoofing.csv"
        df.to_csv(out_path, index=False)
        log.info("Saved processed Yunnan data: %s (%d rows)", out_path, len(df))
        processed_paths.append(out_path)

    return processed_paths


@click.command()
@click.option("--config", default="config.yaml", show_default=True, help="Path to config.yaml")
@click.option(
    "--dataset",
    default="all",
    type=click.Choice(["all", "jammertest", "yunnan", "fgi"]),
    show_default=True,
    help="Target dataset to ingest",
)
@click.option(
    "--generate-schema-samples",
    "generate_samples",  # internal name kept for backward compat
    is_flag=True,
    default=False,
    help=(
        "[SCHEMA VERIFICATION ONLY] Generate synthetic files that mimic the column "
        "structure of real datasets. Output is written to data/synthetic/schema_samples/ "
        "and is NEVER used as evaluation data. Real data must be downloaded separately "
        "(Jammertest-2024: https://zenodo.org/records/15910563)."
    ),
)
def main(config: str, dataset: str, generate_samples: bool) -> None:
    """Ingest and align raw GNSS datasets into data/processed/."""
    cfg = load_config(config)
    log = setup_logging(level=cfg["logging"]["level"], log_file=cfg["logging"]["file"])
    log.info("=== CKEC-GNSS Raw Data Ingestion Pipeline ===")

    raw_dir = Path(cfg["paths"]["data_raw"])
    out_dir = Path(cfg["paths"]["data_processed"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_processed: list[Path] = []

    if dataset in ("all", "jammertest"):
        p = ingest_jammertest(raw_dir, out_dir, generate_samples, log)
        all_processed.extend(p)

    if dataset in ("all", "yunnan"):
        p = ingest_yunnan(raw_dir, out_dir, generate_samples, log)
        all_processed.extend(p)

    log.info("Ingestion complete. Total processed sessions created: %d", len(all_processed))
    for p in all_processed:
        df = pd.read_csv(p)
        print(f"\n--- Ingested Dataset Summary: {p.name} ---")
        print(f"  Rows: {len(df)}")
        print(f"  Duration: {df['time_s'].iloc[-1] - df['time_s'].iloc[0]:.1f} s")
        print(f"  Hardware AGC non-null: {df['hardware_agc'].notna().sum()} / {len(df)}")
        print(f"  Carrier Phase non-null: {df['carrier_phase'].notna().sum()} / {len(df)}")
        print(f"  C/N0 non-null: {df['cn0_db_hz'].notna().sum()} / {len(df)}")
        print(f"  Attack Samples: {(df['attack_label'] != 0).sum()} (Onset at t={df.loc[df['attack_label'] != 0, 'time_s'].iloc[0] if (df['attack_label'] != 0).any() else 'N/A'})")


if __name__ == "__main__":
    main()
