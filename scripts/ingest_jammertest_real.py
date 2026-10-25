"""
scripts/ingest_jammertest_real.py
=================================
Ingest the REAL Jammertest-2024 dataset from a downloaded archive.

This script discovers all scenario folders, parses mon_rf.csv and rinex.csv,
aligns them by timestamp, labels samples from scenario.json (with Norway CEST->GPS
time correction), and writes one CSV per scenario into data/processed/real/.

Usage:
    python scripts/ingest_jammertest_real.py --dataset-root /path/to/GNSS_DATASET
    python scripts/ingest_jammertest_real.py --dataset-root /path/to/dataset --attack-types Spoofing Meaconing
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from phase_quality import flag_phase_discontinuities

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import click
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Scenario Discovery & Attack Timing
# ---------------------------------------------------------------------------

ATTACK_LABEL_MAP = {
    "Jamming": 1,
    "Spoofing": 2,
    "Meaconing": 3,
    "Spoofing + Jamming": 4,
    "Jamming+Spoofing": 4,
}


def extract_attack_periods(meta: dict) -> list[tuple[datetime, datetime]]:
    """Extract all attack periods with the Jammertest-2024 CEST->GPS time adjustment.

    Jammertest scenario.json records attack log timestamps in local Norwegian
    time (CEST = UTC+2) tagged with 'Z'. Meanwhile, the u-blox receiver and RINEX
    recordings are logged in GPS time (GPS = UTC + 18s leap seconds).
    Therefore:
        GPS time = Local Norwegian Time - (2 hours - 18 seconds)
    """
    time_adjustment = timedelta(hours=2, seconds=-18)
    events = meta.get("attack_log", [])
    events = sorted(events, key=lambda x: x.get("timestamp_utc", ""))
    attack_periods = []
    i = 0
    while i < len(events) - 1:
        e1, e2 = events[i], events[i + 1]
        if "start" in e1.get("event", "").lower() and "end" in e2.get("event", "").lower():
            try:
                s_str = e1["timestamp_utc"].replace("Z", "")
                e_str = e2["timestamp_utc"].replace("Z", "")
                start = datetime.fromisoformat(s_str) - time_adjustment
                end = datetime.fromisoformat(e_str) - time_adjustment
                attack_periods.append((start, end))
                i += 2
            except (ValueError, AttributeError):
                i += 1
        else:
            i += 1

    # Fallback if only single start event is listed
    if not attack_periods and events:
        for ev in events:
            if "start" in ev.get("event", "").lower():
                try:
                    s_str = ev["timestamp_utc"].replace("Z", "")
                    start = datetime.fromisoformat(s_str) - time_adjustment
                    end = start + timedelta(minutes=40)
                    attack_periods.append((start, end))
                    break
                except (ValueError, AttributeError):
                    pass

    return attack_periods


def discover_scenarios(dataset_root: Path, attack_types: list[str] | None = None) -> list[dict]:
    """Discover all scenarios by finding scenario.json files recursively."""
    scenarios = []
    for json_path in sorted(dataset_root.rglob("scenario.json")):
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        attack_type = meta.get("attack_type", "Unknown")
        if attack_types and attack_type not in attack_types:
            continue

        scenario_dir = json_path.parent
        mon_rf = scenario_dir / "mon_rf.csv"
        rinex = scenario_dir / "rinex.csv"

        if not mon_rf.exists():
            print(f"  [SKIP] {scenario_dir.name}: mon_rf.csv missing")
            continue
        if not rinex.exists():
            print(f"  [SKIP] {scenario_dir.name}: rinex.csv missing")
            continue

        # Rover mobility
        path_str = str(json_path).lower()
        if "dynamic" in path_str:
            mobility = "dynamic"
        elif "stationary" in path_str:
            mobility = "stationary"
        else:
            mobility = meta.get("attack_parameters", {}).get("Rover states", "unknown").lower()

        attack_periods = extract_attack_periods(meta)

        scenarios.append({
            "scenario_id": meta.get("scenario_id", scenario_dir.name),
            "mobility": mobility,
            "attack_type": attack_type,
            "attack_label": ATTACK_LABEL_MAP.get(attack_type, 2),
            "attack_periods": attack_periods,
            "scenario_dir": scenario_dir,
            "mon_rf_path": mon_rf,
            "rinex_path": rinex,
            "meta": meta,
        })

    return scenarios


# ---------------------------------------------------------------------------
# Mon-RF Parsing (per-epoch hardware AGC from UBX-MON-RF)
# ---------------------------------------------------------------------------

def parse_mon_rf(path: Path) -> pd.DataFrame:
    """Parse mon_rf.csv to extract per-epoch AGC and jamming indicators."""
    df = pd.read_csv(path)

    out = pd.DataFrame()
    out["timestamp"] = pd.to_datetime(df["real_time"])
    out["iTOW"] = df["matched_iTOW"]

    # RF block 0 = L1, block 1 = L2/L5
    out["agcCnt_L1"] = df["agcCnt_01"].astype(float)
    out["agcCnt_L2L5"] = df["agcCnt_02"].astype(float)
    out["jamInd_L1"] = df["jamInd_01"].astype(float)
    out["jamInd_L2L5"] = df["jamInd_02"].astype(float)

    if "noisePerMS_01" in df.columns:
        out["noisePerMS_L1"] = df["noisePerMS_01"].astype(float)
        out["noisePerMS_L2L5"] = df["noisePerMS_02"].astype(float)

    out = out.drop_duplicates(subset=["iTOW"], keep="first").sort_values("timestamp").reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# RINEX Parsing (multi-satellite -> per-epoch aggregation)
# ---------------------------------------------------------------------------


def parse_rinex_to_epoch(
    path: Path,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Screen raw RINEX observations and aggregate them by epoch."""
    kwargs = {"nrows": max_rows} if max_rows else {}
    df = pd.read_csv(path, **kwargs)

    required = {"time", "satellite"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns: {sorted(missing)}")

    df["timestamp"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.loc[df["timestamp"].notna()].copy()

    # Screen numeric observations independently for each frequency band.
    for band in ("L1", "L2"):
        phase_col = f"carrier_phase_{band}"
        snr_col = f"snr_{band}"
        pr_col = f"pseudorange_{band}"

        for col in (phase_col, snr_col, pr_col, f"doppler_{band}"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if snr_col in df.columns:
            invalid = df[snr_col].notna() & ~df[snr_col].between(0, 100)
            if invalid.any():
                print(
                    f"[QUALITY] {path.name} {snr_col}: "
                    f"rejected {int(invalid.sum())} values outside [0, 100]"
                )
                df.loc[invalid, snr_col] = np.nan

        if pr_col in df.columns:
            invalid = df[pr_col].notna() & ~df[pr_col].between(1e6, 1e8)
            if invalid.any():
                print(
                    f"[QUALITY] {path.name} {pr_col}: "
                    f"rejected {int(invalid.sum())} values outside "
                    "[1e6, 1e8] metres"
                )
                df.loc[invalid, pr_col] = np.nan

        if phase_col in df.columns:
            phase = df[phase_col]
            invalid = phase.notna() & (
                ~np.isfinite(phase) | (phase.abs() > 1e9)
            )
            if invalid.any():
                print(
                    f"[QUALITY] {path.name} {phase_col}: "
                    f"rejected {int(invalid.sum())} invalid/extreme values"
                )
                df.loc[invalid, phase_col] = np.nan

            # The quality helper expects canonical column names.
            phase_input = pd.DataFrame({
                "time": df["timestamp"],
                "satellite": df["satellite"],
                "carrier_phase": df[phase_col],
            })

            flagged = flag_phase_discontinuities(
                phase_input,
                max_gap_seconds=2.0,
                max_phase_rate_cycles_per_second=100_000.0,
            )

            count = int(flagged.sum())
            if count:
                print(
                    f"[QUALITY] {path.name} {phase_col}: "
                    f"flagged {count} phase discontinuities"
                )
                df.loc[flagged.to_numpy(), phase_col] = np.nan

    records = []

    for ts, group in df.groupby("timestamp", sort=True):
        l1_data = (
            group[group["carrier_phase_L1"].notna()]
            if "carrier_phase_L1" in group.columns
            else group.iloc[0:0]
        )
        l2_data = (
            group[group["carrier_phase_L2"].notna()]
            if "carrier_phase_L2" in group.columns
            else group.iloc[0:0]
        )

        rec = {
            "timestamp": ts,
            "n_sats_L1": len(l1_data),
            "n_sats_L2": len(l2_data),
        }

        if not l1_data.empty:
            if (
                "snr_L1" in l1_data.columns
                and l1_data["snr_L1"].notna().any()
            ):
                ref_idx = l1_data["snr_L1"].idxmax()
            else:
                ref_idx = l1_data.index[0]

            rec["carrier_phase_L1_ref"] = l1_data.loc[
                ref_idx, "carrier_phase_L1"
            ]
            rec["ref_satellite"] = l1_data.loc[ref_idx, "satellite"]

            snr = l1_data["snr_L1"] if "snr_L1" in l1_data.columns else pd.Series(dtype=float)
            doppler = l1_data["doppler_L1"] if "doppler_L1" in l1_data.columns else pd.Series(dtype=float)

            rec["cn0_L1_mean"] = snr.mean()
            rec["cn0_L1_std"] = snr.std() if snr.notna().sum() > 1 else 0.0
            rec["doppler_L1_mean"] = doppler.mean()
            rec["doppler_L1_std"] = doppler.std() if doppler.notna().sum() > 1 else 0.0
        else:
            rec.update({
                "carrier_phase_L1_ref": np.nan,
                "ref_satellite": None,
                "cn0_L1_mean": np.nan,
                "cn0_L1_std": np.nan,
                "doppler_L1_mean": np.nan,
                "doppler_L1_std": np.nan,
            })

        if not l2_data.empty and "snr_L2" in l2_data.columns:
            rec["cn0_L2_mean"] = l2_data["snr_L2"].mean()
        else:
            rec["cn0_L2_mean"] = np.nan

        records.append(rec)

    return pd.DataFrame(records).sort_values(
        "timestamp"
    ).reset_index(drop=True)

# ---------------------------------------------------------------------------
# Alignment + Labelling
# ---------------------------------------------------------------------------

def apply_attack_labels(
    df: pd.DataFrame,
    attack_periods: list[tuple[datetime, datetime]],
    attack_label: int,
) -> pd.DataFrame:
    """Label samples as benign (0) or attack (attack_label) based on attack_periods."""
    ts_series = pd.to_datetime(df["timestamp"])
    if ts_series.dt.tz is not None:
        ts_series = ts_series.dt.tz_localize(None)

    is_attack = pd.Series(False, index=df.index)
    for start, end in attack_periods:
        s = start.replace(tzinfo=None) if hasattr(start, "tzinfo") and start.tzinfo else start
        e = end.replace(tzinfo=None) if hasattr(end, "tzinfo") and end.tzinfo else end
        is_attack |= (ts_series >= s) & (ts_series <= e)

    df["attack_label"] = np.where(is_attack, attack_label, 0)
    return df


def align_and_label(
    mon_rf: pd.DataFrame,
    rinex_epochs: pd.DataFrame,
    attack_periods: list[tuple[datetime, datetime]],
    attack_label: int,
) -> pd.DataFrame:
    """Align mon_rf and rinex data by nearest timestamp and apply attack labels."""
    mon_rf = mon_rf.sort_values("timestamp").reset_index(drop=True)
    rinex_epochs = rinex_epochs.sort_values("timestamp").reset_index(drop=True)

    merged = pd.merge_asof(
        mon_rf.sort_values("timestamp"),
        rinex_epochs.sort_values("timestamp"),
        on="timestamp",
        tolerance=pd.Timedelta("250ms"),
        direction="nearest",
    )

    t0 = merged["timestamp"].iloc[0]
    merged["time_s"] = (merged["timestamp"] - t0).dt.total_seconds()
    merged = apply_attack_labels(merged, attack_periods, attack_label)

    output = pd.DataFrame({
        "time_s": merged["time_s"],
        "timestamp": merged["timestamp"],
        "carrier_phase": merged.get("carrier_phase_L1_ref", np.nan),
        "ref_satellite": merged.get("ref_satellite", np.nan),
        "hardware_agc": merged["agcCnt_L1"],
        "hardware_agc_L2L5": merged["agcCnt_L2L5"],
        "cn0_db_hz": merged.get("cn0_L1_mean", np.nan),
        "cn0_L2_db_hz": merged.get("cn0_L2_mean", np.nan),
        "cn0_std": merged.get("cn0_L1_std", np.nan),
        "jamInd_L1": merged["jamInd_L1"],
        "jamInd_L2L5": merged["jamInd_L2L5"],
        "n_sats_L1": merged.get("n_sats_L1", 0),
        "n_sats_L2": merged.get("n_sats_L2", 0),
        "doppler_L1_mean": merged.get("doppler_L1_mean", np.nan),
        "attack_label": merged["attack_label"],
        "corr_magnitude": np.nan,
    })

    return output


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--dataset-root",
    required=True,
    type=click.Path(exists=True),
    help="Root directory of the extracted GNSS_DATASET_JAMMING_SPOOFING archive",
)
@click.option(
    "--output-dir",
    default=None,
    type=click.Path(),
    help="Output directory for processed CSVs (default: data/processed/real/)",
)
@click.option(
    "--attack-types",
    multiple=True,
    default=None,
    help="Filter by attack type(s): Spoofing, Meaconing, Jamming, 'Spoofing + Jamming'. Default: all.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Force re-parsing of rinex.csv even if target CSV already exists.",
)
def main(dataset_root: str, output_dir: str | None, attack_types: tuple[str, ...], force: bool) -> None:
    """Ingest REAL Jammertest-2024 scenarios into standardised processed CSVs."""
    root = Path(dataset_root)
    out_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parent.parent / "data" / "processed" / "real"
    out_dir.mkdir(parents=True, exist_ok=True)

    attack_filter = list(attack_types) if attack_types else None
    print(f"Discovering scenarios in: {root}")
    if attack_filter:
        print(f"Filtering to attack types: {attack_filter}")

    scenarios = discover_scenarios(root, attack_types=attack_filter)
    print(f"Found {len(scenarios)} scenarios with both mon_rf.csv and rinex.csv\n")

    if not scenarios:
        print("ERROR: No valid scenarios found. Check dataset path and structure.")
        return

    summary_records = []

    for i, sc in enumerate(scenarios):
        sid = sc["scenario_id"]
        atype = sc["attack_type"]
        mobility = sc["mobility"]
        periods = sc["attack_periods"]
        safe_id = sid.replace(",", "_").replace(" ", "_").replace("/", "_")
        fname = f"jammertest_real_{atype.lower().replace(' ', '_')}_{mobility}_{safe_id}.csv"
        out_path = out_dir / fname

        # Also check older filename format without mobility if it matches
        old_fname = f"jammertest_real_{atype.lower().replace(' ', '_')}_{safe_id}.csv"
        old_path = out_dir / old_fname

        print(f"[{i+1}/{len(scenarios)}] Processing scenario {sid} ({atype}, {mobility})...")
        print(f"  Attack periods ({len(periods)}): {periods}")

        try:
            # Check if we can fast-relabel from an already-processed CSV
            source_existing = None
            if not force:
                if out_path.exists():
                    source_existing = out_path
                elif old_path.exists() and mobility == "stationary":
                    source_existing = old_path

            if source_existing is not None:
                print(f"  Using existing parsed CSV: {source_existing.name} (updating labels)")
                df_existing = pd.read_csv(source_existing)
                aligned = apply_attack_labels(df_existing, periods, sc["attack_label"])
            else:
                # Full parse
                mon_rf_df = parse_mon_rf(sc["mon_rf_path"])
                print(f"  mon_rf: {len(mon_rf_df)} epochs, AGC_L1 range: "
                      f"{mon_rf_df['agcCnt_L1'].min():.0f}-{mon_rf_df['agcCnt_L1'].max():.0f}")

                print(f"  Parsing rinex.csv (~{sc['rinex_path'].stat().st_size / 1e6:.1f} MB)...")
                rinex_epochs = parse_rinex_to_epoch(sc["rinex_path"])
                print(f"  rinex: {len(rinex_epochs)} epochs, "
                      f"mean C/N0: {rinex_epochs['cn0_L1_mean'].mean():.1f} dB-Hz, "
                      f"sats/epoch: {rinex_epochs['n_sats_L1'].mean():.1f}")

                aligned = align_and_label(
                    mon_rf_df, rinex_epochs,
                    attack_periods=periods,
                    attack_label=sc["attack_label"],
                )

            n_benign = int((aligned["attack_label"] == 0).sum())
            n_attack = int((aligned["attack_label"] != 0).sum())
            print(f"  Aligned: {len(aligned)} epochs, benign={n_benign}, attack={n_attack}")

            # Save to standard path with mobility in name
            aligned.to_csv(out_path, index=False)
            print(f"  Saved: {out_path.name}\n")

            summary_records.append({
                "scenario_id": sid,
                "mobility": mobility,
                "attack_type": atype,
                "n_epochs": len(aligned),
                "n_benign": n_benign,
                "n_attack": n_attack,
                "agc_L1_min": aligned["hardware_agc"].min(),
                "agc_L1_max": aligned["hardware_agc"].max(),
                "agc_L2L5_min": aligned["hardware_agc_L2L5"].min(),
                "agc_L2L5_max": aligned["hardware_agc_L2L5"].max(),
                "mean_cn0": aligned["cn0_db_hz"].mean(),
                "output_file": out_path.name,
            })

        except Exception as e:
            print(f"  [ERROR] Failed to ingest {sid}: {e}\n")
            import traceback
            traceback.print_exc()

    # Write summary
    if summary_records:
        summary_df = pd.DataFrame(summary_records)
        summary_path = out_dir / "ingestion_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n{'='*60}")
        print(f"INGESTION COMPLETE: {len(summary_records)}/{len(scenarios)} scenarios processed")
        print(f"Output directory: {out_dir}")
        print(f"Summary: {summary_path}")
        print(f"\nTotal benign epochs: {summary_df['n_benign'].sum()}")
        print(f"Total attack epochs: {summary_df['n_attack'].sum()}")
        print(f"{'='*60}")
    else:
        print("No scenarios were successfully ingested.")


if __name__ == "__main__":
    main()
