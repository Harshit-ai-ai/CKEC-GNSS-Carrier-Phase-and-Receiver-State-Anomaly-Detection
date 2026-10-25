"""
ckec.ingestion.parsers
======================
Real-data parsers for Jammertest-2024 and FGI-SpoofRepo.

These parsers produce a uniform DataFrame schema for downstream feature extraction:

    time_s           float64  — seconds from GPS epoch aligned across streams
    carrier_phase    float64  — unwrapped carrier phase (cycles) [L1C by default]
    hardware_agc     float64  — UBX-MON-RF agcCnt (0-8191), NaN if unavailable
    corr_magnitude   float64  — sqrt(I_P^2+Q_P^2) prompt amplitude, NaN if unavailable
    cn0_db_hz        float64  — carrier-to-noise density (dB-Hz), NaN if unavailable
    attack_label     int      — 0=benign 1=spoofing (2=jamming 3=meaconing where applicable)
    cycle_slip_flag  int      — 1 if RINEX LLI != 0, else 0

Design notes
------------
* None of these parsers download data — they accept local file paths.
* If a file is not present, parsers return an empty DataFrame and log a warning.
* Units and encoding are documented inline so future readers never need to guess.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GPS_WEEK_SECONDS = 604800.0   # seconds per GPS week

# Nominal RINEX phase observables to try in priority order
_PHASE_CANDIDATES = ["L1C", "L1X", "L1W", "L1P", "L5I", "L5Q", "L5X"]
_CN0_CANDIDATES   = ["S1C", "S1X", "S1W", "S5I", "S5Q", "S5X"]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _itow_to_gpst(itow_ms: np.ndarray) -> np.ndarray:
    """
    Convert u-blox iTOW (ms from GPS week rollover) to continuous GPS seconds.
    
    Returns values in [0, GPS_WEEK_SECONDS). Rollover detection is not applied
    because a single session never spans a full week.
    """
    return itow_ms / 1000.0


def _rinex_epoch_to_gpst(obs_df: pd.DataFrame) -> pd.Series:
    """
    Extract time-of-week (seconds) from a georinex observation DataFrame index.
    The index may be DatetimeIndex (UTC) or MultiIndex including time.
    Returns a float64 Series of GPS seconds within the week.
    """
    if isinstance(obs_df.index, pd.DatetimeIndex):
        # RINEX datetimes are UTC; UBX iTOW is GPS time-of-week.
        gps_epoch = pd.Timestamp("1980-01-06 00:00:00", tz="UTC")
        t_utc = (
            obs_df.index.tz_localize("UTC")
            if obs_df.index.tz is None
            else obs_df.index.tz_convert("UTC")
        )

        # GPS - UTC was 18 seconds during the September 2024 recordings.
        elapsed_utc = (t_utc - gps_epoch).total_seconds()
        gps_seconds = elapsed_utc + 18.0

        return pd.Series(
            gps_seconds % GPS_WEEK_SECONDS,
            index=obs_df.index,
            name="time_s",
        )
    else:
        raise TypeError(
            "Cannot convert obs_df index to GPS time. "
            f"Expected DatetimeIndex, got {type(obs_df.index)}."
        )


# ---------------------------------------------------------------------------
# Parser 1 — Jammertest-2024 (UBX + RINEX)
# ---------------------------------------------------------------------------

def parse_jammertest_session(
    ubx_path: str | Path | None = None,
    rinex_obs_path: str | Path | None = None,
    attack_label: int = 0,
    target_sv: str = "G01",
    alignment_tol_s: float = 0.05,
) -> pd.DataFrame:
    """
    Parse one Jammertest-2024 session into the unified DataFrame schema.

    Parameters
    ----------
    ubx_path        : path to raw .ubx file containing UBX-MON-RF messages.
                      If None or not found, hardware_agc column will be all NaN.
    rinex_obs_path  : path to RINEX 3.04 observation file (.rnx / .obs).
                      If None or not found, carrier_phase and cn0 will be all NaN.
    attack_label    : integer label for this session (0=nominal, 1=jamming,
                      2=spoofing, 3=meaconing).
    target_sv       : satellite vehicle identifier to extract from RINEX
                      (e.g. 'G01' for GPS PRN 1).
    alignment_tol_s : maximum tolerated time gap (seconds) for nearest-epoch
                      alignment between UBX and RINEX streams.

    Returns
    -------
    DataFrame with unified schema. Columns that could not be populated are NaN.

    Notes on physical measurements
    --------------------------------
    hardware_agc  : UBX-MON-RF.agcCnt, a 12-bit integer (0-8191). 8191 = 100%
                    gain (quiet environment). Decreases as front-end attenuates
                    strong in-band RF. This is TRUE hardware AGC.
    carrier_phase : RINEX L1C in full cycles (no ambiguity resolution applied).
                    LLI flag (loss-of-lock indicator) is preserved separately.
    cn0_db_hz     : RINEX S1C in dB-Hz. Carrier-to-noise density ratio.
                    NOT hardware AGC. Treated as a separate secondary modality.
    """
    records_agc: list[dict] = []
    records_rinex: pd.DataFrame | None = None

    # ---- UBX stream parsing ----
    if ubx_path is not None:
        ubx_path = Path(ubx_path)
        if not ubx_path.exists():
            log.warning("UBX file not found: %s — hardware_agc will be NaN.", ubx_path)
        else:
            try:
                import pyubx2  # type: ignore

                with open(ubx_path, "rb") as f:
                    reader = pyubx2.UBXReader(f, protfilter=pyubx2.UBX_PROTOCOL)
                    for raw, msg in reader:
                        if msg.identity == "MON-RF":
                            itow = getattr(msg, "iTOW", None)
                            if itow is None:
                                continue
                            # agcCnt may appear as agcCnt_01 etc. for multi-band
                            agc = getattr(msg, "agcCnt_01", getattr(msg, "agcCnt", None))
                            if agc is not None:
                                records_agc.append({
                                    "time_s": _itow_to_gpst(np.array([itow]))[0],
                                    "hardware_agc": float(agc),
                                })
                log.info("Parsed %d UBX-MON-RF records from %s", len(records_agc), ubx_path)
            except ImportError:
                log.error("pyubx2 not installed. Run: pip install pyubx2")
            except Exception as exc:
                log.error("Error parsing UBX file %s: %s", ubx_path, exc)

    # ---- RINEX observation parsing ----
    if rinex_obs_path is not None:
        rinex_obs_path = Path(rinex_obs_path)
        if not rinex_obs_path.exists():
            log.warning("RINEX obs file not found: %s — phase/cn0 will be NaN.", rinex_obs_path)
        else:
            try:
                import georinex as gr  # type: ignore

                obs = gr.load(str(rinex_obs_path), use="G")   # GPS only for now
                obs_df = obs.to_dataframe()
                # Filter to target satellite
                if "sv" in obs_df.index.names:
                    obs_df = obs_df.xs(target_sv, level="sv") if target_sv in obs_df.index.get_level_values("sv") else obs_df
                obs_df.index = pd.to_datetime(obs_df.index.get_level_values(0) if obs_df.index.nlevels > 1 else obs_df.index)

                time_s = _rinex_epoch_to_gpst(obs_df)

                # Pick first available phase and S/N observable
                phase_col = next((c for c in _PHASE_CANDIDATES if c in obs_df.columns), None)
                cn0_col   = next((c for c in _CN0_CANDIDATES   if c in obs_df.columns), None)
                lli_col   = f"{phase_col[:2]}LLI" if phase_col else None

                records_rinex = pd.DataFrame({"time_s": time_s.values})
                records_rinex["carrier_phase"]  = obs_df[phase_col].values if phase_col else np.nan
                records_rinex["cn0_db_hz"]       = obs_df[cn0_col].values   if cn0_col   else np.nan
                records_rinex["cycle_slip_flag"] = (
                    (obs_df[lli_col] != 0).astype(int).values if lli_col and lli_col in obs_df.columns
                    else 0
                )
                log.info(
                    "Parsed RINEX: %d epochs, phase=%s, cn0=%s from %s",
                    len(records_rinex), phase_col, cn0_col, rinex_obs_path,
                )
            except ImportError:
                log.error("georinex not installed. Run: pip install georinex")
            except Exception as exc:
                log.error("Error parsing RINEX file %s: %s", rinex_obs_path, exc)

    # ---- Merge UBX and RINEX on nearest epoch ----
    df_agc = pd.DataFrame(records_agc) if records_agc else pd.DataFrame(columns=["time_s", "hardware_agc"])
    df_rinex = records_rinex if records_rinex is not None else pd.DataFrame(columns=["time_s", "carrier_phase", "cn0_db_hz", "cycle_slip_flag"])

    if df_rinex.empty and df_agc.empty:
        log.warning("Both UBX and RINEX sources empty — returning empty DataFrame.")
        return pd.DataFrame(columns=["time_s", "carrier_phase", "hardware_agc", "corr_magnitude", "cn0_db_hz", "attack_label", "cycle_slip_flag"])

    # Use RINEX epochs as reference grid if available, else AGC epochs
    if not df_rinex.empty:
        base = df_rinex.set_index("time_s")
        if not df_agc.empty:
            agc_indexed = df_agc.set_index("time_s").reindex(base.index, method="nearest", tolerance=alignment_tol_s)
            base = base.join(agc_indexed, rsuffix="_agc")
        result = base.reset_index()
    else:
        result = df_agc.copy()
        result["carrier_phase"]  = np.nan
        result["cn0_db_hz"]      = np.nan
        result["cycle_slip_flag"] = 0

    result["corr_magnitude"] = np.nan           # not available in Jammertest UBX
    result["attack_label"]   = attack_label

    # Ensure column order
    for col in ["hardware_agc", "corr_magnitude", "cn0_db_hz", "carrier_phase", "cycle_slip_flag"]:
        if col not in result.columns:
            result[col] = np.nan

    return result[[
        "time_s", "carrier_phase", "hardware_agc",
        "corr_magnitude", "cn0_db_hz", "attack_label", "cycle_slip_flag",
    ]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Parser 2 — FGI-SpoofRepo (RINEX only; correlator magnitude from MAT)
# ---------------------------------------------------------------------------

def parse_fgi_session(
    rinex_obs_path: str | Path | None = None,
    mat_trackresults_path: str | Path | None = None,
    attack_label: int = 0,
    benign_duration_s: float = 130.0,
    total_duration_s: float = 370.0,
    target_sv: str = "G01",
) -> pd.DataFrame:
    """
    Parse one FGI-SpoofRepo session into the unified DataFrame schema.

    Parameters
    ----------
    rinex_obs_path        : RINEX obs file (.rnx). Provides carrier_phase and cn0.
    mat_trackresults_path : MATLAB .mat file from FGI-GSRx containing trackResults
                           struct. Provides corr_magnitude = sqrt(I_P^2 + Q_P^2).
    attack_label          : 1 if this is a spoofing recording, 0 for benign.
    benign_duration_s     : documented clean-sky duration before attack onset.
                           Per FGI documentation: ~130s for SFMC/DFMC scenarios.
                           NOT universally 60s — do not assume across scenarios.
    total_duration_s      : total recording duration in seconds.
    target_sv             : satellite to extract from RINEX.

    Notes on physical measurements
    --------------------------------
    corr_magnitude : sqrt(I_P^2 + Q_P^2) from FGI-GSRx prompt correlator output.
                     This is post-correlation despread signal amplitude.
                     It is NOT hardware AGC. It reflects tracking loop state.
    hardware_agc   : NOT available in FGI-SpoofRepo (always NaN).
    """
    result_parts = []

    # ---- RINEX carrier phase ----
    if rinex_obs_path is not None:
        rinex_obs_path = Path(rinex_obs_path)
        if rinex_obs_path.exists():
            try:
                import georinex as gr  # type: ignore

                obs = gr.load(str(rinex_obs_path), use="G")
                obs_df = obs.to_dataframe()
                if "sv" in obs_df.index.names:
                    obs_df = obs_df.xs(target_sv, level="sv") if target_sv in obs_df.index.get_level_values("sv") else obs_df
                obs_df.index = pd.to_datetime(obs_df.index.get_level_values(0) if obs_df.index.nlevels > 1 else obs_df.index)

                time_s = _rinex_epoch_to_gpst(obs_df)
                phase_col = next((c for c in _PHASE_CANDIDATES if c in obs_df.columns), None)
                cn0_col   = next((c for c in _CN0_CANDIDATES   if c in obs_df.columns), None)

                part = pd.DataFrame({
                    "time_s":        time_s.values,
                    "carrier_phase": obs_df[phase_col].values if phase_col else np.nan,
                    "cn0_db_hz":     obs_df[cn0_col].values   if cn0_col   else np.nan,
                })
                result_parts.append(part)
                log.info("FGI RINEX: %d epochs from %s", len(part), rinex_obs_path)
            except Exception as exc:
                log.error("Error parsing FGI RINEX %s: %s", rinex_obs_path, exc)
        else:
            log.warning("FGI RINEX not found: %s", rinex_obs_path)

    # ---- MAT trackResults (correlator magnitude) ----
    corr_mag_series: pd.Series | None = None
    if mat_trackresults_path is not None:
        mat_path = Path(mat_trackresults_path)
        if mat_path.exists():
            try:
                from scipy.io import loadmat  # type: ignore

                mat = loadmat(str(mat_path))
                tr = mat.get("trackResults", mat.get("trackResult", None))
                if tr is not None:
                    # Navigate nested struct: trackResults.I_P and Q_P
                    ip = np.array(tr["I_P"].flat[0]).ravel().astype(float)
                    qp = np.array(tr["Q_P"].flat[0]).ravel().astype(float)
                    corr_mag = np.sqrt(ip ** 2 + qp ** 2)
                    corr_mag_series = pd.Series(corr_mag, name="corr_magnitude")
                    log.info("FGI MAT: %d correlator samples from %s", len(corr_mag_series), mat_path)
            except Exception as exc:
                log.error("Error parsing FGI MAT %s: %s", mat_path, exc)
        else:
            log.warning("FGI MAT trackResults not found: %s", mat_path)

    # ---- Assemble output ----
    if result_parts:
        df = result_parts[0]
    else:
        log.warning("FGI: No RINEX data — returning empty DataFrame.")
        return pd.DataFrame(columns=["time_s", "carrier_phase", "hardware_agc",
                                     "corr_magnitude", "cn0_db_hz", "attack_label", "cycle_slip_flag"])

    # Assign correlator magnitude if available and lengths match
    if corr_mag_series is not None:
        if len(corr_mag_series) == len(df):
            df["corr_magnitude"] = corr_mag_series.values
        else:
            log.warning(
                "FGI: Correlator magnitude length (%d) != RINEX epoch count (%d). "
                "Decimation/resampling needed before assignment.",
                len(corr_mag_series), len(df),
            )
            df["corr_magnitude"] = np.nan
    else:
        df["corr_magnitude"] = np.nan

    df["hardware_agc"]   = np.nan   # explicitly absent in FGI-SpoofRepo
    df["cycle_slip_flag"] = 0
    df["attack_label"]   = 0  # will be overwritten below

    # Assign per-epoch labels based on documented onset
    # Per FGI documentation: clean baseline is approx benign_duration_s seconds
    t_min = df["time_s"].min()
    onset_t = t_min + benign_duration_s
    df["attack_label"] = np.where(
        df["time_s"] >= onset_t, attack_label, 0
    ).astype(int)
    log.info(
        "FGI: attack_label assigned; onset at t_rel=%.1fs "
        "(benign_duration_s=%.1f, NOT assumed fixed 60s).",
        benign_duration_s, benign_duration_s,
    )

    return df[[
        "time_s", "carrier_phase", "hardware_agc",
        "corr_magnitude", "cn0_db_hz", "attack_label", "cycle_slip_flag",
    ]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Parser 3 — Yunnan-2024 (UBX tables)
# ---------------------------------------------------------------------------

def parse_yunnan_session(
    rawx_csv_path: str | Path | None = None,
    navsat_csv_path: str | Path | None = None,
    attack_label: int = 0,
    time_col: str = "rcvTow",
) -> pd.DataFrame:
    """
    Parse Yunnan-2024 session from pre-exported UBX table CSVs.

    Parameters
    ----------
    rawx_csv_path  : CSV exported from UBX-RXM-RAWX messages.
                    Expected columns: rcvTow, cpMes (carrier phase in cycles).
    navsat_csv_path: CSV exported from UBX-NAV-SAT messages.
                    Expected columns: iTOW, cno (C/N0 in dB-Hz).
    attack_label   : 0=clean, 1=jammed, 2=spoofed (per Yunnan paper taxonomy).
    time_col       : column name for time in rawx_csv_path.

    Notes on physical measurements
    --------------------------------
    carrier_phase : UBX-RXM-RAWX.cpMes — carrier phase in cycles.
                   cpMes is the half-cycle-ambiguous accumulated phase.
    cn0_db_hz     : UBX-NAV-SAT.cno — C/N0 in dB-Hz. NOT hardware AGC.
    hardware_agc  : NOT available in Yunnan-2024 (always NaN).
    corr_magnitude: NOT available in Yunnan-2024 (always NaN).
    """
    df_rawx: pd.DataFrame = pd.DataFrame()
    df_navsat: pd.DataFrame = pd.DataFrame()

    if rawx_csv_path is not None:
        p = Path(rawx_csv_path)
        if p.exists():
            df_rawx = pd.read_csv(p)
            log.info("Yunnan RAWX: %d rows from %s", len(df_rawx), p)
        else:
            log.warning("Yunnan RAWX CSV not found: %s", p)

    if navsat_csv_path is not None:
        p = Path(navsat_csv_path)
        if p.exists():
            df_navsat = pd.read_csv(p)
            log.info("Yunnan NAVSAT: %d rows from %s", len(df_navsat), p)
        else:
            log.warning("Yunnan NAVSAT CSV not found: %s", p)

    if df_rawx.empty and df_navsat.empty:
        log.warning("Yunnan: Both CSVs empty/missing — returning empty DataFrame.")
        return pd.DataFrame(columns=["time_s", "carrier_phase", "hardware_agc",
                                     "corr_magnitude", "cn0_db_hz", "attack_label", "cycle_slip_flag"])

    result = pd.DataFrame()
    if not df_rawx.empty and time_col in df_rawx.columns:
        result["time_s"]        = df_rawx[time_col].astype(float)
        result["carrier_phase"] = df_rawx["cpMes"].astype(float) if "cpMes" in df_rawx.columns else np.nan
    elif not df_navsat.empty and "iTOW" in df_navsat.columns:
        result["time_s"]        = (df_navsat["iTOW"] / 1000.0).astype(float)
        result["carrier_phase"] = np.nan

    if not df_navsat.empty and "cno" in df_navsat.columns and not result.empty:
        # Align navsat on nearest time
        navsat_t = (df_navsat["iTOW"] / 1000.0).values if "iTOW" in df_navsat.columns else np.arange(len(df_navsat))
        result["cn0_db_hz"] = np.interp(
            result["time_s"].values, navsat_t, df_navsat["cno"].values,
            left=np.nan, right=np.nan,
        )
    else:
        result["cn0_db_hz"] = np.nan

    result["hardware_agc"]    = np.nan   # explicitly absent
    result["corr_magnitude"]  = np.nan   # explicitly absent
    result["cycle_slip_flag"] = 0
    result["attack_label"]    = attack_label

    return result[[
        "time_s", "carrier_phase", "hardware_agc",
        "corr_magnitude", "cn0_db_hz", "attack_label", "cycle_slip_flag",
    ]].reset_index(drop=True)
