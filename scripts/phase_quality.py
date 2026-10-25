"""Quality checks for GNSS carrier-phase observations."""

import numpy as np
import pandas as pd

def flag_phase_discontinuities(
    df: pd.DataFrame,
    *,
    time_col: str = "time",
    satellite_col: str = "satellite",
    phase_col: str = "carrier_phase",
    max_gap_seconds: float = 2.0,
    max_phase_rate_cycles_per_second: float = 100_000.0,
) -> pd.Series:
    """Flag phase jumps within each satellite track.

    Returns a boolean Series aligned to df.index. A sample is flagged
    when its phase difference from the preceding valid sample exceeds
    the configured phase-rate threshold and the time gap is positive
    and no greater than max_gap_seconds.

    This is a configurable quality heuristic, not an attack detector.
    Original observations are never modified.
    """
    if max_gap_seconds <= 0:
        raise ValueError("max_gap_seconds must be positive")
    if max_phase_rate_cycles_per_second <= 0:
        raise ValueError("max_phase_rate_cycles_per_second must be positive")

    required = {time_col, satellite_col, phase_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    times = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    phases = pd.to_numeric(df[phase_col], errors="coerce")
    satellites = df[satellite_col]

    work = pd.DataFrame({
        "_time": times,
        "_satellite": satellites,
        "_phase": phases,
        "_position": np.arange(len(df)),
    })

    flagged = np.zeros(len(df), dtype=bool)

    valid = (
        work["_time"].notna()
        & work["_satellite"].notna()
        & work["_phase"].notna()
        & np.isfinite(work["_phase"])
    )
    work = work.loc[valid].sort_values(
        ["_satellite", "_time", "_position"],
        kind="stable",
    )

    for _, track in work.groupby("_satellite", sort=False):
        dt = track["_time"].diff().dt.total_seconds()
        dphase = track["_phase"].diff().abs()

        suspicious = (
            dt.gt(0)
            & dt.le(max_gap_seconds)
            & (dphase / dt).gt(max_phase_rate_cycles_per_second)
        )

        positions = track.loc[suspicious, "_position"].to_numpy(dtype=int)
        flagged[positions] = True

    return pd.Series(flagged, index=df.index, name="phase_discontinuity")
