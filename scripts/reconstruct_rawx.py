#!/usr/bin/env python3
"""Recover checksum-valid u-blox RXM-RAWX measurements without CSV column shifting.

This exports raw per-signal measurements and validity flags. It does NOT infer
L1/L2 columns, repair the old CSV, or assign attack labels.
"""
from pathlib import Path
from datetime import datetime, timedelta, timezone
from io import BytesIO
import argparse
import json
import pandas as pd
from pyubx2 import UBXReader, UBX_PROTOCOL

GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)
GNSS_PREFIX = {0: "G", 1: "S", 2: "E", 3: "C", 4: "M", 5: "J", 6: "R", 7: "I"}

def valid_frame(data: bytes, pos: int):
    if data[pos:pos+2] != b"\xb5\x62" or pos + 8 > len(data):
        return None
    end = pos + 6 + int.from_bytes(data[pos+4:pos+6], "little") + 2
    if end > len(data):
        return None
    frame = data[pos:end]
    a = b = 0
    for byte in frame[2:-2]:
        a = (a + byte) & 0xff
        b = (b + a) & 0xff
    return frame if frame[-2:] == bytes((a, b)) else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ubx", required=True, help="Input original UBX file")
    ap.add_argument("--out", required=True, help="Output measurement CSV")
    args = ap.parse_args()
    src, out = Path(args.ubx), Path(args.out)
    if not src.is_file():
        raise SystemExit(f"Input not found: {src}")
    out.parent.mkdir(parents=True, exist_ok=True)
    data = src.read_bytes()
    rows = []
    frame_count = invalid_candidates = rawx_count = 0
    pos = 0
    while pos < len(data) - 8:
        sync = data.find(b"\xb5\x62", pos)
        if sync < 0:
            break
        frame = valid_frame(data, sync)
        if frame is None:
            invalid_candidates += 1
            pos = sync + 1
            continue
        frame_count += 1
        pos = sync + len(frame)
        try:
            parsed = next(iter(UBXReader(BytesIO(frame), protfilter=UBX_PROTOCOL)), (None, None))[1]
        except Exception:
            continue
        if getattr(parsed, "identity", "") != "RXM-RAWX":
            continue
        rawx_count += 1
        week = int(parsed.week)
        tow = float(parsed.rcvTow)
        leap_s = int(parsed.leapS)
        gps_time = GPS_EPOCH + timedelta(weeks=week, seconds=tow)
        utc_time = gps_time - timedelta(seconds=leap_s)
        n = int(parsed.numMeas)
        for j in range(1, n + 1):
            def get(field, default=None):
                return getattr(parsed, f"{field}_{j:02d}", default)
            gnss_id = int(get("gnssId"))
            sv_id = int(get("svId"))
            rows.append({
                "gps_time": gps_time.isoformat(timespec="milliseconds"),
                "utc_time_est": utc_time.isoformat(timespec="milliseconds"),
                "gps_week": week, "gps_tow_s": tow, "leap_s": leap_s,
                "satellite": f"{GNSS_PREFIX.get(gnss_id, '?')}{sv_id:02d}",
                "gnss_id": gnss_id, "sv_id": sv_id,
                "sig_id": int(get("sigId")), "freq_id": int(get("freqId", 0)),
                "pseudorange_m_raw": get("prMes"),
                "carrier_phase_cycles_raw": get("cpMes"),
                "doppler_hz_raw": get("doMes"),
                "cno_dbhz": int(get("cno")),
                "pseudorange_valid": int(get("prValid")),
                "carrier_phase_valid": int(get("cpValid")),
                "half_cycle_ambiguous": int(get("halfCyc", 0)),
                "sub_half_cycle": int(get("subHalfCyc", 0)),
                "pseudorange_std_m": get("prStd"),
                "carrier_phase_std_cycles": get("cpStd"),
                "doppler_std_hz": get("doStd"),
            })
    if not rows:
        raise SystemExit("No RXM-RAWX measurements recovered; no output written.")
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    summary = {
        "input": str(src), "input_bytes": len(data),
        "checksum_valid_ubx_frames": frame_count,
        "invalid_sync_candidates": invalid_candidates,
        "rxm_rawx_messages": rawx_count,
        "measurement_rows": len(df),
        "gps_time_min": df["gps_time"].min(),
        "gps_time_max": df["gps_time"].max(),
        "utc_time_est_min": df["utc_time_est"].min(),
        "utc_time_est_max": df["utc_time_est"].max(),
        "pseudorange_valid_rows": int(df["pseudorange_valid"].sum()),
        "carrier_phase_valid_rows": int(df["carrier_phase_valid"].sum()),
        "output_csv": str(out),
        "warning": "Recovery export only; verify message structure and signal mapping before scientific evaluation. No attack labels inferred."
    }
    summary_path = out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
