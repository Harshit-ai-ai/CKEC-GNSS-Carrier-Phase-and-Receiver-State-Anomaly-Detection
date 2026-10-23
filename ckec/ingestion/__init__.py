"""
ckec.ingestion — Dataset loaders and feasibility auditors.

Phase 1 Audit (Revised 2026-10-14):
FGI-SpoofRepo  : CC BY 4.0, NARROW  (Hardware AGC missing; prompt correlator magnitude sqrt(I_P^2+Q_P^2) available; onset per scenario; no antenna baseline)
Jammertest-2024: CC BY 4.0, PROCEED (Hardware AGC via UBX-MON-RF agcCnt, RINEX carrier phase, session event logs)
Yunnan-2024    : CC BY 4.0, NARROW  (Hardware AGC missing; C/N0 density available; UBX-RXM-RAWX phase; no antenna baseline)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

REQUIRED_FIELDS = [
    "hardware_agc",
    "correlator_magnitude",
    "carrier_to_noise_cn0",
    "carrier_phase",
    "attack_label",
    "attack_onset_time",
    "satellite_ephemeris",
    "antenna_baseline",
    "receiver_config",
]


@dataclass
class FieldStatus:
    name: str
    status: str = "unknown"
    notes: str = ""


@dataclass
class DatasetAudit:
    dataset_name: str
    fields: list = field(default_factory=list)
    licence_confirmed: bool = False
    licence_notes: str = ""
    overall_verdict: str = "pending"
    verdict_notes: str = ""

    def summarise(self) -> str:
        lines = [f"\n=== Dataset Audit: {self.dataset_name} ==="]
        for fs in self.fields:
            lines.append(f"  [{fs.status.upper():10s}]  {fs.name}  -  {fs.notes}")
        lines.append(f"  Licence confirmed: {self.licence_confirmed}  ({self.licence_notes})")
        lines.append(f"  Verdict: {self.overall_verdict.upper()}  -  {self.verdict_notes}")
        return "\n".join(lines)


def load_fgi_spoofRepo(data_dir) -> tuple:
    """FGI-SpoofRepo loader stub. Phase 1 audit: NARROW verdict."""
    data_dir = Path(data_dir)
    audit = DatasetAudit(dataset_name="FGI-SpoofRepo")
    audit.licence_confirmed = True
    audit.licence_notes = "CC BY 4.0 - confirmed via fairdata.fi 2026-10-14"
    audit.fields = [
        FieldStatus("hardware_agc", "missing", "Hardware AGC is NOT exported by NSL Stereo / FGI-GSRx."),
        FieldStatus("correlator_magnitude", "available", "sqrt(I_P^2+Q_P^2) from FGI-GSRx prompt tracking results."),
        FieldStatus("carrier_to_noise_cn0", "available", "C/N0 calculated in tracking channels."),
        FieldStatus("carrier_phase", "available", "trackResults.carrPhase or RINEX L1C obs."),
        FieldStatus("attack_label", "derivable", "Binary ground truth derived from scenario event logs."),
        FieldStatus("attack_onset_time", "available", "Per-scenario documentation (clean baseline ~130s in SFMC/DFMC, not fixed 60s)."),
        FieldStatus("satellite_ephemeris", "available", "FGI-GSRx decodes nav msg; RINEX NAV export."),
        FieldStatus("antenna_baseline", "missing", "SINGLE-ANTENNA ONLY. D5/D6 spatial detectors blocked."),
        FieldStatus("receiver_config", "available", "NSL Stereo: L1@1569.03MHz 26MHz SR 4.2MHz BW."),
    ]
    audit.overall_verdict = "narrow"
    audit.verdict_notes = "D1-D4 proceed testing Correlator Magnitude and Phase. D5 blocked by missing baseline."
    return pd.DataFrame(), audit


def load_jammertest_2024(data_dir) -> tuple:
    """Jammertest 2024 loader stub. Phase 1 audit: PROCEED verdict."""
    data_dir = Path(data_dir)
    audit = DatasetAudit(dataset_name="Jammertest-2024")
    audit.licence_confirmed = True
    audit.licence_notes = "CC BY 4.0 - Zenodo DOI:10.5281/zenodo.15910563 confirmed 2026-10-14"
    audit.fields = [
        FieldStatus("hardware_agc", "available", "True hardware AGC via UBX-MON-RF (agcCnt, 0-8191 count scale)."),
        FieldStatus("correlator_magnitude", "unknown", "Requires raw baseband correlator messages if logged."),
        FieldStatus("carrier_to_noise_cn0", "available", "RINEX S1C/S5I observation streams."),
        FieldStatus("carrier_phase", "available", "RINEX 3.04 L1C/L5I observables with LLI flags."),
        FieldStatus("attack_label", "available", "Per-session metadata: 0=nominal, 1=jamming, 2=spoofing, 3=meaconing."),
        FieldStatus("attack_onset_time", "available", "Session schedule and event log timestamps from Andøya field trials."),
        FieldStatus("satellite_ephemeris", "available", "RINEX NAV sidecars included."),
        FieldStatus("antenna_baseline", "missing", "Single-site receiver setups. D5/D6 spatial detectors blocked."),
        FieldStatus("receiver_config", "available", "u-blox ZED-F9P configuration logs included."),
    ]
    audit.overall_verdict = "proceed"
    audit.verdict_notes = "Primary real-data dataset. True hardware AGC and RINEX phase confirmed. D5 blocked."
    return pd.DataFrame(), audit


def load_yunnan_2024(data_dir) -> tuple:
    """Yunnan-2024 loader stub. Phase 1 audit: NARROW verdict."""
    data_dir = Path(data_dir)
    audit = DatasetAudit(dataset_name="Yunnan-2024")
    audit.licence_confirmed = True
    audit.licence_notes = "CC BY 4.0 - Mendeley Data / Data in Brief 2024 confirmed 2026-10-14"
    audit.fields = [
        FieldStatus("hardware_agc", "missing", "Hardware AGC is NOT exported in published UBX tables."),
        FieldStatus("correlator_magnitude", "missing", "Baseband correlator taps not captured."),
        FieldStatus("carrier_to_noise_cn0", "available", "C/N0 from UBX-NAV-SAT and RXM-RAWX streams."),
        FieldStatus("carrier_phase", "available", "UBX-RXM-RAWX carrier phase in cycles (cpMes). GPS+BDS+Galileo."),
        FieldStatus("attack_label", "available", "Documented per test run: clean, jammed, spoofed."),
        FieldStatus("attack_onset_time", "available", "Per-scenario experiment run time logs (Table 1 / text)."),
        FieldStatus("satellite_ephemeris", "available", "UBX-RXM-SFRBX broadcast frames."),
        FieldStatus("antenna_baseline", "missing", "Single rooftop antenna setup. D5/D6 spatial detectors blocked."),
        FieldStatus("receiver_config", "available", "u-blox receiver + HackRF One SDR setup documented in paper."),
    ]
    audit.overall_verdict = "narrow"
    audit.verdict_notes = "D1-D4 proceed testing C/N0 and Phase. Multi-constellation available. D5 blocked."
    return pd.DataFrame(), audit
