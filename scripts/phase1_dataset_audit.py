"""
scripts/phase1_dataset_audit.py — Audit candidate datasets for Phase 1 feasibility.

Audits:
- FGI-SpoofRepo
- Jammertest-2024
- Yunnan-2024

Outputs:
- reports/phase1_feasibility_matrix.csv
- reports/phase1_verdict.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from ckec.utils import setup_logging, load_config, get_path
from ckec.ingestion import (
    load_fgi_spoofRepo,
    load_jammertest_2024,
    load_yunnan_2024,
)


def main() -> None:
    cfg = load_config("config.yaml")
    log = setup_logging(level=cfg["logging"]["level"], log_file=cfg["logging"]["file"])
    log.info("=== Phase 1: Dataset Feasibility Audit (Revised) ===")

    data_raw = get_path(cfg, "data_raw")
    reports_dir = get_path(cfg, "reports")

    loaders = [
        load_fgi_spoofRepo,
        load_jammertest_2024,
        load_yunnan_2024,
    ]

    records = []
    verdict_lines = [
        "Phase 1 Dataset Feasibility Audit — Revised Verdicts",
        "============================================================",
        "Audit Date : 2026-10-14",
        "",
    ]

    for loader in loaders:
        _, audit = loader(data_raw)
        print(audit.summarise())

        rec = {
            "dataset": audit.dataset_name,
            "licence_confirmed": audit.licence_confirmed,
        }
        for f in audit.fields:
            rec[f.name] = f.status
        rec["verdict"] = audit.overall_verdict
        rec["verdict_notes"] = audit.verdict_notes
        records.append(rec)

        verdict_lines.append(f"Dataset : {audit.dataset_name}")
        verdict_lines.append(f"Verdict : {audit.overall_verdict}")
        verdict_lines.append(f"Notes   : {audit.verdict_notes}")
        verdict_lines.append("")

    csv_path = reports_dir / "phase1_feasibility_matrix.csv"
    pd.DataFrame(records).to_csv(csv_path, index=False)
    log.info("Feasibility matrix saved to: %s", csv_path)

    verdict_path = reports_dir / "phase1_verdict.txt"
    with open(verdict_path, "w", encoding="utf-8") as f:
        f.write("\n".join(verdict_lines))
    log.info("Verdict template saved to: %s", verdict_path)
    log.info("Phase 1 audit complete.")


if __name__ == "__main__":
    main()
