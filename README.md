# CKEC-GNSS

**Cross-Modal Kinematic–Electromagnetic Coupling for Single-Antenna GNSS Spoofing Detection**

> 🔬 **Research Status:** Validated on authentic physical field trial recordings from the Jammertest-2024 campaign (Bleik, Andøya, Norway; DOI: [10.5281/zenodo.15910563](https://zenodo.org/records/15910563)).

---

## Project Overview

This repository implements and evaluates the **CKEC** detection framework, which investigates
whether temporal cross-modal coupling between electromagnetic receiver observables (hardware Automatic Gain Control from `UBX-MON-RF`) and kinematic carrier-phase innovations can improve single-antenna GNSS spoofing and meaconing detection beyond conventional single-channel baselines.

### Key Empirical Results on Real Data (Jammertest-2024, 20 Evaluated Scenarios, Target FAR = 0.01):
- **D2 (Carrier Phase Alone):** Mean $F_1 = \mathbf{0.0434}$ ($\text{std} = 0.0265$). In wild open-air environments, carrier-phase drift cannot reliably distinguish subtle spoofing from natural dynamic maneuvers without exceeding acceptable false-alarm ceilings.
- **D3a (Hardware AGC Alone):** Mean $F_1 = \mathbf{0.2605}$ ($\text{std} = 0.1993$). Hardware gain control provides a strong coarse indicator for jamming and power-boosted attacks.
- **D4 (Joint Innovation Energy):** Mean $F_1 = \mathbf{0.2725}$ ($\text{std} = 0.1997$). Combines normalized squared residuals ($Z_\phi^2 + Z_A^2$) without division-by-zero singularities.
- **D6 (Full CKEC with Cross-Modal Consistency):** Mean $F_1 = \mathbf{0.2980}$ ($w_c = 1.0$) and $\mathbf{0.3282}$ ($w_c = 2.0$, $+5.58\%$ absolute gain over D4).
  - Paired bootstrap hypothesis tests ($N = 2,000$, $p < 0.05$) confirm that D6 is **statistically significantly superior to D4 in 15 out of 20 scenarios** and **superior to D2 in 20 out of 20 scenarios**.
  - Systematic ablation across window sizes $W \in [5, 10, 20]$ and weights $w_c \in [0.0, 2.0]$ demonstrates monotonic detection improvement from the cross-modal decorrelation penalty $(1 - |C_t|)$.

---

## Repository Structure

```
ckec-gnss/
├── data/
│   ├── raw/                  # Downloaded raw archives (Jammertest-2024)
│   ├── processed/            # Cleaned, aligned time-series
│   │   └── real/             # 24 real Jammertest-2024 scenarios
│   └── synthetic/            # Controlled simulation outputs
├── notebooks/                # Exploratory Jupyter notebooks
├── ckec/                     # Core Python library
│   ├── ingestion/            # Dataset loaders & schema validators
│   ├── preprocessing/        # Signal cleaning, alignment, unwrapping
│   ├── features/             # Feature extraction (AGC, phase, consistency)
│   ├── detectors/            # Detectors (D1, D2, D3a, D4, D6)
│   ├── evaluation/           # Metrics, Wilson CIs, paired bootstrap tests
│   ├── simulation/           # Synthetic simulation suite
│   └── utils/                # Shared helpers (logging, config, I/O)
├── scripts/                  # Reproducibility scripts
│   ├── ingest_jammertest_real.py  # Ingest real Jammertest-2024 data
│   ├── evaluate_real_data.py      # Real-data evaluation & bootstrap tests
│   ├── run_real_ablation.py       # Systematic w_c and window ablation
│   └── generate_figures.py        # Publication figures
├── reports/
│   ├── real_evaluation_results.csv # 20-scenario real benchmark metrics
│   ├── real_ablation_results.csv   # 300-run parameter ablation results
│   ├── final_evaluation_report.md  # Detailed technical evaluation report
│   └── figures/                    # Generated figures (fig1-fig5)
├── tests/                    # Unit and integration test suite
└── docs/                     # Invention disclosure and feasibility matrix
```

---

## Quick Start

```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# 2. Install dependencies in editable mode
pip install -e .

# 3. Ingest real Jammertest-2024 dataset (adjust path to downloaded archive)
python scripts/ingest_jammertest_real.py --dataset-root "path/to/GNSS_DATASET_JAMMING_SPOOFING"

# 4. Run real-data detector evaluation with Wilson CIs & paired bootstrap tests
python scripts/evaluate_real_data.py --bootstrap-iter 2000

# 5. Run real-data ablation sweep over coupling weight (w_c) and window length (W)
python scripts/run_real_ablation.py

# 6. Generate all figures (including real Jammertest traces)
python scripts/generate_figures.py
```

---

## Key Design Principles

- **Empirical Rigor:** Thresholds are calibrated strictly on benign validation segments prior to attack onset, never on test data.
- **Physical Realism:** Evaluates genuine physical observables (true `UBX-MON-RF` hardware AGC count and multi-constellation RINEX carrier phase) under active field jamming and spoofing trials.
- **Statistically Validated Novelty:** All detector comparisons report Wilson score confidence intervals for FAR and two-sided paired bootstrap hypothesis tests ($p < 0.05$).
- **Reproducibility:** All processing parameters, random seeds, and pipeline steps are fully specified in version-controlled scripts.

---

## Requirements

- Python ≥ 3.10
- Required packages: `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`, `click`, `pyyaml`
- See `pyproject.toml` for complete package specifications.

---

## Licence & Patent Notice

Research software. Technical mechanisms and empirical evaluations described herein are the subject of research disclosure `CKEC-INV-2026-001`.
# CKEC-GNSS-Carrier-Phase-and-Receiver-State-Anomaly-Detection
