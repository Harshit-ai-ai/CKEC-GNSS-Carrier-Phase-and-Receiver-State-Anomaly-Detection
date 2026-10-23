# Detailed Measurement-Validation & Feasibility Audit — Phase 1 (Revised)

**Audit Completed:** 2026-10-14  
**Auditor:** Antigravity AI (CKEC-GNSS Research Project)  
**Status:** Rigorous Baseline Audit with Separated Signal Modalities  

---

## 1. Technical Corrections Applied

### A. Separation of True AGC from Alternate Signal Modalities
1. **FGI-SpoofRepo:** Hardware AGC is **not exported**. The prompt correlator magnitude $\sqrt{I_P^2 + Q_P^2}$ from the FGI-GSRx software receiver is **NOT** AGC. It is classified and evaluated separately as **Correlator Magnitude / Prompt Power**, not AGC.
2. **Yunnan-2024:** Hardware AGC is **not exported** (only `NAV-SAT`, `RXM-RAWX`, etc. logged). The reported $C/N_0$ carrier-to-noise density is **NOT** AGC. It is classified and evaluated separately as **$C/N_0$ Density**, not AGC.
3. **Jammertest-2024:** Contains **actual hardware AGC** via u-blox `UBX-MON-RF` (`agcCnt`). Requires strict validation of units, sampling rate, and multi-band alignment.

### B. Verification of Attack Onsets & Labels
* **FGI-SpoofRepo:** Attack onset is **not universally 60 seconds**. Documentation and FGI-GSRx evaluation show the clean baseline extends up to 130 s in several recordings (e.g., Targeted SFMC 370 s, Meaconing 470 s). Exact attack intervals must be recorded per file scenario and not assumed globally.
* **Jammertest-2024:** Attack sessions (jamming, spoofing, meaconing) are indexed with UTC timestamps from the official test trials in Andøya.
* **Yunnan-2024:** HackRF One and commercial jammers were triggered during specific scenario test runs. Exact start/stop intervals must be extracted from the session tables rather than assumed continuous.

### C. Carrier-Phase Compatibility
* `trackResults.carrPhase` in FGI-GSRx is in cycles/meters at tracking loop rate, whereas RINEX 3.04 (in Jammertest) records observation intervals (e.g. 1 Hz / 5 Hz) with cycle-slip flags (`LLI`) and specific signal codes (`L1C`, `L5I`).
* `UBX-RXM-RAWX` (Yunnan-2024) records half-cycle ambiguity and carrier phase in cycles (`cpMes`).
* Spatial inter-antenna phase differences are **blocked** across all three datasets due to the absence of calibrated multi-antenna baseline arrays.

---

## 2. Updated Feasibility Matrix

| Field | FGI-SpoofRepo (A) | Jammertest-2024 (B) | Yunnan-2024 (C) |
|---|---|---|---|
| **Hardware AGC** | ❌ **Missing** (Not logged) | ✅ **Available** (`UBX-MON-RF.agcCnt`) | ❌ **Missing** (Not logged) |
| **Correlator Magnitude** | ✅ **Available** ($\sqrt{I_P^2 + Q_P^2}$ via FGI-GSRx) | ⚠️ **Derivable** from raw baseband / tracking if UBX-RXM-CORR present | ❌ **Missing** (Standard NAV/RAWX only) |
| **Carrier-to-Noise ($C/N_0$)** | ✅ **Available** (RINEX / GSRx) | ✅ **Available** (RINEX `S1C`, `S5I`) | ✅ **Available** (`UBX-NAV-SAT`, `RXM-RAWX`) |
| **Carrier Phase** | ✅ **Available** (RINEX L1C / GSRx `carrPhase`) | ✅ **Available** (RINEX 3.04 L1C/L5I) | ✅ **Available** (`UBX-RXM-RAWX` `cpMes`) |
| **Attack Labels & Onset** | ⚠️ **Scenario-Specific** (130s baseline for SFMC/DFMC, not fixed 60s) | ✅ **Available** (Trial trial plans + event logs) | ⚠️ **Scenario-Specific** (Per test run documentation) |
| **Satellite Ephemeris** | ✅ **Available** (RINEX NAV broadcast) | ✅ **Available** (RINEX NAV sidecars) | ✅ **Available** (`UBX-RXM-SFRBX`) |
| **Antenna Baseline** | ❌ **Missing** (Single antenna) | ❌ **Missing** (Single site) | ❌ **Missing** (Single antenna) |
| **Multi-Constellation** | GPS L1, Galileo E1 (SFMC); +L5, E5a (DFMC) | GPS L1/L5, Galileo E1/E5a | GPS, BeiDou, Galileo, GLONASS |
| **Verdict** | **NARROW** (Test Correlator Magnitude & Phase) | **PROCEED** (Primary real dataset for true AGC + Phase) | **NARROW** (Test $C/N_0$ & Phase) |

---

## 3. Revised Detector Definitions (D1–D4)

To prevent claiming AGC evaluation when testing other physical measurements, detector evaluation is restructured as follows:

* **D1 — Raw Ratio Baseline:**  
  $$J_t = \frac{|\Delta \phi_t|}{|\Delta M_t| + \epsilon}$$  
  *(where $M_t$ is the evaluated secondary feature: true AGC on Jammertest; correlator magnitude on FGI; $C/N_0$ on Yunnan. Evaluated strictly as an exploratory baseline to document numerical instability when $\Delta M_t \approx 0$).*

* **D2 — Phase-Only Baseline:**  
  $$\text{Score}_t = |z_{\phi, t}| = \left| \frac{\Delta \phi_t - \mu_{\phi, \text{cal}}}{\sigma_{\phi, \text{cal}} + \epsilon} \right|$$  
  *(Quantifies detection capability and false-alarm performance of pure carrier-phase innovations).*

* **D3 — Signal-Strength / Power Baseline (Evaluated by Modality):**  
  * **D3a (True AGC-only):** $|z_{\text{agc}, t}|$ using `UBX-MON-RF` (Jammertest-2024).  
  * **D3b (Correlator-magnitude-only):** $|z_{\text{corr}, t}|$ using $\sqrt{I_P^2 + Q_P^2}$ (FGI-SpoofRepo).  
  * **D3c ($C/N_0$-only):** $|z_{\text{cn0}, t}|$ using carrier-to-noise density (Yunnan-2024 / Jammertest).  

* **D4 — Joint Feature Model (Cross-Modal Baseline Comparison):**  
  $$S_{\text{joint}, t} = w_\phi z_{\phi, t}^2 + w_M z_{M, t}^2$$  
  *(Evaluates whether combining carrier phase with the specific available secondary modality $M$ yields a reproducible, statistically significant improvement over both D2 and D3 at a fixed false-alarm rate).*

* **D5 & D6 — Spatial & Geometry Detectors:**  
  **Deferred from empirical dataset evaluation** due to the absence of multi-antenna baseline arrays across all available real datasets. Tested strictly in synthetic simulations.
