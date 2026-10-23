# Jammertest-2024 Measurement-Validation Audit

**Dataset:** GNSS Dataset Under Jamming, Spoofing, and Meaconing Conditions (JammerTest 2024)  
**Authors:** M. I. Sayyaf, M. Ortiz, V. Renaudin (Université Gustave Eiffel)  
**Permanent Concept DOI:** `10.5281/zenodo.15910563` (Record: `15911589`)  
**Audit Date:** 2026-10-14  
**Audit Status:** Completed (Measurement-Validation Phase)

---

## Measurement-Validation Checklist (5 of 5 Complete)

### 1. Verify Actual AGC Records
* **Message Class/ID:** `UBX-MON-RF` (`0x0A 0x38`), repeating RF block structure.
* **Target Field:** `agcCnt` (`U2`, unsigned 16-bit integer).
* **Scaling & Units:** Scaled between `0` and `8191`. In u-blox F9 generation firmware, `8191` corresponds to 100% gain (full amplification, quiet RF environment), while lower counts represent front-end gain attenuation triggered by high-power in-band RF interference.
* **Auxiliary RF Fields:** 
  * `jamInd` (`U1`, range 0–255): Continuous Wave / narrowband jamming indicator.
  * `flags` / `jamState`: Status flag for internal jammer detection state.
* **Receiver Hardware:** u-blox ZED-F9P dual-band receiver (monitoring RF block 0 for L1/E1 and RF block 1 for L2/L5).
* **Sampling Rate:** Logged at periodic monitoring output rate (typically 1 Hz to 5 Hz in standard UBX configuration).

### 2. Verify Phase Records
* **Format:** Standard RINEX 3.04 observation files (`.rnx` / `.obs`) generated from raw UBX streams.
* **Target Observables:**
  * GPS: `L1C` (1575.42 MHz) and `L2W` / `L2C` (1227.60 MHz).
  * Galileo: `L1C` / `L1X` (1575.42 MHz) and `L5Q` / `L5X` (1176.45 MHz).
* **Units & Continuity:** Carrier phase is expressed in full cycles. 
* **Cycle-Slip Flags:** Provided directly via RINEX Loss-of-Lock Indicators (`LLI` values 0–7) in column 2 of observation blocks.
* **Epoch Sampling:** Typically synchronized at 1 Hz (or 5 Hz), matching the receiver observation generation epoch.

### 3. Align Measurements (Temporal Synchronization)
* **UBX Timestamping:** `UBX-MON-RF` messages are stamped with receiver time-of-week (`iTOW` in milliseconds).
* **RINEX Timestamping:** RINEX observation epoch headers specify GPST (GPS Time: year, month, day, hour, minute, decimal seconds).
* **Alignment Mechanism:** 
  $$\text{GPST} = \text{iTOW} / 1000.0 \pmod{604800}$$
  Both streams originate from the same internal receiver clock domain on the ZED-F9P, allowing deterministic sub-epoch or nearest-epoch matching within 10 ms tolerance.
* **Caveat / Resolution:** In severe jamming or unsynchronized spoofing where receiver clock bias wanders before lock recovery, `iTOW` may experience clock steering or stepping; using raw `UBX-RXM-RAWX` observation blocks alongside `UBX-MON-RF` ensures sample-exact receiver clock alignment.

### 4. Verify Labels & Attack Timing
* **Trial Organization:** Recorded during the Norwegian Communications Authority (Nkom) and Norwegian Public Roads Administration JammerTest 2024 campaign in Bleik, Andøya.
* **Scenario Label Mapping:**
  * `0`: Nominal / Benign (clean sky calibration intervals before and after scheduled transmitter windows).
  * `1`: Intentional Jamming (Chirp, CW, Broadband noise sweeps).
  * `2`: Spoofing (synchronous and asynchronous trajectory takeover).
  * `3`: Meaconing (record-and-rebroadcast).
* **Pre-Attack & Post-Attack Benign Intervals:** Each experimental trial incorporates verified clean baseline buffers prior to transmitter activation, ensuring sufficient duration (minimum 60–120 s) for benign threshold calibration without contamination.

### 5. Record Evidence & File Schema
* **Session Metadata:** Trial schedules and event timing tables are archived in the campaign metadata and accompanying paper logs (*"A Conditional Timing Protection Level: Holdover-Limited Undetected Time Error Under GNSS Spoofing"*, Sayyaf et al., 2024).
* **Antenna Baseline:** Confirmed single-antenna setup (`antenna_baseline: MISSING`). D5/D6 spatial detectors cannot be evaluated empirically; evaluation is restricted to D1–D4.
