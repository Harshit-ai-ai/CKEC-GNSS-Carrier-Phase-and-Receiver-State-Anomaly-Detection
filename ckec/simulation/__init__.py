"""
ckec.simulation — Controlled synthetic GNSS signal generation.

PURPOSE
-------
Synthetic data is used to:
  1. Verify that detectors behave as expected under controlled conditions.
  2. Test scenarios not available in real recordings.
  3. Run ablation experiments across known ground truth.

IMPORTANT LIMITATIONS
---------------------
* Synthetic results DO NOT demonstrate real-world detection capability.
* Every synthetic scenario must be clearly labelled as simulated data.
* Physical signal models are deliberately simplified.  Do not present
  simplification assumptions as validated physics without supporting evidence.
* Synthetic accuracy scores must NEVER be reported as real-data results.

All generators return a pandas DataFrame with columns:
  time_s        : sample timestamps in seconds
  carrier_phase : synthetic unwrapped phase (radians)
  agc           : synthetic AGC value (dB, conceptual units; 0-8191 scale)
  cn0_db_hz     : synthetic C/N0 in dB-Hz
  attack_label  : 0 (benign) or 1 (spoofing)
And a metadata dict describing the scenario parameters.

V2 Improvements
---------------
* Authentic phase uses Ornstein-Uhlenbeck process (mean-reverting) for realistic
  correlation structure between successive innovations.
* AGC is now physically coupled to phase innovations during authentic tracking:
  when phase innovations spike (multipath), AGC also reacts slightly.
* Spoofing transitions include a realistic AGC dip (attacker captures lock) then
  spike (counterfeit power increase), matching real Jammertest observations.
* Added Rayleigh fading scenario and stealth spoofing (power-matched attack).
* C/N0 column added to all scenarios (correlated with AGC).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_SYNTHETIC_TAG = "SYNTHETIC"   # always present in metadata['data_source']


# ---------------------------------------------------------------------------
# Low-level primitive generators (V2 — physically coupled)
# ---------------------------------------------------------------------------

_AGC_BASELINE_DB = 30.0      # conceptual dB, maps to ~3600/8191 AGC count
_CN0_BASELINE_DBHz = 42.0   # typical GPS L1 open-sky C/N0


def _ou_process(
    n: int,
    rng: np.random.Generator,
    theta: float = 0.05,   # mean-reversion speed
    mu: float = 0.0,
    sigma: float = 0.05,
    x0: float = 0.0,
) -> np.ndarray:
    """
    Ornstein-Uhlenbeck process: mean-reverting correlated noise.
    Used for realistic phase innovation correlation structure.
    """
    x = np.empty(n)
    x[0] = x0
    noise = rng.normal(0, sigma, size=n)
    for i in range(1, n):
        x[i] = x[i-1] + theta * (mu - x[i-1]) + noise[i]
    return x


def _benign_phase(
    n: int,
    rng: np.random.Generator,
    noise_std: float = 0.05,
    drift_rate: float = 0.002,   # rad/sample — slow geometric drift
) -> np.ndarray:
    """
    Benign phase: slow drift + mean-reverting correlated noise (OU process).
    The OU process better represents real carrier-phase innovation dynamics
    where successive samples are correlated due to the tracking loop.
    """
    drift = drift_rate * np.arange(n)
    corr_noise = _ou_process(n, rng, theta=0.15, sigma=noise_std)
    return drift + corr_noise


def _benign_agc(
    n: int,
    rng: np.random.Generator,
    noise_std: float = 0.15,
    baseline_db: float = _AGC_BASELINE_DB,
    phase_innovations: np.ndarray | None = None,
    phase_coupling: float = 0.25,
) -> np.ndarray:
    """
    Synthetic AGC level with an optional controlled coupling to phase
    innovations.

    IMPORTANT:
    This is an experimental mechanism model, not a validated physical
    GNSS receiver model. When phase_innovations is supplied, the coupling
    is deliberately defined so that differenced AGC contains a component
    proportional to standardized phase innovations.

    The coupling can be tested in benign simulations and contrasted with
    attacks in which phase innovations change independently of AGC.
    """
    agc = (
        baseline_db
        + _ou_process(n, rng, theta=0.02, sigma=noise_std * 0.6)
        + rng.normal(0, noise_std, size=n)
    )

    if phase_innovations is not None:
        innovations = np.asarray(phase_innovations, dtype=float)

        if len(innovations) != n:
            raise ValueError(
                "phase_innovations must have the same length as n."
            )

        scale = np.nanstd(innovations)
        if not np.isfinite(scale) or scale < 1e-12:
            standardized = np.zeros(n, dtype=float)
        else:
            standardized = np.nan_to_num(
                innovations / scale,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

        # Integrating the controlled component into the AGC level means
        # that differencing AGC exposes a component related to the phase
        # innovations. This is intentional for a mechanism test.
        agc += phase_coupling * np.cumsum(standardized)

    return agc


def _cn0_from_agc(
    agc: np.ndarray,
    rng: np.random.Generator,
    noise_std: float = 0.5,
    baseline_cn0: float = _CN0_BASELINE_DBHz,
    agc_baseline: float = _AGC_BASELINE_DB,
) -> np.ndarray:
    """
    Synthesise C/N0 (dB-Hz) correlated with AGC:
    C/N0 decreases when AGC decreases (signal strength drops).
    A small noise term models receiver-internal C/N0 estimation error.
    """
    delta_agc = agc - agc_baseline
    cn0 = baseline_cn0 + 0.8 * delta_agc + rng.normal(0, noise_std, size=len(agc))
    return np.clip(cn0, 20.0, 60.0)


def _multipath_phase(
    n: int,
    rng: np.random.Generator,
    noise_std: float = 0.05,
    mp_amplitude: float = 0.3,    # rad
    mp_period: int = 200,         # samples
) -> np.ndarray:
    """
    Benign multipath: slow sinusoidal + OU noise.
    Uses two superimposed sinusoids at different periods to mimic
    realistic multi-reflector environments.
    """
    drift = 0.002 * np.arange(n)
    mp1 = mp_amplitude * np.sin(2 * np.pi * np.arange(n) / mp_period)
    mp2 = (mp_amplitude * 0.4) * np.sin(2 * np.pi * np.arange(n) / (mp_period * 1.7 + 30))
    ou_noise = _ou_process(n, rng, theta=0.1, sigma=noise_std)
    return drift + mp1 + mp2 + ou_noise


# ---------------------------------------------------------------------------
# Scenario generators
# ---------------------------------------------------------------------------

def generate_abrupt_spoofing(
    n_samples: int = 10_000,
    onset: int | None = None,
    noise_std_phase: float = 0.05,
    noise_std_agc: float = 0.2,
    spoof_phase_drift_rate: float = 0.18,   # rad/sample — persistent innovation bias
    spoof_agc_jump_db: float = 3.5,         # dB — AGC step (opposed to phase)
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Scenario: authentic GNSS signal → abrupt spoofing transition at sample `onset`.

    V3 FIX — Root cause corrected:
    Previously we added `phase[onset:] += offset` which only creates a SINGLE
    SPIKE in delta_phase at the onset sample; all subsequent innovations are
    identical to benign.  This made detectors see no signal post-onset.

    Now we inject a PERSISTENT DRIFT RATE post-onset:
      phase[onset + k] += drift_rate * k  for k = 0, 1, ...
    This means delta_phase innovations are elevated by `drift_rate` for every
    sample after onset — the innovation mean shifts, which is what CUSUM,
    JointDetector and CKEC can actually detect.

    AGC is given an opposed step (C/N0 drops at spoofer capture) which breaks
    the phase-AGC coupling and gives D4/D6 a discriminating signal.
    """
    if onset is None or onset >= n_samples:
        onset = n_samples // 2
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / 10.0     # 10 Hz → time in seconds

    # --- Benign segment (OU-process phase + coupled AGC) ---
    phase = _benign_phase(n_samples, rng, noise_std=noise_std_phase)
    phase_inno = np.diff(phase, prepend=phase[0])
    agc = _benign_agc(n_samples, rng, noise_std=noise_std_agc,
                      phase_innovations=phase_inno, phase_coupling=0.06)
    label = np.zeros(n_samples, dtype=int)

    # --- Attack: inject persistent innovation-level drift post-onset ---
    # A drift of drift_rate rad/sample shifts innovations away from ~N(0, noise_std)
    # toward ~N(drift_rate, noise_std).  With drift_rate >> noise_std this is
    # immediately detectable; with drift_rate ~ noise_std it requires accumulation.
    n_attack = n_samples - onset
    attack_ramp = spoof_phase_drift_rate * np.arange(n_attack)   # cumulative drift
    phase[onset:] += attack_ramp

    # AGC: short capture dip (5 samples) then persistent elevation
    dip_end = min(onset + 5, n_samples)
    agc[onset:dip_end] -= spoof_agc_jump_db * 0.6
    agc[dip_end:] += spoof_agc_jump_db
    # Post-onset phase and AGC are now DECOUPLED — phase drifts up, AGC steps up.
    # This is the discriminating feature for D4 (joint) and D6 (full CKEC).
    label[onset:] = 1

    cn0 = _cn0_from_agc(agc, rng)
    df = pd.DataFrame({
        "time_s": t, "carrier_phase": phase, "agc": agc,
        "cn0_db_hz": cn0, "attack_label": label,
    })
    meta = {
        "data_source": _SYNTHETIC_TAG,
        "scenario": "abrupt_spoofing",
        "onset_sample": onset,
        "spoof_phase_drift_rate_rad_per_sample": spoof_phase_drift_rate,
        "spoof_agc_jump_db": spoof_agc_jump_db,
        "seed": seed,
    }
    log.info("Generated abrupt-spoofing scenario: %d samples, onset=%d, seed=%d",
             n_samples, onset, seed)
    return df, meta


def generate_gradual_spoofing(
    n_samples: int = 10_000,
    onset: int | None = None,
    ramp_duration: int | None = None,
    noise_std_phase: float = 0.05,
    noise_std_agc: float = 0.2,
    max_phase_drift_rate: float = 0.12,   # rad/sample at full ramp
    max_spoof_agc_increase_db: float = 2.5,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Scenario: authentic GNSS signal → gradual spoofing ramp starting at `onset`.

    V3 FIX: Like abrupt, previously the accumulated phase offset only produced
    a one-sample innovation spike. Now we inject a RAMPING DRIFT RATE that
    grows from 0 to max_phase_drift_rate over ramp_duration samples, creating
    a slowly-increasing innovation bias that CUSUM and D4 can accumulate.
    After ramp_end, drift rate is held constant (persistent pull-off).
    """
    if onset is None or onset >= n_samples:
        onset = n_samples // 2
    if ramp_duration is None:
        ramp_duration = max(1, (n_samples - onset) // 5)
    ramp_duration = min(ramp_duration, max(1, n_samples - onset))

    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / 10.0

    phase = _benign_phase(n_samples, rng, noise_std=noise_std_phase)
    phase_inno = np.diff(phase, prepend=phase[0])
    agc = _benign_agc(n_samples, rng, noise_std=noise_std_agc,
                      phase_innovations=phase_inno, phase_coupling=0.06)
    label = np.zeros(n_samples, dtype=int)

    ramp_end = min(onset + ramp_duration, n_samples)
    ramp_samples = max(1, ramp_end - onset)

    # Drift rate ramps up with sigmoid during [onset, ramp_end]
    x = np.linspace(-6, 6, ramp_samples)
    sigmoid = 1.0 / (1.0 + np.exp(-x))               # rate profile [0→1]
    drift_rates_ramp = max_phase_drift_rate * sigmoid  # rad/sample at each step

    # Cumulative phase injection = cumsum of per-step drift rates
    phase_inj_ramp = np.cumsum(drift_rates_ramp)
    phase[onset:ramp_end] += phase_inj_ramp

    # Post-ramp: continue at max drift rate
    if ramp_end < n_samples:
        n_post = n_samples - ramp_end
        phase_inj_post = phase_inj_ramp[-1] + max_phase_drift_rate * np.arange(1, n_post + 1)
        phase[ramp_end:] += phase_inj_post

    # AGC: same sigmoid ramp but in opposite direction (C/N0 change at capture)
    agc_inj_ramp = max_spoof_agc_increase_db * sigmoid
    agc[onset:ramp_end] += agc_inj_ramp
    if ramp_end < n_samples:
        agc[ramp_end:] += max_spoof_agc_increase_db

    label[onset:] = 1

    cn0 = _cn0_from_agc(agc, rng)
    df = pd.DataFrame({
        "time_s": t, "carrier_phase": phase, "agc": agc,
        "cn0_db_hz": cn0, "attack_label": label,
    })
    meta = {
        "data_source": _SYNTHETIC_TAG,
        "scenario": "gradual_spoofing",
        "onset_sample": onset,
        "ramp_duration_samples": ramp_duration,
        "max_phase_drift_rate_rad_per_sample": max_phase_drift_rate,
        "max_agc_increase_db": max_spoof_agc_increase_db,
        "seed": seed,
    }
    log.info("Generated gradual-spoofing scenario: %d samples, onset=%d, ramp=%d, seed=%d",
             n_samples, onset, ramp_duration, seed)
    return df, meta


def generate_benign_multipath(
    n_samples: int = 10_000,
    noise_std_phase: float = 0.05,
    noise_std_agc: float = 0.2,
    mp_amplitude: float = 0.3,
    mp_period: int = 200,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Scenario: entirely benign recording with multi-reflector multipath.

    V2: Two-component multipath model + OU noise + AGC coupled to phase.
    Used to calibrate false-alarm rates. A good detector should NOT flag this.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / 10.0

    phase = _multipath_phase(n_samples, rng, noise_std=noise_std_phase,
                              mp_amplitude=mp_amplitude, mp_period=mp_period)
    phase_inno = np.diff(phase, prepend=phase[0])
    agc = _benign_agc(n_samples, rng, noise_std=noise_std_agc,
                       phase_innovations=phase_inno, phase_coupling=0.06)
    label = np.zeros(n_samples, dtype=int)

    cn0 = _cn0_from_agc(agc, rng)
    df = pd.DataFrame({
        "time_s": t, "carrier_phase": phase, "agc": agc,
        "cn0_db_hz": cn0, "attack_label": label,
    })
    meta = {
        "data_source": _SYNTHETIC_TAG,
        "scenario": "benign_multipath",
        "mp_amplitude_rad": mp_amplitude,
        "mp_period_samples": mp_period,
        "seed": seed,
    }
    log.info("Generated benign-multipath scenario: %d samples, seed=%d", n_samples, seed)
    return df, meta


def generate_agc_variation_only(
    n_samples: int = 10_000,
    agc_step_db: float = 2.0,
    step_onset: int | None = None,
    noise_std_phase: float = 0.05,
    noise_std_agc: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Scenario: benign phase + sudden AGC step (no spoofing).

    Tests whether AGC variation alone (e.g. due to fading or interference)
    causes false alarms in the raw-ratio or AGC-only detectors.
    V2: Phase remains uncorrelated with the AGC step.
    """
    if step_onset is None or step_onset >= n_samples:
        step_onset = n_samples // 2
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / 10.0

    phase = _benign_phase(n_samples, rng, noise_std=noise_std_phase)
    agc   = _benign_agc(n_samples, rng, noise_std=noise_std_agc)
    agc[step_onset:] += agc_step_db   # benign AGC change, phase is NOT affected
    label = np.zeros(n_samples, dtype=int)

    cn0 = _cn0_from_agc(agc, rng)
    df = pd.DataFrame({
        "time_s": t, "carrier_phase": phase, "agc": agc,
        "cn0_db_hz": cn0, "attack_label": label,
    })
    meta = {
        "data_source": _SYNTHETIC_TAG,
        "scenario": "agc_variation_only",
        "agc_step_db": agc_step_db,
        "step_onset": step_onset,
        "seed": seed,
    }
    return df, meta


def generate_phase_anomaly_only(
    n_samples: int = 10_000,
    phase_drift_rate: float = 0.15,   # rad/sample — innovation-level drift
    jump_onset: int | None = None,
    noise_std_phase: float = 0.05,
    noise_std_agc: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Scenario: power-matched spoofer — phase drifts, AGC barely changes.

    V3 FIX: Uses persistent drift rate instead of accumulated offset.
    Tests whether phase-only detectors (D2, CUSUM) can detect attacks
    that leave AGC nearly unchanged.
    """
    if jump_onset is None or jump_onset >= n_samples:
        jump_onset = n_samples // 2
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / 10.0

    phase = _benign_phase(n_samples, rng, noise_std=noise_std_phase)
    agc = _benign_agc(n_samples, rng, noise_std=noise_std_agc)

    n_attack = n_samples - jump_onset
    phase[jump_onset:] += phase_drift_rate * np.arange(n_attack)
    # Power-matched: AGC shows only tiny decorrelation (spoofer isn't perfect)
    agc[jump_onset:] += rng.normal(0, noise_std_agc * 0.25, size=n_attack)
    label = np.zeros(n_samples, dtype=int)
    label[jump_onset:] = 1

    cn0 = _cn0_from_agc(agc, rng)
    df = pd.DataFrame({
        "time_s": t, "carrier_phase": phase, "agc": agc,
        "cn0_db_hz": cn0, "attack_label": label,
    })
    meta = {
        "data_source": _SYNTHETIC_TAG,
        "scenario": "phase_anomaly_only",
        "phase_drift_rate_rad_per_sample": phase_drift_rate,
        "jump_onset": jump_onset,
        "seed": seed,
    }
    return df, meta


def generate_rayleigh_fading(
    n_samples: int = 10_000,
    noise_std_phase: float = 0.05,
    noise_std_agc: float = 0.2,
    fading_rate_hz: float = 0.5,       # fade cycles per second
    fading_depth_db: float = 6.0,      # max AGC reduction during deep fade
    sampling_rate_hz: float = 10.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Scenario: entirely benign recording with Rayleigh fading envelope.

    Models a receiver moving through an urban canyon where signal amplitude
    follows a Rayleigh distribution. AGC tracks the fading envelope while
    carrier phase is largely unaffected (tracking loop compensates).
    Used to test for false alarms from the D4 joint detector under fading.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / sampling_rate_hz

    # Rayleigh fading envelope: sum of two Gaussian quadratures
    f0 = fading_rate_hz / sampling_rate_hz
    fading_i = np.convolve(
        rng.normal(0, 1, n_samples + 200),
        np.hanning(int(1 / (f0 + 1e-6)) + 2),
        mode="full",
    )[:n_samples]
    fading_q = np.convolve(
        rng.normal(0, 1, n_samples + 200),
        np.hanning(int(1 / (f0 + 1e-6)) + 2),
        mode="full",
    )[:n_samples]
    rayleigh_env = np.sqrt(fading_i**2 + fading_q**2)
    rayleigh_env /= np.median(rayleigh_env)   # normalise to unit median
    fading_db = 20 * np.log10(np.clip(rayleigh_env, 0.01, None))
    fading_db = np.clip(fading_db, -fading_depth_db, 3.0)

    phase = _benign_phase(n_samples, rng, noise_std=noise_std_phase)
    agc = _benign_agc(n_samples, rng, noise_std=noise_std_agc) + fading_db
    label = np.zeros(n_samples, dtype=int)

    cn0 = _cn0_from_agc(agc, rng)
    df = pd.DataFrame({
        "time_s": t, "carrier_phase": phase, "agc": agc,
        "cn0_db_hz": cn0, "attack_label": label,
    })
    meta = {
        "data_source": _SYNTHETIC_TAG,
        "scenario": "rayleigh_fading",
        "fading_rate_hz": fading_rate_hz,
        "fading_depth_db": fading_depth_db,
        "seed": seed,
    }
    log.info("Generated Rayleigh-fading scenario: %d samples, seed=%d", n_samples, seed)
    return df, meta


def generate_stealth_spoofing(
    n_samples: int = 10_000,
    onset: int | None = None,
    ramp_duration: int | None = None,
    noise_std_phase: float = 0.05,
    noise_std_agc: float = 0.2,
    max_phase_drift_rate: float = 0.06,  # low drift — hard to detect per-sample
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Scenario: stealth spoofing — power-matched, very slow drift rate.

    V3 FIX: Uses innovation-level drift (not accumulated offset). Drift rate
    is intentionally small so single-sample detectors (D2, D3a) struggle but
    CUSUM (D7) accumulates the evidence over time — the key contrast.
    AGC barely changes (power-matched attacker).
    """
    if onset is None or onset >= n_samples:
        onset = n_samples // 2
    if ramp_duration is None:
        ramp_duration = max(1, (n_samples - onset) // 3)
    ramp_duration = min(ramp_duration, max(1, n_samples - onset))

    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / 10.0

    phase = _benign_phase(n_samples, rng, noise_std=noise_std_phase)
    phase_inno = np.diff(phase, prepend=phase[0])
    agc = _benign_agc(n_samples, rng, noise_std=noise_std_agc,
                      phase_innovations=phase_inno, phase_coupling=0.06)
    label = np.zeros(n_samples, dtype=int)

    ramp_end = min(onset + ramp_duration, n_samples)
    ramp_samples = max(1, ramp_end - onset)

    # Drift rate ramps from 0 to max over ramp_duration
    x = np.linspace(-6, 6, ramp_samples)
    sigmoid = 1.0 / (1.0 + np.exp(-x))
    drift_rates = max_phase_drift_rate * sigmoid
    phase[onset:ramp_end] += np.cumsum(drift_rates)
    if ramp_end < n_samples:
        n_post = n_samples - ramp_end
        base = phase[ramp_end - 1] - phase[ramp_end - 2]  # continue at final rate
        phase[ramp_end:] += np.cumsum(np.full(n_post, max_phase_drift_rate))

    # AGC: only tiny random perturbation — stealth attack
    agc[onset:] += rng.normal(0, noise_std_agc * 0.12, size=n_samples - onset)
    label[onset:] = 1

    cn0 = _cn0_from_agc(agc, rng)
    df = pd.DataFrame({
        "time_s": t, "carrier_phase": phase, "agc": agc,
        "cn0_db_hz": cn0, "attack_label": label,
    })
    meta = {
        "data_source": _SYNTHETIC_TAG,
        "scenario": "stealth_spoofing",
        "onset_sample": onset,
        "ramp_duration_samples": ramp_duration,
        "max_phase_drift_rate_rad_per_sample": max_phase_drift_rate,
        "seed": seed,
    }
    log.info("Generated stealth-spoofing scenario: %d samples, onset=%d, seed=%d",
             n_samples, onset, seed)
    return df, meta


# ---------------------------------------------------------------------------
# Scenario suite
# ---------------------------------------------------------------------------

ALL_SCENARIOS = {
    "abrupt_spoofing":        generate_abrupt_spoofing,
    "gradual_spoofing":       generate_gradual_spoofing,
    "benign_multipath":       generate_benign_multipath,
    "agc_variation_only":     generate_agc_variation_only,
    "phase_anomaly_only":     generate_phase_anomaly_only,
    "rayleigh_fading":        generate_rayleigh_fading,
    "stealth_spoofing":       generate_stealth_spoofing,
}


def generate_all_scenarios(
    n_samples: int = 10_000,
    seeds: list[int] | None = None,
) -> dict[str, list[tuple[pd.DataFrame, dict[str, Any]]]]:
    """
    Generate all scenarios for multiple random seeds.

    Returns a dict mapping scenario name → list of (df, meta) tuples, one per seed.
    """
    if seeds is None:
        seeds = [42]
    results: dict[str, list] = {name: [] for name in ALL_SCENARIOS}
    for name, generator in ALL_SCENARIOS.items():
        for seed in seeds:
            df, meta = generator(n_samples=n_samples, seed=seed)
            results[name].append((df, meta))
    return results
