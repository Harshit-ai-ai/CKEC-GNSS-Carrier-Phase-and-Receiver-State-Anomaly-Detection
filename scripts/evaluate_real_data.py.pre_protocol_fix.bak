"""
scripts/evaluate_real_data.py   (v2 - provenance-aware, controlled comparisons)
=============================================================================
Evaluate detectors on the processed scenario CSVs in data/processed/real/.

DATA PROVENANCE
---------------
Every output row carries a ``data_source`` value.  It is taken from a
``data_source`` column in each input CSV if present, otherwise from the
``--data-source`` CLI option (default: UNVERIFIED).  Whatever the data is
(field recordings or simulation), the label travels with the results, and the
banner printed at start/end says so.  Never describe results from SYNTHETIC or
UNVERIFIED data as field-trial validation.

What changed vs v1
------------------
1. Benign calibration statistics use *benign-labelled* samples only (v1 used the
   first 60 s regardless of labels, which can include attack samples).
2. Level-based features (z of smoothed AGC / C/N0 level) added next to the
   first-difference features; a level step is invisible to a first difference.
3. C/N0 is a first-class secondary modality: D4/D6 are run with BOTH AGC and
   C/N0, and compared against the best single-channel baseline (D3c).
4. D4 and D6 use IDENTICAL weights (1,1) so the cross-modal term is the only
   difference.  Legacy D4/D6 (v1 settings) are kept for continuity.
5. Control detectors: D6 with the secondary channel circularly shifted before
   computing C_t.  If real C_t does not beat this control, the gain is not
   coming from coupling.
6. Block bootstrap (autocorrelation-aware) replaces the i.i.d. bootstrap, with
   a correct two-sided p-value; default 2000 iterations.
7. Scenario-level paired Wilcoxon tests with Holm correction, overall and per
   attack class (jamming / meaconing / spoofing / spoofing+jamming).
8. ROC-AUC / PR-AUC / detection delay reported next to F1; ``far_resolvable``
   flags scenarios whose benign test set is too small to measure the target FAR.

Outputs
-------
  <output>                      per-scenario results
  <output stem>_by_detector.csv mean metrics per detector (overall and per class)
  <output stem>_paired_tests.csv scenario-level paired tests, Holm-adjusted
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import click
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

from ckec.detectors import (
    AGCOnlyDetector,
    BaseDetector,
    CN0Detector,
    FullCKECDetector,
    JointDetector,
    PhaseOnlyDetector,
    RawRatioDetector,
)
from ckec.features import cross_modal_consistency
from ckec.preprocessing import smooth_agc, unwrap_phase
from ckec.utils import load_config, set_seed, setup_logging


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

class ColumnAbsDetector(BaseDetector):
    """score = |column|  (used for level-based AGC / C/N0 detectors)."""

    def __init__(self, column: str, name: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.column = column
        self.name = name

    def score(self, X: pd.DataFrame) -> pd.Series:
        self._require_columns(X, self.column)
        return X[self.column].abs().rename(self.name)


def wilson_ci(successes: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion."""
    if total == 0:
        return (0.0, 1.0)
    p_hat = successes / total
    z = stats.norm.ppf(1 - alpha / 2)
    denom = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denom
    spread = z * np.sqrt(p_hat * (1 - p_hat) / total + z**2 / (4 * total**2)) / denom
    return (max(0.0, center - spread), min(1.0, center + spread))


def attack_class(name: str) -> str:
    """Attack class parsed from the scenario name."""
    n = name.lower()
    if "spoofing_+_jamming" in n or "spoofing+jamming" in n:
        return "spoof_jam"
    if "meaconing" in n:
        return "meaconing"
    if "spoofing" in n:
        return "spoofing"
    if "jamming" in n:
        return "jamming"
    return "other"


def _benign_stats(series: pd.Series, labels: np.ndarray, n: int) -> tuple[float, float]:
    """Mean/std from the first ``n`` BENIGN-labelled finite samples.

    Falls back to the first ``n`` finite samples (any label) only if fewer than
    5 benign samples exist.
    """
    vals = series.to_numpy(dtype=float)
    finite = np.isfinite(vals)
    benign = vals[finite & (labels == 0)][:n]
    if len(benign) < 5:
        benign = vals[finite][:n]
    if len(benign) < 2:
        return 0.0, 1.0
    return float(np.mean(benign)), float(np.std(benign, ddof=1))


def _z(series: pd.Series, labels: np.ndarray, n: int, eps: float) -> pd.Series:
    mu, sd = _benign_stats(series, labels, n)
    return (series - mu) / (sd + eps)


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------

def prepare_features(
    df: pd.DataFrame,
    window_corr: int = 20,
    epsilon: float = 1e-4,
    cal_seconds: float = 60.0,
) -> pd.DataFrame:
    """Build the per-sample feature table for one scenario."""
    d = df.reset_index(drop=True).copy()
    labels = d["attack_label"].to_numpy()

    dt = d["time_s"].diff().median() if "time_s" in d.columns else 1.0
    if not dt or np.isnan(dt) or dt <= 0:
        dt = 1.0
    cal_n = max(20, int(cal_seconds / dt))
    cal_n = min(cal_n, max(20, len(d) // 4))

    feat = pd.DataFrame(index=d.index)

    # Phase (first difference)
    phase = d["carrier_phase"]
    if phase.notna().sum() > 10:
        d_phi = unwrap_phase(phase).diff()
        feat["z_phi"] = _z(d_phi, labels, cal_n, epsilon)
        feat["delta_phi"] = d_phi
    else:
        feat["z_phi"] = np.nan
        feat["delta_phi"] = np.nan

    # Hardware AGC: difference and level
    secondaries: list[str] = []
    if "hardware_agc" in d.columns and d["hardware_agc"].notna().sum() > 10:
        agc_s = smooth_agc(d["hardware_agc"], window=5)
        d_agc = agc_s.diff()
        feat["z_agc"] = _z(d_agc, labels, cal_n, epsilon)
        feat["z_agc_lvl"] = _z(agc_s, labels, cal_n, epsilon)
        feat["delta_agc"] = d_agc
        secondaries.append("agc")

    # C/N0: difference and level
    if "cn0_db_hz" in d.columns and d["cn0_db_hz"].notna().sum() > 10:
        cn0 = d["cn0_db_hz"]
        cn0_s = cn0.rolling(5, min_periods=1).mean()
        feat["z_cn0"] = _z(cn0.diff(), labels, cal_n, epsilon)
        feat["z_cn0_lvl"] = _z(cn0_s, labels, cal_n, epsilon)
        secondaries.append("cn0")

    # Cross-modal correlation (real) and shifted-secondary control
    w = max(3, window_corr)
    lag = max(5 * w, len(d) // 3)
    for sec in secondaries:
        z_sec = feat[f"z_{sec}"]
        feat[f"corr_{sec}"] = cross_modal_consistency(feat["z_phi"], z_sec, window=w)
        shifted = pd.Series(np.roll(z_sec.to_numpy(), lag), index=z_sec.index)
        feat[f"corr_ctrl_{sec}"] = cross_modal_consistency(feat["z_phi"], shifted, window=w)

    feat["attack_label"] = labels
    if "time_s" in d.columns:
        feat["time_s"] = d["time_s"].to_numpy()
    feat.attrs["secondaries"] = secondaries
    return feat.dropna(subset=["z_phi"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def split_indices(feat_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return positional (val_idx, test_idx).

    Val = benign calibration block; test = everything after it (benign tail +
    attack + post-attack).  Same logic as v1's ``temporal_split``.
    """
    n = len(feat_df)
    labels = feat_df["attack_label"].to_numpy()
    benign = labels == 0
    attack = ~benign

    if not benign.any() or not attack.any():
        return np.arange(int(n * 0.6), int(n * 0.8)), np.arange(int(n * 0.8), n)

    first_attack = int(np.where(attack)[0][0])
    if first_attack >= 60:
        train_end = int(first_attack * 0.4)
        val_end = int(first_attack * 0.7)
        return np.arange(train_end, val_end), np.arange(val_end, n)

    benign_idx = np.where(benign)[0]
    cal_end = int(len(benign_idx) * 0.4)
    cal = benign_idx[:cal_end]
    mid = len(cal) // 2
    val = cal[mid:]
    cal_set = set(cal.tolist())
    test = np.array([i for i in range(n) if i not in cal_set])
    return val, test


def temporal_split(feat_df: pd.DataFrame):
    """Backward-compatible wrapper returning (train, val, test) DataFrames."""
    val_idx, test_idx = split_indices(feat_df)
    return feat_df.iloc[:0], feat_df.iloc[val_idx], feat_df.iloc[test_idx]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _f1(y_t: np.ndarray, y_p: np.ndarray) -> float:
    tp = int(((y_t == 1) & (y_p == 1)).sum())
    fp = int(((y_t == 0) & (y_p == 1)).sum())
    fn = int(((y_t == 1) & (y_p == 0)).sum())
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return 2 * prec * rec / (prec + rec)


def block_bootstrap_f1_diff(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    n_iter: int = 2000,
    block_len: int = 30,
    seed: int = 42,
) -> dict:
    """Moving-block paired bootstrap for F1(A) - F1(B).

    Resamples contiguous blocks so serial correlation is preserved.  The
    two-sided p-value is 2*min(P(d*<=0), P(d*>=0)) with +1 smoothing.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    L = int(max(1, min(block_len, n)))
    n_blocks = math.ceil(n / L)
    offs = np.arange(L)

    observed = _f1(y_true, y_pred_a) - _f1(y_true, y_pred_b)
    deltas = np.empty(n_iter)
    for i in range(n_iter):
        starts = rng.integers(0, n - L + 1, size=n_blocks)
        idx = (starts[:, None] + offs).ravel()[:n]
        yt = y_true[idx]
        deltas[i] = _f1(yt, y_pred_a[idx]) - _f1(yt, y_pred_b[idx])

    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p = 2 * min((np.sum(deltas <= 0) + 1) / (n_iter + 1), (np.sum(deltas >= 0) + 1) / (n_iter + 1))
    return {
        "delta_f1": float(observed),
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "p_value": float(min(1.0, p)),
        "significant": bool(lo > 0 or hi < 0),
    }


def _metrics(
    y_test: np.ndarray,
    scores: np.ndarray,
    y_pred: np.ndarray,
    time_s: np.ndarray | None,
) -> dict:
    tp = int(((y_test == 1) & (y_pred == 1)).sum())
    fp = int(((y_test == 0) & (y_pred == 1)).sum())
    fn = int(((y_test == 1) & (y_pred == 0)).sum())
    tn = int(((y_test == 0) & (y_pred == 0)).sum())
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    far = fp / (fp + tn) if (fp + tn) else 0.0
    lo, hi = wilson_ci(fp, fp + tn)

    roc = pr = np.nan
    ok = np.isfinite(scores)
    if ok.sum() > 0 and len(np.unique(y_test[ok])) == 2:
        roc = roc_auc_score(y_test[ok], scores[ok])
        pr = average_precision_score(y_test[ok], scores[ok])

    delay = np.nan
    if (y_test == 1).any():
        a0 = int(np.argmax(y_test == 1))
        alarms = np.where((y_pred == 1) & (np.arange(len(y_pred)) >= a0))[0]
        if len(alarms):
            delay = float(time_s[alarms[0]] - time_s[a0]) if time_s is not None else float(alarms[0] - a0)

    return {
        "recall": recall, "precision": precision, "f1": f1, "far": far,
        "far_ci": f"[{lo:.4f}, {hi:.4f}]", "roc_auc": roc, "pr_auc": pr,
        "delay_s": delay,
    }


# ---------------------------------------------------------------------------
# Per-scenario evaluation
# ---------------------------------------------------------------------------

COMPARISONS = [
    # (A, B, description)
    ("D6_full_ckec_cn0", "D4_joint_cn0",        "coupling term vs equal-weight joint (C/N0)"),
    ("D6_full_ckec_agc", "D4_joint_agc",        "coupling term vs equal-weight joint (AGC)"),
    ("D6_full_ckec_cn0", "D6ctrl_shifted_cn0",  "real C_t vs shifted-secondary control (C/N0)"),
    ("D6_full_ckec_agc", "D6ctrl_shifted_agc",  "real C_t vs shifted-secondary control (AGC)"),
    ("D6_full_ckec_cn0", "D3c_cn0_only",        "D6(C/N0) vs best single-channel baseline"),
    ("D3c_cn0_only_lvl", "D3c_cn0_only",        "C/N0 level vs C/N0 difference"),
    ("D3a_agc_only_lvl", "D3a_agc_only",        "AGC level vs AGC difference"),
    ("D6_full_ckec",     "D4_joint",            "legacy D6 vs legacy D4 (v1 settings)"),
    ("D6_full_ckec",     "D2_phase_only",       "legacy D6 vs phase-only"),
]


def evaluate_scenario(
    scenario_name: str,
    raw_df: pd.DataFrame,
    target_far: float = 0.01,
    epsilon: float = 1e-4,
    window_corr: int = 20,
    w_c: float = 1.0,
    bootstrap_iter: int = 2000,
    block_len: int = 30,
    min_benign_for_far: int = 200,
    seed: int = 42,
) -> dict:
    feat = prepare_features(raw_df, window_corr=window_corr, epsilon=epsilon)
    secs: list[str] = feat.attrs.get("secondaries", [])
    val_idx, test_idx = split_indices(feat)

    y_val = (feat["attack_label"].to_numpy()[val_idx] != 0).astype(int)
    y_test = (feat["attack_label"].to_numpy()[test_idx] != 0).astype(int)
    time_test = feat["time_s"].to_numpy()[test_idx] if "time_s" in feat.columns else None

    res: dict = {
        "scenario": scenario_name,
        "attack_class": attack_class(scenario_name),
        "n_test": int(len(test_idx)),
        "n_benign_test": int((y_test == 0).sum()),
        "n_attack_test": int((y_test == 1).sum()),
        "attack_prevalence_test": float(y_test.mean()) if len(y_test) else np.nan,
        "far_resolvable": bool((y_test == 0).sum() >= min_benign_for_far),
    }

    # Frames: base + (real C_t) + (shifted-control C_t) per secondary
    frames: dict[str, pd.DataFrame] = {"base": feat}
    for s in secs:
        frames[f"c_{s}"] = feat.assign(cross_modal_corr=feat[f"corr_{s}"].fillna(1.0))
        frames[f"ctrl_{s}"] = feat.assign(cross_modal_corr=feat[f"corr_ctrl_{s}"].fillna(1.0))

    legacy_sec = "agc" if "agc" in secs else ("cn0" if "cn0" in secs else None)

    specs: list[tuple[str, BaseDetector, str]] = []
    if "delta_agc" in feat.columns:
        specs.append(("D1_raw_ratio", RawRatioDetector(secondary_col="delta_agc", epsilon=epsilon), "base"))
    specs.append(("D2_phase_only", PhaseOnlyDetector(), "base"))
    if "agc" in secs:
        specs.append(("D3a_agc_only", AGCOnlyDetector(), "base"))
        specs.append(("D3a_agc_only_lvl", ColumnAbsDetector("z_agc_lvl", "D3a_agc_only_lvl"), "base"))
    if "cn0" in secs:
        specs.append(("D3c_cn0_only", CN0Detector(), "base"))
        specs.append(("D3c_cn0_only_lvl", ColumnAbsDetector("z_cn0_lvl", "D3c_cn0_only_lvl"), "base"))
    if legacy_sec:  # v1 settings kept for continuity
        specs.append(("D4_joint", JointDetector(secondary_col=f"z_{legacy_sec}"), "base"))
        specs.append(("D6_full_ckec",
                      FullCKECDetector(secondary_col=f"z_{legacy_sec}", w_phi=1.0, w_secondary=1.0, w_c=1.0),
                      f"c_{legacy_sec}"))
    for s in secs:  # controlled comparison: equal weights, only C_t differs
        col = f"z_{s}"
        specs.append((f"D4_joint_{s}", JointDetector(secondary_col=col, w_phi=1.0, w_secondary=1.0), "base"))
        specs.append((f"D6_full_ckec_{s}",
                      FullCKECDetector(secondary_col=col, w_phi=1.0, w_secondary=1.0, w_c=w_c), f"c_{s}"))
        specs.append((f"D6ctrl_shifted_{s}",
                      FullCKECDetector(secondary_col=col, w_phi=1.0, w_secondary=1.0, w_c=w_c), f"ctrl_{s}"))

    preds: dict[str, np.ndarray] = {}
    for name, det, fkey in specs:
        frame = frames[fkey]
        try:
            det.fit(frame.iloc[val_idx], y_val, target_far=target_far)
            scores = det.score(frame.iloc[test_idx]).to_numpy(dtype=float)
            y_pred = (scores > det._threshold).astype(int)  # NaN -> 0
            preds[name] = y_pred
            m = _metrics(y_test, scores, y_pred, time_test)
            for k, v in m.items():
                res[f"{name}_{k}"] = v
            res[f"{name}_status"] = "OK"
        except Exception as e:  # keep going; record failure
            res[f"{name}_status"] = f"FAILED: {e}"
            res[f"{name}_f1"] = np.nan

    for a, b, _ in COMPARISONS:
        if a in preds and b in preds:
            bs = block_bootstrap_f1_diff(
                y_test, preds[a], preds[b], n_iter=bootstrap_iter, block_len=block_len, seed=seed
            )
            tag = f"bootstrap_{a}_vs_{b}"
            res[f"{tag}_delta_f1"] = bs["delta_f1"]
            res[f"{tag}_ci"] = f"[{bs['ci_lower']:.4f}, {bs['ci_upper']:.4f}]"
            res[f"{tag}_p_value"] = bs["p_value"]
            res[f"{tag}_significant"] = bs["significant"]
    return res


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def holm(p: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (NaN-safe)."""
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.where(np.isfinite(p))[0]
    if len(ok) == 0:
        return out
    order = ok[np.argsort(p[ok])]
    m = len(order)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        out[i] = min(1.0, running)
    return out


def detector_names(df: pd.DataFrame) -> list[str]:
    return sorted({c[: -len("_status")] for c in df.columns if c.endswith("_status")})


def summarise_by_detector(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [("all", df)] + [(c, g) for c, g in df.groupby("attack_class")]
    for gname, g in groups:
        for det in detector_names(df):
            if f"{det}_f1" not in g.columns:
                continue
            rows.append({
                "group": gname, "detector": det, "n_scenarios": int(g[f"{det}_f1"].notna().sum()),
                "mean_roc_auc": g.get(f"{det}_roc_auc", pd.Series(dtype=float)).mean(),
                "mean_pr_auc": g.get(f"{det}_pr_auc", pd.Series(dtype=float)).mean(),
                "mean_f1": g[f"{det}_f1"].mean(),
                "median_delay_s": g.get(f"{det}_delay_s", pd.Series(dtype=float)).median(),
            })
    return pd.DataFrame(rows)


def paired_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [("all", df)] + [(c, g) for c, g in df.groupby("attack_class")]
    for metric in ("roc_auc", "f1"):
        start = len(rows)
        for a, b, desc in COMPARISONS:
            ca, cb = f"{a}_{metric}", f"{b}_{metric}"
            if ca not in df.columns or cb not in df.columns:
                continue
            for gname, g in groups:
                d = (g[ca] - g[cb]).dropna().to_numpy()
                if len(d) < 3:
                    continue
                if np.allclose(d, 0):
                    p = 1.0
                else:
                    try:
                        p = float(stats.wilcoxon(d, zero_method="wilcox").pvalue)
                    except ValueError:
                        p = np.nan
                rows.append({
                    "metric": metric, "A": a, "B": b, "comparison": desc, "group": gname,
                    "n_scenarios": len(d), "mean_diff": float(d.mean()), "median_diff": float(np.median(d)),
                    "n_A_better": int((d > 0).sum()), "n_B_better": int((d < 0).sum()),
                    "wilcoxon_p": p,
                })
        if len(rows) > start:  # Holm across all tests for this metric
            adj = holm(np.array([r["wilcoxon_p"] for r in rows[start:]]))
            for r, q in zip(rows[start:], adj):
                r["holm_p"] = float(q)
                r["significant_holm_0.05"] = bool(q < 0.05) if np.isfinite(q) else False
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _banner(sources: set[str]) -> str:
    s = ", ".join(sorted(sources)) or "UNVERIFIED"
    line = "=" * 80
    msg = f"{line}\nDATA SOURCE: {s}\n"
    if any(x.upper() != "REAL" for x in sources) or not sources:
        msg += ("NOTE: data is not confirmed REAL. Do not describe these results as field-trial\n"
                "validation. Label every table/figure/claim with the source above.\n")
    return msg + line


@click.command()
@click.option("--config", default="config.yaml", show_default=True)
@click.option("--data-dir", default=None, help="Directory with processed CSVs (default: data/processed/real)")
@click.option("--target-far", default=0.01, type=float, show_default=True)
@click.option("--output", default="reports/real_evaluation_results.csv", show_default=True)
@click.option("--bootstrap-iter", default=2000, type=int, show_default=True)
@click.option("--block-len", default=30, type=int, show_default=True, help="Bootstrap block length (samples)")
@click.option("--w-c", "w_c", default=1.0, type=float, show_default=True, help="Coupling weight for D6 variants")
@click.option("--window-corr", default=20, type=int, show_default=True)
@click.option("--min-benign-for-far", default=200, type=int, show_default=True,
              help="Min benign test samples for the target FAR to be resolvable")
@click.option("--data-source", default="UNVERIFIED",
              type=click.Choice(["REAL", "SYNTHETIC", "UNVERIFIED"], case_sensitive=False),
              show_default=True, help="Used when a CSV has no 'data_source' column")
def main(config, data_dir, target_far, output, bootstrap_iter, block_len, w_c, window_corr,
         min_benign_for_far, data_source):
    """Evaluate detectors on processed scenario CSVs with provenance-stamped results."""
    cfg = load_config(config)
    log = setup_logging(level=cfg["logging"]["level"], log_file=cfg["logging"]["file"])
    seed = cfg["experiment"].get("seed", 42)
    set_seed(seed)

    data_path = Path(data_dir) if data_dir else Path("data/processed/real")
    if not data_path.exists():
        log.error("Data directory not found: %s", data_path)
        return
    csv_files = [f for f in sorted(data_path.glob("*.csv")) if f.name != "ingestion_summary.csv"]
    if not csv_files:
        log.error("No CSV files found in %s", data_path)
        return

    print(_banner({data_source.upper()}))
    print(f"Found {len(csv_files)} scenario files in {data_path}\n")

    all_results: list[dict] = []
    sources: set[str] = set()
    for f in csv_files:
        print(f"--- {f.stem} ---")
        try:
            df = pd.read_csv(f)
            src = str(df["data_source"].iloc[0]).upper() if "data_source" in df.columns and len(df) else data_source.upper()
            sources.add(src)
            n_b = int((df["attack_label"] == 0).sum())
            n_a = int((df["attack_label"] != 0).sum())
            print(f"  {len(df)} samples ({n_b} benign, {n_a} attack), source={src}")
            if n_b < 20 or n_a < 10:
                print("  [SKIP] too few benign or attack samples")
                continue

            r = evaluate_scenario(
                f.stem, df, target_far=target_far, window_corr=window_corr, w_c=w_c,
                bootstrap_iter=bootstrap_iter, block_len=block_len,
                min_benign_for_far=min_benign_for_far, seed=seed,
            )
            r["data_source"] = src
            if not r["far_resolvable"]:
                print(f"  [WARN] only {r['n_benign_test']} benign test samples: FAR={target_far:.0%} not resolvable")
            for det in ("D2_phase_only", "D3a_agc_only_lvl", "D3c_cn0_only", "D3c_cn0_only_lvl",
                        "D4_joint_cn0", "D6_full_ckec_cn0", "D6ctrl_shifted_cn0"):
                if f"{det}_f1" in r and pd.notna(r[f"{det}_f1"]):
                    print(f"  {det:22s} AUC={r.get(f'{det}_roc_auc', np.nan):.3f} "
                          f"F1={r[f'{det}_f1']:.3f} FAR={r.get(f'{det}_far', np.nan):.3f}")
            all_results.append(r)
        except Exception as e:
            import traceback
            print(f"  [ERROR] {e}")
            traceback.print_exc()

    if not all_results:
        print("No scenarios were successfully evaluated.")
        return

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    res_df = pd.DataFrame(all_results)
    res_df.to_csv(out, index=False)

    by_det = summarise_by_detector(res_df)
    tests = paired_tests(res_df)
    by_det.to_csv(out.with_name(out.stem + "_by_detector.csv"), index=False)
    tests.to_csv(out.with_name(out.stem + "_paired_tests.csv"), index=False)

    print("\n" + _banner(sources))
    print(f"{len(all_results)} scenarios evaluated -> {out}")
    print("\n--- MEAN METRICS (all scenarios) ---")
    show = by_det[by_det["group"] == "all"].sort_values("mean_roc_auc", ascending=False)
    print(show[["detector", "n_scenarios", "mean_roc_auc", "mean_pr_auc", "mean_f1", "median_delay_s"]]
          .round(3).to_string(index=False))

    print("\n--- SCENARIO-LEVEL PAIRED TESTS, ROC-AUC, group=all (Holm-adjusted) ---")
    t = tests[(tests["metric"] == "roc_auc") & (tests["group"] == "all")]
    if len(t):
        print(t[["A", "B", "n_scenarios", "mean_diff", "n_A_better", "n_B_better", "holm_p"]]
              .round(4).to_string(index=False))
        ctrl = t[t["B"].str.startswith("D6ctrl")]
        if len(ctrl) and not ctrl["significant_holm_0.05"].any():
            print("\n  NOTE: real C_t does not significantly beat the shifted-secondary control;"
                  "\n  the coupling term's gain cannot be attributed to cross-modal coupling.")
        base = t[t["B"] == "D3c_cn0_only"]
        if len(base) and (base["mean_diff"] <= 0).all():
            print("\n  NOTE: D6 does not beat the C/N0-only baseline on mean AUC.")


if __name__ == "__main__":
    main()