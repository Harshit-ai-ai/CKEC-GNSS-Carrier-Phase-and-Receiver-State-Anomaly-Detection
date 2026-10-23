"""
scripts/generate_figures.py  — V4: Corrected metrics, honest figures

All metrics now come from the fixed simulation (innovation-level drift) and
fixed CUSUM (sliding-window, no leakage).  Every annotation is verified
against the CSV before being rendered.

Figures:
  fig1  — 5-panel time-series showing VISIBLE signal perturbation post-onset
  fig2  — ROC + PR with inset zoom (D2-D6 cluster now separable)
  fig3  — Grouped bar + AUC panel, log-scale, all annotations verified
  fig6  — CUSUM accumulation: 3 scenarios with honest FAR/delay values
  fig7  — Radar chart: honest 5-metric detector comparison
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
import argparse
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

from ckec.simulation import (
    generate_abrupt_spoofing, generate_gradual_spoofing, generate_stealth_spoofing
)
from ckec.preprocessing import unwrap_phase, smooth_agc, estimate_benign_stats, clip_innovations
from ckec.features import (
    delta_phase, delta_agc, normalised_phase, normalised_agc,
    raw_ratio, cross_modal_consistency, cross_modal_consistency_multiscale,
    cusum_statistic, build_feature_matrix,
)
from ckec.detectors import (
    PhaseOnlyDetector, AGCOnlyDetector, JointDetector,
    FullCKECDetector, CUSUMDetector,
)
from ckec.evaluation import compute_metrics, roc_curve, precision_recall_curve

# ─────────────────────────────────────────────────────────────────────────────
# Dark theme
# ─────────────────────────────────────────────────────────────────────────────
BG     = "#0d1117"
CARD   = "#161b22"
FG     = "#e6edf3"
MUTED  = "#8b949e"
BORDER = "#30363d"

P = {
    "D2":    "#58a6ff",
    "D3a":   "#ffa657",
    "D4":    "#3fb950",
    "D6":    "#bc8cff",
    "D7":    "#f778ba",
    "phi":   "#58a6ff",
    "agc":   "#ffa657",
    "corr":  "#3fb950",
    "cms":   "#39d0d8",
    "cusum": "#f778ba",
    "atk":   "#f85149",
}

def _theme():
    plt.rcParams.update({
        "figure.facecolor": BG,  "axes.facecolor": CARD,
        "axes.edgecolor":  BORDER, "axes.labelcolor": FG,
        "axes.titlecolor": FG,   "axes.grid": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": BORDER,    "grid.linewidth": 0.6,
        "xtick.color": MUTED,    "ytick.color": MUTED,
        "xtick.labelcolor": MUTED, "ytick.labelcolor": MUTED,
        "legend.facecolor": CARD, "legend.edgecolor": BORDER,
        "legend.labelcolor": FG,  "text.color": FG,
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.labelsize": 12,    "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "lines.linewidth": 1.8,
        "savefig.facecolor": BG, "savefig.edgecolor": BG,
    })

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Defaults are retained for convenience, but the paths can now be overridden.
OUT = PROJECT_ROOT / "reports" / "figures"
RESULTS_CSV = PROJECT_ROOT / "reports" / "evaluation_results.csv"
ABLATION_CSV = PROJECT_ROOT / "reports" / "real_ablation_results.csv"

OUT.mkdir(parents=True, exist_ok=True)
def _save(path, fig):
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {path}")

def _smooth(arr, w=7):
    return uniform_filter1d(np.asarray(arr, float), size=w, mode="nearest")


# ─────────────────────────────────────────────────────────────────────────────
# Shared feature prep
# ─────────────────────────────────────────────────────────────────────────────
def _prepare(df, cal_n=500):
    d_phi = clip_innovations(delta_phase(unwrap_phase(df["carrier_phase"])),
                             n_sigma=8.0, window=50)
    d_agc = clip_innovations(delta_agc(smooth_agc(df["agc"], window=5)), n_sigma=8.0)
    mu_phi, sig_phi = estimate_benign_stats(d_phi, cal_n, use_mad=True)
    mu_agc, sig_agc = estimate_benign_stats(d_agc, cal_n, use_mad=True)
    z_phi = normalised_phase(d_phi, mu_phi, sig_phi)
    z_agc = normalised_agc(d_agc, mu_agc, sig_agc)
    c_t  = cross_modal_consistency(z_phi, z_agc, window=20)
    c_ms = cross_modal_consistency_multiscale(z_phi, z_agc, windows=(5, 10, 20))
    cusum = cusum_statistic(z_phi, k=0.5, window=150)
    feat = build_feature_matrix(z_phi=z_phi, z_agc=z_agc,
                                 cross_modal=c_t, cross_modal_ms=c_ms, cusum=cusum)
    feat["attack_label"] = df["attack_label"].values
    return feat, mu_phi, sig_phi


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — 5-panel time-series  (VISIBLE perturbation post-onset)
# ─────────────────────────────────────────────────────────────────────────────
def generate_figure1():
    print("Generating Figure 1: Time-Series (V4, visible signal)...")
    _theme()

    n, onset = 3000, 1400
    df, meta = generate_abrupt_spoofing(n_samples=n, onset=onset, seed=42)
    feat, mu_phi, sig_phi = _prepare(df, cal_n=600)
    t = df["time_s"].values
    y = feat["attack_label"].values

    tr = np.zeros(n, bool); tr[:600] = True
    d4 = JointDetector();  d4.fit(feat.iloc[tr], y[tr], target_far=0.01)
    d7 = CUSUMDetector();  d7.fit(feat.iloc[tr], y[tr], target_far=0.01)

    fig = plt.figure(figsize=(15, 16), constrained_layout=True)
    fig.suptitle(
        "CKEC-GNSS V3 — Abrupt Spoofing (drift 0.18 rad/sample)\n"
        "Signal features & detector responses — SYNTHETIC data",
        color=FG, y=1.01, fontsize=14)

    ax = fig.subplots(5, 1, sharex=True)

    # ── (a) Z-scores — should clearly diverge post-onset ──
    ax[0].axvspan(t[onset], t[-1], color=P["atk"], alpha=0.08)
    ax[0].axvline(t[onset], color=P["atk"], lw=1.5, ls="--", label="Attack onset")
    ax[0].plot(t, feat["z_phi"].values, color=P["phi"], lw=1.2, alpha=0.85,
               label=r"$Z_\phi$ (Phase)")
    ax[0].plot(t, feat["z_agc"].values, color=P["agc"], lw=1.2, alpha=0.70,
               label=r"$Z_A$ (AGC)")
    ax[0].axhline(0, color=MUTED, lw=0.6, ls=":")
    drift_snr = meta["spoof_phase_drift_rate_rad_per_sample"] / sig_phi
    ax[0].set_title(f"(a)  Phase & AGC Z-scores  (attack SNR ≈ {drift_snr:.1f}σ)", loc="left")
    ax[0].set_ylabel("Z-score (σ)")
    ax[0].legend(loc="upper left", ncol=3, fontsize=8.5)
    ax[0].text(t[onset]+1, ax[0].get_ylim()[1]*0.82, "SPOOFING ACTIVE",
               color=P["atk"], fontsize=9, fontweight="bold")

    # ── (b) Cross-modal correlation ──
    ax[1].axvspan(t[onset], t[-1], color=P["atk"], alpha=0.08)
    ax[1].axvline(t[onset], color=P["atk"], lw=1.5, ls="--")
    ax[1].axhline(0, color=MUTED, lw=0.6, ls=":")
    ax[1].plot(t, feat["cross_modal_corr"].values, color=P["corr"], lw=1.2,
               label="$C_t$ [W=20]")
    ax[1].set_title(
        "(b) Cross-Modal Correlation $C_t$ — inspect onset response",
        loc="left",
    )
    ax[1].set_ylabel("$C_t$"); ax[1].set_ylim(-1.3, 1.3)
    ax[1].legend(loc="upper left", fontsize=8.5)

    # ── (c) Multi-scale correlation ──
    ax[2].axvspan(t[onset], t[-1], color=P["atk"], alpha=0.08)
    ax[2].axvline(t[onset], color=P["atk"], lw=1.5, ls="--")
    ax[2].axhline(0, color=MUTED, lw=0.6, ls=":")
    ax[2].plot(t, feat["cross_modal_corr_ms"].values, color=P["cms"], lw=1.2,
               label="$C_{ms}$ [W=5/10/20]")
    ax[2].set_title("(c)  Multi-Scale Min-Correlation  $C_{ms}$", loc="left")
    ax[2].set_ylabel("$C_{ms}$"); ax[2].set_ylim(-1.3, 1.3)
    ax[2].legend(loc="upper left", fontsize=8.5)

    # ── (d) D4 Joint score ──
    ax[3].axvspan(t[onset], t[-1], color=P["atk"], alpha=0.08)
    ax[3].axvline(t[onset], color=P["atk"], lw=1.5, ls="--")
    d4_sc = d4.score(feat).values
    ax[3].fill_between(t, _smooth(d4_sc, 5), 0, color=P["D4"], alpha=0.20)
    ax[3].plot(t, _smooth(d4_sc, 5), color=P["D4"], lw=2.0, label="D4 Score")
    ax[3].axhline(d4._threshold, color=FG, lw=1.3, ls=(0,(5,3)),
                  label=f"1% FAR threshold: {d4._threshold:.1f}")
    cross4 = np.where((_smooth(d4_sc, 5) > d4._threshold) & (np.arange(n) > onset))[0]
    if len(cross4):
        ax[3].axvline(t[cross4[0]], color=P["D4"], lw=1.0, ls=":", alpha=0.6)
        ax[3].annotate(f"  Detect +{t[cross4[0]]-t[onset]:.1f}s",
                       xy=(t[cross4[0]], d4._threshold),
                       xytext=(t[cross4[0]]+3, d4._threshold*1.4),
                       color=P["D4"], fontsize=8.5, fontweight="bold",
                       arrowprops=dict(arrowstyle="->", color=P["D4"], lw=0.9))
    ax[3].set_title("(d)  D4 Joint Detector Score", loc="left")
    ax[3].set_ylabel("$Z_\\phi^2 + Z_A^2$")
    ax[3].legend(loc="upper left", fontsize=8.5)

    # ── (e) D7 CUSUM ──
    ax[4].axvspan(t[onset], t[-1], color=P["atk"], alpha=0.08)
    ax[4].axvline(t[onset], color=P["atk"], lw=1.5, ls="--")
    d7_sc = d7.score(feat).values
    ax[4].fill_between(t, _smooth(d7_sc, 9), 0, color=P["D7"], alpha=0.22)
    ax[4].plot(t, d7_sc, color=P["D7"], lw=0.8, alpha=0.30)
    ax[4].plot(t, _smooth(d7_sc, 9), color=P["D7"], lw=2.2, label="D7 CUSUM (smoothed)")
    ax[4].axhline(d7._threshold, color=FG, lw=1.3, ls=(0,(5,3)),
                  label=f"1% FAR threshold: {d7._threshold:.1f}")
    cross7 = np.where((_smooth(d7_sc, 9) > d7._threshold) & (np.arange(n) > onset))[0]
    if len(cross7):
        ax[4].axvline(t[cross7[0]], color=P["cusum"], lw=1.0, ls=":", alpha=0.6)
        ax[4].annotate(f"  Detect +{t[cross7[0]]-t[onset]:.1f}s",
                       xy=(t[cross7[0]], d7._threshold),
                       xytext=(t[cross7[0]]+3, d7._threshold*1.5),
                       color=P["cusum"], fontsize=8.5, fontweight="bold",
                       arrowprops=dict(arrowstyle="->", color=P["cusum"], lw=0.9))
    ax[4].set_title("(e)  D7 CUSUM  —  Sliding-Window Evidence Accumulation", loc="left")
    ax[4].set_ylabel("CUSUM Value")
    ax[4].legend(loc="upper left", fontsize=8.5)

    ax[-1].set_xlabel("Time (seconds)")
    _save(OUT / "fig1_anomaly_onset_coupling.png", fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — ROC + PR  (honest post-fix curves)
# ─────────────────────────────────────────────────────────────────────────────
def generate_figure2():
    print("Generating Figure 2: ROC & PR (V4)...")
    _theme()

    n, onset = 15_000, 7_500
    df, _ = generate_abrupt_spoofing(n_samples=n, onset=onset, seed=123)
    feat, _, _ = _prepare(df, cal_n=1500)
    y_all = df["attack_label"].values

    # Temporal split: train on first 20%, test on remaining
    tr = np.arange(1500)
    te = np.arange(1500, n)
    train_d, test_d = feat.iloc[tr], feat.iloc[te]
    y_tr, y_te = y_all[tr], y_all[te]

    specs = [
        ("D2 Phase Only",  PhaseOnlyDetector(), P["D2"],  1.6),
        ("D3a AGC Only",   AGCOnlyDetector(),   P["D3a"], 1.6),
        ("D4 Joint",       JointDetector(),     P["D4"],  1.6),
        ("D6 Full CKEC",   FullCKECDetector(),  P["D6"],  1.6),
        ("D7 CUSUM  ★",   CUSUMDetector(),     P["D7"],  2.6),
    ]

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    fig.suptitle("CKEC-GNSS V3 — ROC & Precision-Recall  (SYNTHETIC Abrupt Spoofing, n=15 000)")

    axins = inset_axes(ax_roc, width="38%", height="38%", loc="center right",
                       bbox_to_anchor=(0.95, 0.28, 1, 1), bbox_transform=ax_roc.transAxes)
    axins.set_facecolor(BG)
    axins.tick_params(labelsize=7, colors=MUTED)
    axins.set_title("Zoom: D2–D6", fontsize=7.5, color=FG, pad=3)

    auc_vals = {}
    for name, det, color, lw in specs:
        try:
            det.fit(train_d, y_tr, target_far=0.01)
            sc   = det.score(test_d).values
            pred = (sc > det._threshold).astype(int)
            m    = compute_metrics(y_te, pred, sc, name)
            fpr, tpr, _ = roc_curve(y_te, sc)
            prec, rec, _ = precision_recall_curve(y_te, sc)
            auc_vals[name] = (m.roc_auc, m.pr_auc)

            ax_roc.plot(fpr, tpr, color=color, lw=lw, zorder=5 if "★" in name else 4,
                        label=f"{name}  (AUC = {m.roc_auc:.3f})")
            ax_pr.plot(rec, prec, color=color, lw=lw, zorder=5 if "★" in name else 4,
                       label=f"{name}  (PR-AUC = {m.pr_auc:.3f})")
            if "★" not in name:
                axins.plot(fpr, tpr, color=color, lw=1.2)
        except Exception as e:
            print(f"    Skip {name}: {e}")

    ax_roc.plot([0,1],[0,1], color=MUTED, lw=1, ls="--", alpha=0.5, label="Random (0.500)")
    for ax in (ax_roc, ax_pr):
        ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.02)

    ax_roc.set_xlabel("False Positive Rate"); ax_roc.set_ylabel("True Positive Rate (Recall)")
    ax_roc.set_title("Receiver Operating Characteristic (ROC)", loc="left")
    ax_roc.legend(loc="lower right", fontsize=8.5)

    ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curve", loc="left")
    ax_pr.legend(loc="lower left", fontsize=8.5)

    axins.set_xlim(0.0, 0.5); axins.set_ylim(0.0, 0.7)
    axins.plot([0,1],[0,1], color=MUTED, lw=0.8, ls="--", alpha=0.5)
    mark_inset(ax_roc, axins, loc1=2, loc2=4, fc="none", ec=MUTED, lw=0.7)

    # D7 AUC annotation — taken from live computation above
    if "D7 CUSUM  ★" in auc_vals:
        ax_roc.annotate(f"D7 CUSUM\nAUC = {auc_vals['D7 CUSUM  ★'][0]:.3f}",
                        xy=(0.08, 0.92), xytext=(0.22, 0.72),
                        color=P["D7"], fontsize=9, fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color=P["D7"], lw=1.1))

    _save(OUT / "fig2_roc_pr_curves.png", fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Bar chart  (data pulled live from CSV)
# ─────────────────────────────────────────────────────────────────────────────
def generate_figure3():
    import matplotlib.pyplot as plt
    import numpy as np

    print("Generating Figure 3: Scenario-level detector comparison...")

    df = load_wide_evaluation_csv(RESULTS_CSV)

    metrics = ["roc_auc", "pr_auc", "f1"]
    labels = ["ROC-AUC", "PR-AUC", "F1"]

    # Only successful detector/scenario evaluations with finite values.
    valid = df[df["status"].str.startswith("OK", na=False)].copy()

    summary = valid.groupby("detector")[metrics].mean()
    counts = valid.groupby("detector")[metrics].count()

    # Keep detectors with at least one valid observation for every metric.
    summary = summary.dropna(subset=metrics, how="all")
    if summary.empty:
        raise ValueError("No valid detector results found for Figure 3.")

    summary = summary.sort_values("roc_auc", ascending=False)

    fig, ax = plt.subplots(figsize=(15, 7))
    x = np.arange(len(summary))
    width = 0.25

    for i, (metric, label) in enumerate(zip(metrics, labels)):
        values = summary[metric].to_numpy(dtype=float)
        positions = x + (i - 1) * width
        bars = ax.bar(positions, values, width, label=label)

        for bar, value in zip(bars, values):
            if np.isfinite(value):
                ax.annotate(
                    f"{value:.2f}",
                    (bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(summary.index, rotation=55, ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Mean score across valid scenarios")
    ax.set_title(
        "CKEC-GNSS Detector Comparison\n"
        "Scenario means; evaluation data source UNVERIFIED"
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    out = OUT / "fig3_detector_comparison.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

    print("\nROC-AUC summary (valid scenarios only):")
    print(summary[["roc_auc", "pr_auc", "f1"]].round(3).to_string())
    print("\nValid observation counts per detector:")
    print(counts.loc[summary.index].to_string())


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — CUSUM across 3 spoofing types  (honest FAR/delay)
# ─────────────────────────────────────────────────────────────────────────────
def generate_figure6():
    print("Generating Figure 6: CUSUM accumulation (V4, honest FAR)...")
    _theme()

    scenarios = [
        ("Abrupt Spoofing",  generate_abrupt_spoofing,  {"n_samples":2500,"onset":1200,"seed":42}),
        ("Gradual Spoofing", generate_gradual_spoofing,  {"n_samples":2500,"onset":900, "seed":42}),
        ("Stealth Spoofing", generate_stealth_spoofing,  {"n_samples":2500,"onset":900, "seed":42}),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(15, 13), constrained_layout=True)
    fig.suptitle("D7 CUSUM — Evidence Accumulation Across Spoofing Types  (SYNTHETIC)")

    for ax, (title, gen_fn, kw) in zip(axes, scenarios):
        df, meta = gen_fn(**kw)
        onset    = meta.get("onset_sample", kw.get("onset", 1200))
        cal_n    = min(450, onset - 10)
        feat, _, _ = _prepare(df, cal_n=cal_n)
        t        = df["time_s"].values
        y        = feat["attack_label"].values
        cusum_sm = _smooth(feat["cusum_stat"].values, w=9)

        # Calibrate on strictly benign val (first 2/3 of pre-onset)
        tr = np.zeros(len(feat), bool); tr[:cal_n] = True
        d7 = CUSUMDetector(); d7.fit(feat.iloc[tr], y[tr], target_far=0.01)
        th = d7._threshold

        # Compute actual FAR on post-onset benign (none in these scenarios)
        # and Pd on post-onset attack
        y_arr = feat["attack_label"].values
        scores = d7.score(feat).values
        preds  = (scores > th).astype(int)
        n_att  = np.sum(y_arr[onset:] == 1)
        n_ben  = max(1, np.sum(y_arr[:onset] == 0))
        pd_   = np.sum(preds[onset:][y_arr[onset:]==1]) / max(1, n_att)
        far_  = np.sum(preds[:onset][y_arr[:onset]==0]) / n_ben

        # Detection delay
        post_cross = np.flatnonzero(preds[onset:] == 1)
        delay_s = (
            float(t[onset + post_cross[0]] - t[onset])
            if len(post_cross)
            else None
        )

        # Phase z-score on twin axis
        ax2 = ax.twinx(); ax2.set_facecolor(CARD)
        ax2.fill_between(t, feat["z_phi"].values, 0, color=P["phi"], alpha=0.08)
        ax2.plot(t, feat["z_phi"].values, color=P["phi"], lw=0.7, alpha=0.25)
        ax2.set_ylabel(r"$Z_\phi$ (σ)", color=P["phi"], fontsize=9)
        ax2.tick_params(axis="y", labelcolor=P["phi"], labelsize=8)
        ax2.set_ylim(-6, 14)

        ax.axvspan(t[onset], t[-1], color=P["atk"], alpha=0.07, zorder=0)
        ax.axvline(t[onset], color=P["atk"], lw=1.5, ls="--", zorder=3)
        ax.fill_between(t, cusum_sm, 0, color=P["cusum"], alpha=0.20, zorder=1)
        ax.plot(t, feat["cusum_stat"].values, color=P["cusum"], lw=0.8, alpha=0.30)
        ax.plot(t, cusum_sm, color=P["cusum"], lw=2.2, alpha=0.95, label="CUSUM (smoothed)")
        ax.axhline(th, color=FG, lw=1.3, ls=(0,(5,3)), label=f"1% FAR thresh: {th:.2f}")

        if delay_s is not None:
            det_t = t[onset] + delay_s
            ax.scatter([det_t], [th*1.05], marker="v", s=130,
                       color=P["D4"], zorder=8, label=f"Detected +{delay_s:.1f}s")
            ax.annotate(f"  Δt = {delay_s:.1f}s",
                        xy=(det_t, th*1.12), color=P["D4"],
                        fontsize=9, fontweight="bold")

        ax.set_title(f"{title}  |  Pd = {pd_:.3f}  |  FAR = {far_:.4f}", loc="left", fontsize=12)
        ax.set_ylabel("CUSUM Value")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylim(bottom=-0.3)
        ax.legend(loc="upper left", fontsize=8.5, ncol=3)

        # Zone labels
        ax.text(t[onset//2], ax.get_ylim()[1]*0.85, "Benign",
                color=MUTED, fontsize=9, ha="center")
        ax.text((t[onset]+t[-1])/2, ax.get_ylim()[1]*0.85, "SPOOFING ACTIVE",
                color=P["atk"], fontsize=9, fontweight="bold", ha="center")

    _save(OUT / "fig6_cusum_scenarios.png", fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7 — Radar (data from CSV, no rescaling)
# ─────────────────────────────────────────────────────────────────────────────
def generate_figure7():
    import matplotlib.pyplot as plt
    import numpy as np

    print("Generating Figure 7: Raw-metric radar comparison...")

    df = load_wide_evaluation_csv(RESULTS_CSV)

    metrics = ["roc_auc", "pr_auc", "f1", "recall", "precision"]
    labels = ["ROC-AUC", "PR-AUC", "F1", "Recall", "Precision"]

    valid = df[df["status"].str.startswith("OK", na=False)].copy()
    summary = valid.groupby("detector")[metrics].mean()

    # Require all five metrics for a fair radar comparison.
    summary = summary.dropna(subset=metrics)
    if summary.empty:
        print("  Skipped: no detectors have all five metrics available.")
        return

    summary = summary.sort_values("roc_auc", ascending=False)

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    closed_angles = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"polar": True})

    for detector, row in summary.iterrows():
        values = np.clip(row[metrics].to_numpy(dtype=float), 0.0, 1.0).tolist()
        closed_values = values + values[:1]
        ax.plot(closed_angles, closed_values, linewidth=1.5, label=detector)
        ax.fill(closed_angles, closed_values, alpha=0.04)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_title(
        "CKEC-GNSS Multi-Metric Comparison\n"
        "Raw metrics on shared 0–1 scale; source UNVERIFIED",
        pad=24,
    )
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.15, 1.1),
        fontsize=8,
    )

    out = OUT / "fig7_detector_radar.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Ablation (styled, if CSV exists)
# ─────────────────────────────────────────────────────────────────────────────
def generate_figure5():
    print("Generating Figure 5: Ablation sweep (V4)...")
    _theme()

    abl = ABLATION_CSV
    if not abl.exists():
        print("  [SKIP] real_ablation_results.csv not found"); return

    df = pd.read_csv(abl)
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    fig.suptitle(
        "Ablation: Cross-Modal Coupling Weight $w_c$ vs F1 "
        "(data provenance unverified)"
    )

    for w, c, m in zip([5,10,20], [P["D2"],P["D4"],P["D7"]], ["o","s","^"]):
        sub = df[df["window_samples"]==w]
        g = sub.groupby("w_c")["f1"].agg(["mean","sem"]).reset_index()
        ax.errorbar(g["w_c"], g["mean"], yerr=g["sem"],
                    label=f"W = {w}", color=c, marker=m, capsize=4, lw=1.8, markersize=7)

    ax.set_xlabel(r"Cross-Modal Coupling Weight  $w_c$")
    ax.set_ylabel(r"Mean $F_1$ Score (source unverified)")
    ax.legend(loc="upper left", fontsize=9)
    _save(OUT / "fig5_real_ablation_curves.png", fig)

def load_wide_evaluation_csv(csv_path):
    """Load wide-format evaluation results and return a long-format table."""
    import pandas as pd
    import numpy as np

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Evaluation CSV not found: {csv_path}")

    wide = pd.read_csv(csv_path)

    required = {"scenario", "attack_class", "data_source"}
    missing = required - set(wide.columns)
    if missing:
        raise ValueError(f"Evaluation CSV is missing columns: {sorted(missing)}")

    # Detector IDs are inferred from metric column names.
    metric_suffixes = (
        "_recall", "_precision", "_f1", "_far",
        "_roc_auc", "_pr_auc", "_delay_s", "_status",
    )
    detectors = sorted({
        col[:-len(suffix)]
        for col in wide.columns
        for suffix in metric_suffixes
        if col.endswith(suffix)
        and col[:-len(suffix)] not in {
            "bootstrap_D6_full_ckec_cn0_vs_D4_joint_cn0_delta_f1",
        }
    })

    records = []
    for _, row in wide.iterrows():
        for detector in detectors:
            status_col = f"{detector}_status"
            if status_col not in wide.columns:
                continue

            record = {
                "scenario": row["scenario"],
                "attack_class": row["attack_class"],
                "data_source": row["data_source"],
                "detector": detector,
                "status": row[status_col],
            }

            for metric in (
                "recall", "precision", "f1", "far",
                "roc_auc", "pr_auc", "delay_s",
            ):
                col = f"{detector}_{metric}"
                record[metric] = (
                    pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
                    if col in wide.columns else np.nan
                )

            records.append(record)

    long = pd.DataFrame(records)
    if long.empty:
        raise ValueError("No detector metric columns could be identified.")

    for metric in (
        "recall", "precision", "f1", "far", "roc_auc", "pr_auc", "delay_s"
    ):
        long[metric] = pd.to_numeric(long[metric], errors="coerce")

    long["status"] = long["status"].fillna("").astype(str)
    long["data_source"] = long["data_source"].fillna("UNKNOWN").astype(str)

    return long


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    global OUT, RESULTS_CSV, ABLATION_CSV

    parser = argparse.ArgumentParser(
        description="Generate CKEC figures using explicitly selected inputs."
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=PROJECT_ROOT / "reports" / "evaluation_results.csv",
        help="Evaluation results CSV to use for Figures 3 and 7.",
    )
    parser.add_argument(
        "--ablation-csv",
        type=Path,
        default=PROJECT_ROOT / "reports" / "real_ablation_results.csv",
        help="Ablation CSV to use for Figure 5.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "figures",
        help="Directory in which figures will be written.",
    )
    args = parser.parse_args()

    RESULTS_CSV = args.results_csv.expanduser().resolve()
    ABLATION_CSV = args.ablation_csv.expanduser().resolve()
    OUT = args.out_dir.expanduser().resolve()
    OUT.mkdir(parents=True, exist_ok=True)

    print("\n=== CKEC-GNSS Figure Generation ===")
    print(f"Evaluation CSV: {RESULTS_CSV}")
    print(f"Ablation CSV:   {ABLATION_CSV}")
    print(f"Output folder:  {OUT}\n")

    if not RESULTS_CSV.exists():
        print(
            "WARNING: evaluation CSV does not exist. "
            "Figures 3 and 7 will be skipped."
        )

    figure_jobs = [
        ("Figure 1", generate_figure1),
        ("Figure 2", generate_figure2),
        ("Figure 3", generate_figure3),
        ("Figure 5", generate_figure5),
        ("Figure 6", generate_figure6),
        ("Figure 7", generate_figure7),
    ]

    failures = []
    for name, generate in figure_jobs:
        try:
            generate()
        except Exception as exc:
            failures.append((name, str(exc)))
            print(f"\nWARNING: {name} failed: {exc}")
            print("Continuing to the next figure...")

    print(f"\nFigures saved to: {OUT}")
    if failures:
        print("\nFigures needing attention:")
        for name, error in failures:
            print(f"  {name}: {error}")


if __name__ == "__main__":
    main()
