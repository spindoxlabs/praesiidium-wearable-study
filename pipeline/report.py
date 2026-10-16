"""Tables, summary CSVs and the RESULTS.md write-up, built from the Monte
Carlo results of every device.

Outputs (all under output_dir):
  - summary_<device>.csv            per subject: reference vs perturbed years,
                                    delta-years over the draws, flip probability
  - summary_delta_years.csv         one row per device: distribution of the
                                    delta years-to-onset + paired t-test
  - table_activity_cells.csv        reference vs perturbed intensity/duration
                                    at activity cells (paper Table 2)
  - confusion_onset_lt7.csv         confusion matrix restricted to subjects
                                    with true onset < 7 years (paper Table 3)
  - RESULTS.md                      tables + figure descriptions
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from constants import RISK_CLASSES, RISK_MEDIUM_MAX
from pipeline.analysis import confusion_matrix
from pipeline.model import risk_class
from pipeline.plots import sample_draw_indices

CHANNELS = ("intensity", "duration")
ONSET_CUTOFF = RISK_MEDIUM_MAX  # "elevated risk" == true onset before 7 years


# --------------------------------------------------------------------------- #
# per-subject aggregation
# --------------------------------------------------------------------------- #
def modal_risk(years_draws: np.ndarray) -> np.ndarray:
    """Most frequent risk class per subject across draws (ties -> first class
    in RISK_CLASSES)."""
    draw_labels = np.vstack([risk_class(y) for y in years_draws])
    counts = np.stack([(draw_labels == c).sum(axis=0) for c in RISK_CLASSES])
    return np.array(RISK_CLASSES, dtype=object)[counts.argmax(axis=0)]


def per_subject_summary(results: dict) -> pd.DataFrame:
    years_ref = results["years_ref"]
    years_draws = results["years_draws"]
    delta = results["delta_years"]
    ref_labels = results["ref_labels"]
    draw_labels = np.vstack([risk_class(y) for y in years_draws])

    df = pd.DataFrame({
        "sample_index": np.arange(years_ref.shape[0]),
        "years_ref": years_ref,
        "risk_ref": ref_labels,
        "risk_err_mode": modal_risk(years_draws),
        "years_err_mean": years_draws.mean(axis=0),
        "years_err_std": years_draws.std(axis=0),
        "delta_years_mean": delta.mean(axis=0),
        "delta_years_std": delta.std(axis=0),
        "delta_years_min": delta.min(axis=0),
        "delta_years_max": delta.max(axis=0),
        "flip_prob": (draw_labels != ref_labels[None, :]).mean(axis=0),
    })
    return df.iloc[np.argsort(-np.abs(df["delta_years_mean"].to_numpy()), kind="stable")].reset_index(drop=True)


def delta_years_row(device: str, results: dict, subjects: pd.DataFrame) -> dict:
    delta = results["delta_years"]
    n_draws, n_subjects = delta.shape
    paired = sps.ttest_rel(subjects["years_err_mean"], subjects["years_ref"])
    cms = results["confusion_matrices"]
    pooled = cms.sum(axis=0)
    return {
        "device": device,
        "n_subjects": n_subjects,
        "n_draws": n_draws,
        "years_ref_mean": float(results["years_ref"].mean()),
        "years_perturbed_mean": float(results["years_draws"].mean()),
        # over every (draw, subject) pair -- signed, nothing cancelled first
        "delta_mean": float(delta.mean()),
        "delta_std": float(delta.std()),
        "delta_median": float(np.median(delta)),
        "delta_p2.5": float(np.percentile(delta, 2.5)),
        "delta_p97.5": float(np.percentile(delta, 97.5)),
        "delta_min": float(delta.min()),
        "delta_max": float(delta.max()),
        "abs_delta_mean": float(np.abs(delta).mean()),
        "frac_pairs_delta_neg": float((delta < 0).mean()),
        "frac_pairs_delta_pos": float((delta > 0).mean()),
        # per draw: mean delta over subjects, then spread of that across draws
        "draw_mean_delta_std": float(delta.mean(axis=1).std()),
        "paired_t_years": float(paired.statistic),
        "paired_p_years": float(paired.pvalue),
        "flip_rate_pooled": float(1 - np.trace(pooled) / pooled.sum()),
    }


# --------------------------------------------------------------------------- #
# paper tables
# --------------------------------------------------------------------------- #
def _cohen_d(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp) if sp else 0.0


def activity_cells_table(results_by_device: dict) -> pd.DataFrame:
    """Table 2: reference vs wearable-perturbed intensity/duration at the
    activity cells. Means over all draws; t-test / Cohen's d on draw 0."""
    rows = []
    for device, res in results_by_device.items():
        for chan in CHANNELS:
            ref = res["reference_cells"][chan].astype(float)
            pert0 = res["perturbed_cells"][chan].astype(float)
            means = res["perturbed_means"][chan]
            paired = sps.ttest_rel(pert0, ref)
            rows.append({
                "device": device,
                "channel": chan,
                "n_cells": len(ref),
                "reference_mean": float(ref.mean()),
                "perturbed_mean": float(means.mean()),
                "perturbed_mean_std_over_draws": float(means.std()),
                "delta_mean": float(means.mean() - ref.mean()),
                "delta_pct": float(100 * (means.mean() - ref.mean()) / ref.mean()),
                "paired_t_draw0": float(paired.statistic),
                "paired_p_draw0": float(paired.pvalue),
                "cohen_d_draw0": _cohen_d(pert0, ref),
            })
    return pd.DataFrame(rows)


def onset_lt7_confusion(results: dict) -> np.ndarray:
    """Table 3: same 3x3 layout, only subjects whose reference onset < 7 yr;
    measured class = modal class across the draws (one count per subject)."""
    keep = results["years_ref"] < ONSET_CUTOFF
    return confusion_matrix(results["ref_labels"][keep], modal_risk(results["years_draws"][:, keep]))


# --------------------------------------------------------------------------- #
# markdown helpers
# --------------------------------------------------------------------------- #
def _fmt_p(p):
    if np.isnan(p):
        return "n/a"
    return "< 1e-300" if p == 0.0 else f"{p:.1e}"


def _cm_markdown(cm, rows=RISK_CLASSES):
    lines = ["| true \\ measured | " + " | ".join(RISK_CLASSES) + " |",
             "|---|" + "---|" * len(RISK_CLASSES)]
    for r in rows:
        i = RISK_CLASSES.index(r)
        lines.append(f"| **{r}** | " + " | ".join(str(int(v)) for v in cm[i]) + " |")
    return "\n".join(lines)


def _transitions(cm, total_per_row_div=1):
    """Off-diagonal cells, largest first, as readable strings."""
    out = []
    for i, r in enumerate(RISK_CLASSES):
        row_total = cm[i].sum()
        for j, c in enumerate(RISK_CLASSES):
            if i != j and cm[i, j] > 0:
                out.append((cm[i, j], f"{r} → {c}: {cm[i, j] / total_per_row_div:,.1f} "
                                      f"({cm[i, j] / row_total:.2%} of true-{r})"))
    return [s for _, s in sorted(out, reverse=True)]


def _confusion_sample_description(device, cms):
    n_draws, n_subjects = cms.shape[0], int(cms[0].sum())
    shown = sample_draw_indices(n_draws)
    flips = n_subjects - np.trace(cms, axis1=1, axis2=2)
    shown_flips = ", ".join(f"draw {d}: {int(flips[d])}" for d in shown)
    return (
        f"**`confusion_matrices_sample_{device}.png`** — {len(shown)} of the {n_draws} per-draw "
        f"confusion matrices for {device.capitalize()} (draws {', '.join(map(str, shown))}, evenly spaced). "
        f"Each panel is one Monte Carlo draw: all {n_subjects:,} subjects of `x_test.npy`, rows = risk class "
        f"predicted from the true activity, columns = risk class predicted after adding one realization of "
        f"{device.capitalize()} error to every activity cell. Each panel sums to {n_subjects:,}; the diagonal "
        f"counts subjects whose class is unchanged, the off-diagonal cells are reclassifications. "
        f"Reclassified subjects in the panels shown — {shown_flips}. Across all {n_draws} draws the number of "
        f"reclassified subjects per draw is {flips.mean():.1f} ± {flips.std():.1f} "
        f"(min {int(flips.min())}, max {int(flips.max())}), i.e. a per-draw flip rate of "
        f"{flips.mean() / n_subjects:.2%}. The panels differ only by Monte Carlo sampling of the error; "
        f"their spread is the draw-to-draw uncertainty that a single pooled matrix hides. All {n_draws} "
        f"matrices are saved in `confusion_matrices_{device}.npy` (shape {n_draws}×3×3)."
    )


def _confusion_pooled_description(device, cms):
    n_draws, n_subjects = cms.shape[0], int(cms[0].sum())
    pooled = cms.sum(axis=0)
    total = int(pooled.sum())
    flip = 1 - np.trace(pooled) / total
    trans = _transitions(pooled, total_per_row_div=n_draws)
    trans_txt = "; ".join(trans) if trans else "none"
    return (
        f"**`confusion_matrix_pooled_{device}.png`** — element-wise sum of the {n_draws} per-draw matrices "
        f"for {device.capitalize()}. Each cell counts (subject, draw) pairs, so the matrix totals "
        f"{n_draws} × {n_subjects:,} = {total:,}; dividing a cell by {n_draws} gives the expected number of "
        f"subjects per draw. Rows = true-activity risk, columns = wearable-measured risk. Pooled flip rate "
        f"{flip:.2%}. Off-diagonal transitions (expected subjects per draw, share of the true class): "
        f"{trans_txt}. Row totals per draw: "
        + ", ".join(f"{c} {pooled[i].sum() / n_draws:,.0f}" for i, c in enumerate(RISK_CLASSES))
        + ". The pooled view shows *where* reclassifications go; it does not show how much they vary "
          "between draws — see the per-draw figure for that.\n\n"
        + _cm_markdown(pooled)
    )


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def write_report(results_by_device: dict, output_dir: Path, seed: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    n_draws = next(iter(results_by_device.values()))["years_draws"].shape[0]

    # summary CSVs ---------------------------------------------------------
    delta_rows = []
    for device, res in results_by_device.items():
        subjects = per_subject_summary(res)
        subjects.to_csv(output_dir / f"summary_{device}.csv", index=False, float_format="%.4f")
        delta_rows.append(delta_years_row(device, res, subjects))
    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(output_dir / "summary_delta_years.csv", index=False, float_format="%.6g")
    print(f"  summary CSVs -> {output_dir}/summary_delta_years.csv, summary_<device>.csv")

    # Table 2 --------------------------------------------------------------
    cells = activity_cells_table(results_by_device)
    cells.to_csv(output_dir / "table_activity_cells.csv", index=False, float_format="%.6g")

    # Table 3 --------------------------------------------------------------
    lt7 = {d: onset_lt7_confusion(r) for d, r in results_by_device.items()}
    lt7_rows = [{"device": d, "true_class": RISK_CLASSES[i], **{f"measured_{c}": int(cm[i, j])
                for j, c in enumerate(RISK_CLASSES)}}
                for d, cm in lt7.items() for i in range(len(RISK_CLASSES))]
    pd.DataFrame(lt7_rows).to_csv(output_dir / "confusion_onset_lt7.csv", index=False)
    print(f"  tables -> {output_dir}/table_activity_cells.csv, confusion_onset_lt7.csv")

    md = _render_markdown(results_by_device, delta_df, cells, lt7, n_draws, seed)
    (output_dir / "RESULTS.md").write_text(md)
    print(f"  write-up -> {output_dir}/RESULTS.md")


def _render_markdown(results_by_device, delta_df, cells, lt7, n_draws, seed):
    devices = list(results_by_device)
    n_cells = int(cells["n_cells"].iloc[0])
    n_subjects = int(delta_df["n_subjects"].iloc[0])
    L = []
    L.append("# Wearable-error impact on predicted prediabetes onset — results\n")
    L.append(f"Devices: {', '.join(d.capitalize() for d in devices)} · Monte Carlo draws: {n_draws} · "
             f"seed: {seed} · subjects: {n_subjects:,} · activity cells: {n_cells:,}\n")
    L.append("Generated by `run.py`; every number below is recomputed on each run.\n")

    # Figure 4 -------------------------------------------------------------
    L.append("## Figures — reference vs perturbed activity inputs\n")
    for d in devices:
        for chan in CHANNELS:
            L.append(f"![{d} {chan}](violin_ref_vs_{d}_{chan}.png)\n")
    L.append(
        f"**`violin_ref_vs_<device>_<channel>.png`** — {len(devices) * len(CHANNELS)} violin plots, one per device × channel "
        f"({', '.join(f'{d.capitalize()} {c}' for d in devices for c in CHANNELS)}). Each compares the "
        f"distribution of that channel at the {n_cells:,} activity cells of `x_test.npy` (cells where "
        f"intensity or duration is non-zero): grey = reference (true activity), coloured = the same cells "
        f"after adding that device's error (Monte Carlo draw 0; every draw has the same distribution up to "
        f"sampling noise — see the std over draws in Table 2). Horizontal bars mark mean and extrema. "
        f"Perturbed values are clamped at 0, since a device cannot report negative activity.\n")

    # Table 2 --------------------------------------------------------------
    L.append("## Table 2 — reference vs perturbed, activity-cell values\n")
    head = "| Channel | Reference mean | " + " | ".join(f"{d.capitalize()} mean" for d in devices) + " |"
    L.append(head)
    L.append("|---|---|" + "---|" * len(devices))
    for chan in CHANNELS:
        sub = cells[cells["channel"] == chan].set_index("device")
        ref = sub["reference_mean"].iloc[0]
        vals = " | ".join(f"{sub.loc[d, 'perturbed_mean']:.2f} ({sub.loc[d, 'delta_pct']:+.1f} %)" for d in devices)
        L.append(f"| {chan} | {ref:.2f} | {vals} |")
    L.append("")
    L.append(f"Perturbed means are averaged over the {n_draws} draws (percentage = change vs reference). "
             f"Paired t-test and Cohen's d, perturbed (draw 0) vs reference, n = {n_cells:,} cells:\n")
    for _, r in cells.iterrows():
        L.append(f"* {r['device'].capitalize()} {r['channel']}: Δ = {r['delta_mean']:+.2f}, "
                 f"t = {r['paired_t_draw0']:+.1f}, p {_fmt_p(r['paired_p_draw0'])}, d = {r['cohen_d_draw0']:+.2f} "
                 f"(std of the mean over draws: {r['perturbed_mean_std_over_draws']:.3f})")
    L.append("")

    # delta years summary -------------------------------------------------
    L.append("## Delta years-to-onset (`summary_delta_years.csv`)\n")
    L.append("Δ = perturbed − reference years-to-prediabetes, over every (draw, subject) pair, signed.\n")
    L.append("| Device | ref mean (yr) | perturbed mean (yr) | Δ mean | Δ std | Δ median | Δ 95 % range | "
             "Δ min / max | mean abs(Δ) | paired t (p) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in delta_df.iterrows():
        L.append(f"| {r['device'].capitalize()} | {r['years_ref_mean']:.3f} | {r['years_perturbed_mean']:.3f} | "
                 f"{r['delta_mean']:+.3f} | {r['delta_std']:.3f} | {r['delta_median']:+.3f} | "
                 f"[{r['delta_p2.5']:+.3f}, {r['delta_p97.5']:+.3f}] | {r['delta_min']:+.2f} / {r['delta_max']:+.2f} | "
                 f"{r['abs_delta_mean']:.3f} | {r['paired_t_years']:+.1f} ({_fmt_p(r['paired_p_years'])}) |")
    L.append("")
    L.append("Paired t-test: per-subject mean over draws vs reference (n = subjects). "
             "Per-subject detail (sorted by |mean Δ|) in `summary_<device>.csv`.\n")

    # Table 3 --------------------------------------------------------------
    L.append(f"## Table 3 — confusion matrix, subjects with true onset < {ONSET_CUTOFF:g} years\n")
    L.append(f"Only subjects whose reference onset is < {ONSET_CUTOFF:g} years (the `medium` and `high` rows; "
             f"`low` is empty by construction). Measured class = modal class across the {n_draws} draws, so "
             f"each subject counts once. Also in `confusion_onset_lt7.csv`.\n")
    for d, cm in lt7.items():
        total = int(cm.sum())
        recl = total - int(np.trace(cm))
        L.append(f"### {d.capitalize()} — n = {total}, {recl} reclassified ({recl / total:.1%})\n" if total
                 else f"### {d.capitalize()} — n = 0\n")
        L.append(_cm_markdown(cm, rows=("medium", "high")))
        L.append("")

    # confusion figures ----------------------------------------------------
    L.append("## Confusion-matrix figures\n")
    for d, res in results_by_device.items():
        cms = res["confusion_matrices"]
        L.append(f"### {d.capitalize()}\n")
        L.append(f"![per-draw](confusion_matrices_sample_{d}.png)\n")
        L.append(_confusion_sample_description(d, cms) + "\n")
        L.append(f"![pooled](confusion_matrix_pooled_{d}.png)\n")
        L.append(_confusion_pooled_description(d, cms) + "\n")

    return "\n".join(L)
