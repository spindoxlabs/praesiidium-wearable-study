"""Steps 8-9: run the Monte Carlo simulation and report results.

Two things are deliberately NOT averaged into a single number, because
errors can be positive or negative and a mean would hide that they cancel
out:
  - delta_years is kept per draw, per subject (n_draws, n_subjects) -- never
    collapsed to one mean-delta before it's looked at.
  - one confusion matrix per draw (n_draws, 3, 3) -- not pooled into one.

Aggregates (mean/std, a summed confusion matrix) are offered as a *view* on
top of the full data, not as a replacement for it -- the full arrays are
always saved alongside.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from constants import COL_DURATION, COL_INTENSITY, RISK_CLASSES
from pipeline.errors import active_mask, perturb
from pipeline.model import risk_class, years_to_prediabetes


def confusion_matrix(ref_labels, draw_labels) -> np.ndarray:
    """3x3 counts, rows = true-activity risk, cols = wearable-measured risk."""
    idx = {c: i for i, c in enumerate(RISK_CLASSES)}
    cm = np.zeros((len(RISK_CLASSES), len(RISK_CLASSES)), dtype=int)
    for r, e in zip(ref_labels, draw_labels):
        cm[idx[r], idx[e]] += 1
    return cm


def run_monte_carlo(x_test, model, ckpt, torch_device, combo_pool, rng, n_draws, batch_size) -> dict:
    years_ref = years_to_prediabetes(x_test, model, ckpt, torch_device, batch_size)
    ref_labels = risk_class(years_ref)
    n_samples = years_ref.shape[0]

    years_draws = np.empty((n_draws, n_samples))
    confusion_matrices = np.empty((n_draws, len(RISK_CLASSES), len(RISK_CLASSES)), dtype=int)

    # Activity-cell values (reference vs wearable-perturbed), for the
    # reference-vs-perturbed violin and table. The perturbed values of draw 0
    # are kept in full; every draw contributes its channel mean.
    mask = active_mask(x_test)
    reference_cells = {"intensity": x_test[:, :, COL_INTENSITY][mask], "duration": x_test[:, :, COL_DURATION][mask]}
    perturbed_cells = {}
    perturbed_means = {"intensity": np.empty(n_draws), "duration": np.empty(n_draws)}

    for d in range(n_draws):
        perturbed = perturb(x_test, combo_pool, rng)
        years_draws[d] = years_to_prediabetes(perturbed, model, ckpt, torch_device, batch_size)
        confusion_matrices[d] = confusion_matrix(ref_labels, risk_class(years_draws[d]))

        for chan, col in (("intensity", COL_INTENSITY), ("duration", COL_DURATION)):
            cells = perturbed[:, :, col][mask]
            perturbed_means[chan][d] = cells.mean()
            if d == 0:
                perturbed_cells[chan] = cells

    delta_years = years_draws - years_ref[None, :]  # (n_draws, n_samples), signed, kept as-is

    return {
        "years_ref": years_ref,
        "ref_labels": ref_labels,
        "years_draws": years_draws,
        "delta_years": delta_years,
        "confusion_matrices": confusion_matrices,
        "reference_cells": reference_cells,
        "perturbed_cells": perturbed_cells,
        "perturbed_means": perturbed_means,
    }


def save_results(results: dict, device: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"years_ref_{device}.npy", results["years_ref"])
    np.save(output_dir / f"years_draws_{device}.npy", results["years_draws"])
    np.save(output_dir / f"delta_years_{device}.npy", results["delta_years"])
    np.save(output_dir / f"confusion_matrices_{device}.npy", results["confusion_matrices"])
    print(f"  saved per-draw arrays -> {output_dir}/*_{device}.npy")


def print_summary(results: dict, device: str, n_draws: int):
    delta = results["delta_years"]
    cms = results["confusion_matrices"]
    pooled = cms.sum(axis=0)
    total = pooled.sum()
    agree = int(np.trace(pooled))

    print(f"\n  --- {device} : {n_draws} draws ---")
    print(f"  delta-years across all draws: mean {delta.mean():+.3f}, "
          f"std {delta.std():.3f}, min {delta.min():+.3f}, max {delta.max():+.3f}")
    print(f"  pooled confusion (sum of {n_draws} per-draw matrices, rows=true {RISK_CLASSES}):")
    for row in pooled:
        print(f"    {row}")
    print(f"  overall class-flip rate (pooled): {(total - agree) / total:.2%}")
