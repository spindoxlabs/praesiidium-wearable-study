"""Figures: error-pool distribution (violin/box) and confusion matrices."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from constants import RISK_CLASSES


def plot_error_pool(combo_pool, device: str, output_dir: Path):
    """Violin + box plots of the combined-session error distribution
    (duration_error, intensity_error) that Monte Carlo draws sample from."""
    durations, intensities = combo_pool
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    for ax, data, label in zip(axes, (durations, intensities), ("duration error (min)", "intensity error")):
        parts = ax.violinplot([data], showmeans=True, showextrema=True)
        for body in parts["bodies"]:
            body.set_facecolor("#4C72B0")
            body.set_alpha(0.65)
        ax.set_xticks([1])
        ax.set_xticklabels([device.capitalize()])
        ax.set_ylabel(label)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    fig.suptitle(f"{device.capitalize()}: combined-session error distribution (violin)")
    fig.tight_layout()
    fig.savefig(output_dir / f"error_violin_{device}.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    for ax, data, label in zip(axes, (durations, intensities), ("duration error (min)", "intensity error")):
        bp = ax.boxplot([data], showfliers=False, patch_artist=True, medianprops=dict(color="black"))
        for patch in bp["boxes"]:
            patch.set_facecolor("#4C72B0")
            patch.set_alpha(0.65)
        ax.set_xticks([1])
        ax.set_xticklabels([device.capitalize()])
        ax.set_ylabel(label)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    fig.suptitle(f"{device.capitalize()}: combined-session error distribution (box)")
    fig.tight_layout()
    fig.savefig(output_dir / f"error_box_{device}.png", dpi=200)
    plt.close(fig)
    print(f"  error violin/box plots -> {output_dir}/error_{{violin,box}}_{device}.png")


def _draw_cm(ax, cm, title):
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(title, fontsize=9)
    ax.set_xticks(range(len(RISK_CLASSES)))
    ax.set_yticks(range(len(RISK_CLASSES)))
    ax.set_xticklabels(RISK_CLASSES, fontsize=8)
    ax.set_yticklabels(RISK_CLASSES, fontsize=8)
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > thresh else "black")
    return im


def sample_draw_indices(n_draws: int, n_show: int = 6) -> np.ndarray:
    """Evenly spaced draw indices shown in the per-draw confusion figure."""
    return np.linspace(0, n_draws - 1, min(n_show, n_draws), dtype=int)


def plot_confusion_sample(confusion_matrices: np.ndarray, device: str, output_dir: Path, n_show: int = 6):
    """Show a handful of individual per-draw confusion matrices side by side
    -- a reminder that there are n_draws of these, not one pooled matrix."""
    n_draws = confusion_matrices.shape[0]
    show_idx = sample_draw_indices(n_draws, n_show)

    fig, axes = plt.subplots(1, len(show_idx), figsize=(2.6 * len(show_idx), 3))
    axes = np.atleast_1d(axes)
    for ax, d in zip(axes, show_idx):
        _draw_cm(ax, confusion_matrices[d], f"draw {d}")
    fig.suptitle(f"{device.capitalize()}: per-draw confusion matrices (sample of {len(show_idx)}/{n_draws})")
    fig.tight_layout()
    fig.savefig(output_dir / f"confusion_matrices_sample_{device}.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = _draw_cm(ax, confusion_matrices.sum(axis=0), f"{device.capitalize()}: pooled over {n_draws} draws")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("Wearable-measured risk")
    ax.set_ylabel("True-activity risk")
    fig.tight_layout()
    fig.savefig(output_dir / f"confusion_matrix_pooled_{device}.png", dpi=200)
    plt.close(fig)
    print(f"  confusion matrix figures -> {output_dir}/confusion_matrices_sample_{device}.png, "
          f"confusion_matrix_pooled_{device}.png")


REF_COLOR = "#8C8C8C"
DEVICE_COLORS = {"fitbit": "#4C72B0", "garmin": "#DD8452"}


CHANNEL_UNITS = {"intensity": "intensity", "duration": "duration (min)"}


def plot_reference_vs_perturbed(results_by_device: dict, output_dir: Path):
    """One violin plot per (device, channel): reference (true activity) vs
    wearable-perturbed (Monte Carlo draw 0) values at the active cells of
    x_test.npy -> violin_ref_vs_<device>_<channel>.png."""
    for device, res in results_by_device.items():
        for chan, unit in CHANNEL_UNITS.items():
            fig, ax = plt.subplots(figsize=(5.5, 4.5))
            parts = ax.violinplot([res["reference_cells"][chan], res["perturbed_cells"][chan]],
                                  showmeans=True, showextrema=True)
            for body, c in zip(parts["bodies"], (REF_COLOR, DEVICE_COLORS.get(device, "#4C72B0"))):
                body.set_facecolor(c)
                body.set_alpha(0.65)
            ax.set_xticks([1, 2])
            ax.set_xticklabels(["Reference", device.capitalize()])
            ax.set_ylabel(unit)
            ax.set_title(f"{chan.capitalize()}: reference vs {device.capitalize()}")
            fig.tight_layout()
            fig.savefig(output_dir / f"violin_ref_vs_{device}_{chan}.png", dpi=200)
            plt.close(fig)
    print(f"  reference-vs-perturbed violins -> {output_dir}/violin_ref_vs_<device>_<channel>.png")
