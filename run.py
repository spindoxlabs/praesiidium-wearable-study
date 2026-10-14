#!/usr/bin/env python3
"""Wearable-error risk-impact pipeline, standalone, run start to finish:

1. Read raw per-user session data for Fitbit and Garmin (data/raw_sessions/).
2. Preprocess (flat-segment / outlier removal).
3. Compute HR zones.
4. From zones, compute intensity/duration, and each session's error vs Polar.
5. Build the empirical error pool (combined-session draws).
6. Add errors to the MT2D medical records (data/x_test.npy).
7. Run the surrogate model's predictions, with and without error.
8. Report per-draw differences (never averaged away).
9. Build one confusion matrix per Monte Carlo draw.
10. Write summary CSVs, paper tables and RESULTS.md (figure descriptions).

Everything under data/ is a local, standalone copy -- nothing here reads
from the original PRAESIIDIUM/che_fatica_praes project directories.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from constants import DEFAULT_SEED, DEVICES
from pipeline.analysis import print_summary, run_monte_carlo, save_results
from pipeline.errors import build_combo_pool
from pipeline.model import load_surrogate
from pipeline.plots import plot_confusion_sample, plot_error_pool, plot_reference_vs_perturbed
from pipeline.report import write_report
from pipeline.sessions import build_session_error_pool, build_session_error_pool_zeros_included

ROOT = Path(__file__).resolve().parent


def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--devices", nargs="+", choices=DEVICES, default=list(DEVICES))
    ap.add_argument("--draws", type=int, default=100, help="Monte Carlo draws")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "output")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    torch_device = torch.device("cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=== step 1-4: raw sessions -> per-session errors ===")
    session_pool = build_session_error_pool_zeros_included(args.data_dir / "raw_sessions")
    print(f"  {len(session_pool)} sessions with measurable error "
          f"({(session_pool['device'] == 'fitbit').sum()} fitbit, "
          f"{(session_pool['device'] == 'garmin').sum()} garmin)")

    print("\n=== loading x_test.npy + surrogate checkpoint ===")
    x_test = np.load(args.data_dir / "x_test.npy")
    model, ckpt = load_surrogate(args.data_dir / "model_I2O.checkpoint", torch_device)
    print(f"  x_test shape {x_test.shape}")

    results_by_device = {}
    for device in args.devices:
        print(f"\n=== {device.upper()} ===")
        rng = make_rng(args.seed)

        print("  step 5: building error pool (combined-session draws)")
        combo_pool = build_combo_pool(session_pool, device, rng)
        plot_error_pool(combo_pool, device, args.output_dir)

        print(f"  steps 6-9: {args.draws} Monte Carlo draws")
        results = run_monte_carlo(x_test, model, ckpt, torch_device, combo_pool, rng, args.draws, args.batch_size)

        save_results(results, device, args.output_dir)
        plot_confusion_sample(results["confusion_matrices"], device, args.output_dir)
        print_summary(results, device, args.draws)
        results_by_device[device] = results

    print("\n=== step 10: summary CSVs, tables, figure descriptions ===")
    plot_reference_vs_perturbed(results_by_device, args.output_dir)
    write_report(results_by_device, args.output_dir, args.seed)

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
