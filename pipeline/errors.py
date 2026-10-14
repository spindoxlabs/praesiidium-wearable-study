"""Steps 5-6: turn per-session errors into a Monte Carlo error pool, then
apply it to the MT2D medical records (x_test.npy).

A wearable doesn't just mis-measure one activity in isolation -- a full
day's error is the sum of several sessions' errors. So each "combo" is: keep
drawing random indoor/outdoor sessions until their combined total_time
reaches HOURS_THRESHOLD_MINUTES, then add exactly one more, non-indoor/
outdoor session (mirrors che_fatica_praes/add_errors.py's pick_sessions_errors).

We precompute a large pool of these combos once per device (N_COMBOS), then
every Monte Carlo draw resamples from that pool with replacement -- this is
the empirical device-error distribution, built entirely from real session
data, no synthetic shortcuts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from constants import COL_DURATION, COL_INTENSITY, HOURS_THRESHOLD_MINUTES, INDOOR_OUTDOOR_ACTIVITY_TYPES, N_COMBOS


def _one_combo(indoor_outdoor: pd.DataFrame, other: pd.DataFrame, rng: np.random.Generator) -> tuple[float, float]:
    shuffled = indoor_outdoor.sample(frac=1.0, random_state=rng).reset_index(drop=True)

    total_time = 0.0
    duration_error = 0.0
    intensity_error = 0.0
    for _, row in shuffled.iterrows():
        if total_time >= HOURS_THRESHOLD_MINUTES:
            break
        total_time += row["total_time"]
        duration_error += row["duration_error"]
        intensity_error += row["intensity_error"]

    extra = other.sample(1, random_state=rng).iloc[0]
    duration_error += extra["duration_error"]
    intensity_error += extra["intensity_error"]
    return duration_error, intensity_error


def build_combo_pool(session_pool: pd.DataFrame, device: str, rng: np.random.Generator,
                      n_combos: int = N_COMBOS) -> tuple[np.ndarray, np.ndarray]:
    """Precompute n_combos independent (duration_error, intensity_error) draws
    for one device, each the sum of a realistic multi-session day."""
    pool = session_pool[session_pool["device"] == device]
    if pool.empty:
        raise ValueError(f"no session errors found for device={device!r}")

    indoor_outdoor = pool[pool["activity_type"].isin(INDOOR_OUTDOOR_ACTIVITY_TYPES)].reset_index(drop=True)
    other = pool[~pool["activity_type"].isin(INDOOR_OUTDOOR_ACTIVITY_TYPES)].reset_index(drop=True)
    if indoor_outdoor.empty or other.empty:
        raise ValueError(
            f"device={device!r} needs both indoor/outdoor and other-activity sessions "
            f"to build combos (indoor/outdoor={len(indoor_outdoor)}, other={len(other)})"
        )

    durations = np.empty(n_combos)
    intensities = np.empty(n_combos)
    for i in range(n_combos):
        durations[i], intensities[i] = _one_combo(indoor_outdoor, other, rng)
    return durations, intensities


def active_mask(x: np.ndarray) -> np.ndarray:
    """Cells with real activity logged -- the only ones a wearable can
    mis-measure."""
    return (x[:, :, COL_INTENSITY] != 0) | (x[:, :, COL_DURATION] != 0)


def perturb(x_test: np.ndarray, combo_pool: tuple[np.ndarray, np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """One Monte Carlo draw: every active cell gets one combo, sampled with
    replacement, added to intensity/duration and clamped at zero (a device
    can't report negative activity)."""
    durations, intensities = combo_pool
    mask = active_mask(x_test)
    n_active = int(mask.sum())

    idx = rng.integers(0, durations.shape[0], size=n_active)

    perturbed = x_test.copy()
    new_intensity = perturbed[:, :, COL_INTENSITY][mask] + intensities[idx]
    new_duration = perturbed[:, :, COL_DURATION][mask] + durations[idx]
    np.clip(new_intensity, 0.0, None, out=new_intensity)
    np.clip(new_duration, 0.0, None, out=new_duration)

    perturbed[:, :, COL_INTENSITY][mask] = new_intensity
    perturbed[:, :, COL_DURATION][mask] = new_duration
    return perturbed
