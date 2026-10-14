"""Steps 1-4: read raw per-user session data, preprocess, compute HR zones,
then intensity/duration (and their errors) from those zones.

Each user folder under data/raw_sessions/<user_id>/ holds:
  - user.json                                   patient info + HR-zone thresholds
  - reports/sessions/records/polar_*.json       one file per session, with the
    Polar (reference) HR series and, when a Fitbit/Garmin was worn at the same
    time, that device's HR series under "associated_device_records"

For every session with both a Polar and a device HR series, we preprocess
both, compute time-in-zone, turn that into intensity/duration for each side,
and return the device's error (device - polar) for that session.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from constants import (
    DEFAULT_HR_THRESHOLDS,
    VO2_FACTORS_DEVICE,
    VO2_FACTORS_REFERENCE,
    ZONE_NAMES,
)


def remove_flat_segments(times, values, min_length=30):
    """Collapse runs of `min_length`+ identical consecutive values to one point.

    A stuck sensor repeats the same reading; long repeats are dropped rather
    than treated as real dwell time in one HR zone.
    """
    cleaned_times, cleaned_values = [], []
    n, i = len(values), 0
    while i < n:
        j = i + 1
        while j < n and values[j] == values[i]:
            j += 1
        if j - i < min_length:
            cleaned_times.extend(times[i:j])
            cleaned_values.extend(values[i:j])
        else:
            cleaned_times.append(times[i])
            cleaned_values.append(values[i])
        i = j
    return cleaned_times, cleaned_values


def remove_outliers(times, values, min_threshold=50, max_threshold=220):
    """Drop HR samples outside a physiologically plausible range (also
    catches negative/zero sensor glitches)."""
    filtered_times, filtered_values = [], []
    for t, v in zip(times, values):
        if min_threshold <= v <= max_threshold:
            filtered_times.append(t)
            filtered_values.append(v)
    return filtered_times, filtered_values


def preprocess_hr(times, values):
    times, values = remove_flat_segments(times, values)
    times, values = remove_outliers(times, values)
    return times, values


def zone_durations(times, values, hr_thresholds) -> dict[str, float]:
    """Minutes spent in each HR zone, from a preprocessed HR series."""
    if len(times) < 2 or len(values) == 0:
        return {z: 0.0 for z in ZONE_NAMES}

    df = pd.DataFrame({"time": pd.to_datetime(times, utc=True), "hr": values})
    df = df.sort_values("time").reset_index(drop=True)
    df["delta_s"] = (df["time"].shift(-1) - df["time"]).dt.total_seconds()
    median_dt = float(np.nanmedian(df["delta_s"].dropna())) if df["delta_s"].notna().any() else 1.0
    df.loc[df.index[-1], "delta_s"] = median_dt

    m, v, p = hr_thresholds["moderate"], hr_thresholds["vigorous"], hr_thresholds["peak"]

    def classify(hr):
        if hr < m:
            return "rest"
        if hr < v:
            return "moderate"
        if hr < p:
            return "vigorous"
        return "peak"

    df["zone"] = df["hr"].apply(classify)
    minutes = df.groupby("zone")["delta_s"].sum().reindex(ZONE_NAMES, fill_value=0.0) / 60.0
    return {z: float(minutes[z]) for z in ZONE_NAMES}


def zone_intensity_duration(zone_minutes: dict[str, float], vo2_factors: dict[str, float]) -> tuple[float, float]:
    """(intensity, duration) from time-in-zone: duration = active minutes,
    intensity = sum(minutes_in_zone * VO2_factor_for_zone)."""
    duration = zone_minutes["moderate"] + zone_minutes["vigorous"] + zone_minutes["peak"]
    intensity = sum(zone_minutes[z] * vo2_factors[z] for z in ZONE_NAMES)
    return intensity, duration


def _load_session_errors_for_user(user_dir: Path) -> list[dict]:
    user_json = user_dir / "user.json"
    records_dir = user_dir / "reports" / "sessions" / "records"
    if not user_json.exists() or not records_dir.exists():
        return []

    user_data = json.loads(user_json.read_text())
    hr_thresholds = {
        "moderate": user_data.get("moderate_heart_rate", DEFAULT_HR_THRESHOLDS["moderate"]),
        "vigorous": user_data.get("vigorous_heart_rate", DEFAULT_HR_THRESHOLDS["vigorous"]),
        "peak": user_data.get("peak_heart_rate", DEFAULT_HR_THRESHOLDS["peak"]),
    }

    rows = []
    for record_path in sorted(records_dir.glob("*.json")):
        data = json.loads(record_path.read_text())

        assoc = data.get("associated_device_records") or {}
        if not assoc.get("heart_rate_times"):
            continue  # no device worn during this session

        device = "fitbit" if data.get("associated_fitbit") else "garmin" if data.get("associated_garmin") else None
        if device is None:
            continue

        polar_times, polar_values = preprocess_hr(
            data.get("heart_rate_times", []), data.get("heart_rate_values", [])
        )
        dev_times, dev_values = preprocess_hr(
            assoc.get("heart_rate_times", []), assoc.get("heart_rate_values", [])
        )
        if len(polar_times) < 2 or len(dev_times) < 2:
            continue

        # Only compare the two series over their common time window.
        p_times = pd.to_datetime(polar_times, utc=True)
        d_times = pd.to_datetime(dev_times, utc=True)
        start, end = max(p_times[0], d_times[0]), min(p_times[-1], d_times[-1])
        if end <= start:
            continue
        p_mask = (p_times >= start) & (p_times <= end)
        d_mask = (d_times >= start) & (d_times <= end)

        polar_zones = zone_durations(p_times[p_mask], np.asarray(polar_values)[p_mask], hr_thresholds)
        device_zones = zone_durations(d_times[d_mask], np.asarray(dev_values)[d_mask], hr_thresholds)

        polar_intensity, polar_duration = zone_intensity_duration(polar_zones, VO2_FACTORS_REFERENCE)
        device_intensity, device_duration = zone_intensity_duration(device_zones, VO2_FACTORS_DEVICE)

        rows.append({
            "session_id": record_path.stem,
            "device": device,
            "activity_type": data.get("activity_type", "unknown"),
            "duration_error": device_duration - polar_duration,
            "intensity_error": device_intensity - polar_intensity,
            "total_time": sum(device_zones.values()),
        })
    return rows


def build_session_error_pool(raw_sessions_dir: Path) -> pd.DataFrame:
    """Steps 1-4 end to end: every user's raw sessions -> one DataFrame of
    per-session (duration_error, intensity_error), device vs Polar reference.

    Rows where both errors are ~0 are dropped -- no measurable error to pool.
    """
    rows = []
    for user_dir in sorted(p for p in raw_sessions_dir.iterdir() if p.is_dir()):
        rows.extend(_load_session_errors_for_user(user_dir))

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    eps = 1e-6
    no_error = (df["duration_error"].abs() < eps) & (df["intensity_error"].abs() < eps)
    return df.loc[~no_error].reset_index(drop=True)

def build_session_error_pool_zeros_included(raw_sessions_dir: Path) -> pd.DataFrame:
    """Steps 1-4 end to end: every user's raw sessions -> one DataFrame of
    per-session (duration_error, intensity_error), device vs Polar reference.

    Unlike build_session_error_pool, this version keeps rows where both errors are ~0.
    """
    rows = []
    for user_dir in sorted(p for p in raw_sessions_dir.iterdir() if p.is_dir()):
        rows.extend(_load_session_errors_for_user(user_dir))

    df = pd.DataFrame(rows)
    return df.reset_index(drop=True)