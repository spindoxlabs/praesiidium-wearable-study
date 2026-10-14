"""Single source of truth for constants used across the pipeline.

Zone thresholds fall back to these values only when a user's own
user.json doesn't specify them. VO2-per-zone intensity factors and the
polar-vs-device rounding mismatch are preserved from the original
PRAESIIDIUM/che_fatica_praes scripts -- not invented here.
"""

DEVICES = ("fitbit", "garmin")

# --- HR zone classification (fallback thresholds, bpm) ---
DEFAULT_HR_THRESHOLDS = {"moderate": 120, "vigorous": 150, "peak": 180}
ZONE_NAMES = ("rest", "moderate", "vigorous", "peak")

# --- VO2-per-zone intensity factors ---
# NOTE: polar-vigorous (0.546) vs device-vigorous (0.5469) differ by a
# rounding digit in the original scripts. Preserved as-is.
VO2_FACTORS_REFERENCE = {"rest": 0.0, "moderate": 0.203, "vigorous": 0.5469, "peak": 0.8225}
VO2_FACTORS_DEVICE = {"rest": 0.0, "moderate": 0.203, "vigorous": 0.5469, "peak": 0.8225}

# --- Error-pool session combination (from che_fatica_praes/add_errors.py) ---
INDOOR_OUTDOOR_ACTIVITY_TYPES = ("OTHER_INDOOR", "OTHER_OUTDOOR")
HOURS_THRESHOLD_MINUTES = 16.0 * 60.0
N_COMBOS = 20_000  # precomputed pool of combined-session error draws

# --- x_test.npy feature columns (Indices_explanation.txt) ---
INPUT_COLUMNS = ["carb", "prot", "fat", "intensity", "duration", "sex", "age", "BMI"]
COL_INTENSITY = 3
COL_DURATION = 4

# --- Surrogate model output / risk formula (che_fatica_praes reference) ---
WARM_STEPS = 9
GBL_OUTPUT_IDX = 10
STARTING_VALUE = 5.0
THRESHOLD_PREDIABETES = 5.6
CAP_YEARS = 10.0

RISK_HIGH_MAX = 2.0
RISK_MEDIUM_MAX = 7.0
RISK_CLASSES = ("low", "medium", "high")

DEFAULT_SEED = 7
