# Wearable Risk Pipeline

Estimates how heart-rate measurement error from consumer wearables (**Fitbit**, **Garmin**) changes a surrogate model's predicted **time to prediabetes onset**, using a chest-strap **Polar** as the reference device.

The pipeline:

1. Measures each wearable's real-world error against Polar on recorded training sessions.
2. Builds an empirical distribution of "full-day" errors from those sessions.
3. Injects that error into the activity features of the MT2D test set (`x_test.npy`).
4. Runs a pretrained LSTM surrogate with and without the error, in a Monte Carlo loop.
5. Reports per-subject changes in years-to-onset and risk-class flips.

---

## Quick start (Docker + Task)

Requires [Docker](https://docs.docker.com/get-docker/) and [Task](https://taskfile.dev/installation/).

```bash
task setup    # download and extract the input data into ./data (skipped if already there)
task up       # build the image and run the full pipeline; results land in ./output
task down     # stop and remove the container
```

| Task | What it does |
|---|---|
| `task setup` | Downloads `DATA_URL` with `curl`, extracts it into `data/`, deletes the archive. It is skipped when `data/x_test.npy`, `data/model_I2O.checkpoint` and `data/raw_sessions/` already exist |
| `task up` | Runs `task build`, then starts the container `wearable-risk-pipeline` in the background and follows its logs. `data/` is mounted read-only and `output/` read-write. The container runs as your user, so output files are not owned by root |
| `task down` | Removes the container (stopping it first if it is still running) |
| `task build` | Builds only the image `wearable-risk-pipeline` |

`DATA_URL` in `Taskfile.yml` is a placeholder (`https://example.com/...`). Set it to the real location of a `.tar.gz` that contains `x_test.npy`, `model_I2O.checkpoint` and `raw_sessions/` at its top level. You can also override it per call:

```bash
task setup DATA_URL=https://my-server/data.tar.gz
```

Pipeline flags (see below) go after `--`:

```bash
task up -- --devices garmin --draws 20
```

Pressing Ctrl-C during `task up` stops following the logs, but the container keeps running. Use `docker logs -f wearable-risk-pipeline` to reattach, or `task down` to stop it.

The image (`Dockerfile`) is based on `python:3.13-slim`. It installs the pinned versions in `requirements.txt` (CPU-only torch) and contains only the code. Data and results always stay on the host.

## Running without Docker

```bash
# from the project root
python -m venv .venv && source .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
python run.py                      # both devices, 100 draws, seed 7
```

Options:

| Flag | Default | Meaning |
|---|---|---|
| `--devices` | `fitbit garmin` | Devices to simulate (one or both) |
| `--draws` | `100` | Monte Carlo draws per device |
| `--seed` | `7` | Seed for NumPy RNG and `torch.manual_seed` |
| `--batch-size` | `512` | Inference batch size for the surrogate |
| `--data-dir` | `./data` | Input directory |
| `--output-dir` | `./output` | Output directory (created if missing) |

Examples:

```bash
python run.py --devices garmin --draws 20            # quick single-device run
python run.py --draws 500 --output-dir output_500    # bigger run, separate folder
```

Inference runs on the CPU. Each device builds 20,000 error combinations once, then each draw runs one full pass of the surrogate over 2,183 subjects.

### Dependencies

`requirements.txt` pins numpy 2.5.3, pandas 3.0.5, scipy 1.18.1, matplotlib 3.11.2, torch 2.14.0, joblib 1.5.2 and scikit-learn 1.7.2. The checkpoint's scalers were pickled with an older scikit-learn. The resulting version warning is suppressed in `pipeline/model.py`.

---

## Project layout

```
wearable_risk_pipeline/
├── run.py                 # entry point: runs steps 1-10 end to end
├── constants.py           # all thresholds, factors, column indices, risk cut-offs
├── pipeline/
│   ├── sessions.py        # steps 1-4: raw sessions -> per-session device error
│   ├── errors.py          # steps 5-6: combined-session error pool, perturbation of x_test
│   ├── model.py           # step 7: LSTM surrogate, years-to-prediabetes, risk class
│   ├── analysis.py        # steps 8-9: Monte Carlo loop, per-draw confusion matrices
│   ├── plots.py           # figures (error pool, confusion matrices, ref-vs-perturbed violins)
│   └── report.py          # step 10: summary CSVs, paper tables, RESULTS.md
├── Dockerfile             # image that runs run.py (code only; data/output mounted)
├── Taskfile.yml           # task setup / up / down / build
├── requirements.txt       # pinned Python dependencies
├── data/                  # inputs (≈1.2 GB, see below; fetched by `task setup`)
└── output/                # generated results (overwritten on each run)
```

---

## Data structure (`data/`)

### `data/raw_sessions/<user_id>/`

There is one folder per participant: 40 folders, 39 of which have session data. `hrdbny5` has only `user.json`.

```
<user_id>/
├── user.json
└── reports/
    ├── aggregated/
    │   ├── aggregated_report.csv
    │   └── activities_summary.csv
    └── sessions/
        ├── sessions_report.csv
        └── records/
            └── polar_training-session-<date>-<id>-<uuid>.json   # one per session
```

#### `user.json`: participant info and HR-zone thresholds

```json
{
  "sex": "female", "age": 41, "height": 166, "weight": 62,
  "wrist_circumference": 14.5,
  "moderate_heart_rate": 113,
  "vigorous_heart_rate": 135,
  "peak_heart_rate": 162
}
```

Only the three `*_heart_rate` fields are used, as the personal zone thresholds in bpm. A missing threshold falls back to `DEFAULT_HR_THRESHOLDS` in `constants.py` (120 / 150 / 180).

#### `records/polar_training-session-*.json`: one training session

There are 1,416 files in total. Fields:

| Field | Type | Description |
|---|---|---|
| `activity_type` | str | e.g. `OTHER_INDOOR`, `WALKING`, `RUNNING`, `STRENGTH_TRAINING`, … (25 types) |
| `active` | bool | session flag from the export |
| `heart_rate_times` | list[str] | Polar (reference) timestamps, ISO-8601, ~1 Hz |
| `heart_rate_values` | list[number] | Polar HR in bpm |
| `associated_device_records.heart_rate_times` | list[str] | Wearable timestamps for the same window |
| `associated_device_records.heart_rate_values` | list[number] | Wearable HR in bpm |
| `associated_fitbit` / `associated_garmin` | truthy | Which wearable was worn (older export format) |
| `associated_device_name` | str | `"FitbitFile"`, `"GarminFile"` or `"NoneType"` (newer export format) |

The wearable is identified by `associated_fitbit` / `associated_garmin` when present (older format, 464 records), otherwise by `associated_device_name` (newer format, 901 records). Records with `"NoneType"` or no device data are skipped. After preprocessing and time-window alignment, 1,326 sessions remain in the error pool: 874 Fitbit and 452 Garmin.

#### `sessions_report.csv`, `aggregated_report.csv`, `activities_summary.csv`

These are per-user summaries produced by the upstream export: session coverage, average HR per device, hours of sport activity, and similar. They are **not read by the pipeline** and are kept for reference only.

### `data/x_test.npy`: MT2D medical records

A `float32` array of shape **`(2183, 540, 8)`**: subjects × time steps × features. Each time step is 1/3 of a day, so 540 steps cover 180 days.

| Index | Feature | Notes |
|---|---|---|
| 0 | `carb` | diet |
| 1 | `prot` | diet |
| 2 | `fat` | diet |
| 3 | `intensity` | **perturbed** (`COL_INTENSITY`) |
| 4 | `duration` | **perturbed** (`COL_DURATION`), minutes |
| 5 | `sex` | |
| 6 | `age` | |
| 7 | `BMI` | |

An **activity cell** is a `(subject, step)` with non-zero intensity or duration. There are 99,578 of them, and only these cells receive wearable error.

### `data/model_I2O.checkpoint`: surrogate model

This is a `joblib` dict holding a trained 2-layer LSTM: `input_size=8`, `n_hidden=25`, `n_layers=2` and `output_size=11`. The dict also contains the state dict (`model`) and the `x_scaler` / `y_scaler` (`MinMaxScaler`). The pipeline reuses it as-is and does not retrain it. Output index 10 (`GBL_OUTPUT_IDX`) is the glucose base level.

---

## Pipeline

The step numbers below match the `=== step N ===` lines that `run.py` prints.

### Steps 1-4: raw sessions → per-session error (`pipeline/sessions.py`)

For each session that has both a Polar series and a Fitbit/Garmin series:

1. **Preprocess** each HR series:
   - `remove_flat_segments`: runs of 30 or more identical consecutive samples (a stuck sensor) collapse to a single point.
   - `remove_outliers`: samples outside 50–220 bpm are dropped.
2. **Align** both series to their common time window `[max(start), min(end)]`.
3. **Compute HR zones** with the user's thresholds: `rest < moderate ≤ … < vigorous ≤ … < peak ≤`. Each sample is weighted by the time until the next sample (the last sample gets the median Δt), giving minutes per zone.
4. **Compute intensity and duration** from time-in-zone:
   - `duration = min(moderate) + min(vigorous) + min(peak)`
   - `intensity = Σ minutes_in_zone × VO2_factor[zone]`, with factors `rest 0`, `moderate 0.203`, `vigorous 0.5469`, `peak 0.8225`.
5. **Compute the error** as `device − polar` for both duration and intensity. `total_time` is the device's minutes across all zones.

The output is a DataFrame with the columns `session_id, device, activity_type, duration_error, intensity_error, total_time`. `run.py` uses `build_session_error_pool_zeros_included`, which keeps zero-error sessions. `build_session_error_pool` drops them.

### Step 5: combined-session error pool (`pipeline/errors.py::build_combo_pool`)

A day's error is the sum of several sessions, not a single one. So for each device, 20,000 combinations (`N_COMBOS`) are precomputed. Each combination is built as follows:

- Shuffle the `OTHER_INDOOR` / `OTHER_OUTDOOR` sessions and keep adding their errors until the accumulated `total_time` reaches 16 h (`HOURS_THRESHOLD_MINUTES`).
- Add exactly one session of another activity type.

The result is the device's empirical error distribution (see `error_violin_<device>.png` and `error_box_<device>.png`).

### Step 6: perturb the records (`errors.py::perturb`)

In each draw, every activity cell gets one combination, sampled with replacement. Its duration and intensity errors are added to columns 4 and 3. The results are clamped at 0, since a device cannot report negative activity.

### Step 7: surrogate prediction (`pipeline/model.py`)

1. Scale the input with `x_scaler`, run the LSTM, drop the first 9 warm-up steps, invert with `y_scaler`, and take output 10, the glucose base level (GBL).
2. Fit a line through the origin to `GBL − 5.0` against time in years. The slope gives `years = (5.6 − 5.0) / slope`.
3. Map negative values, NaN and anything above 10 to **10 years** (`CAP_YEARS`, meaning no onset within the horizon).
4. Assign a risk class: **high** for ≤ 2 years, **medium** for ≤ 7 years, **low** otherwise.

### Steps 8-9: Monte Carlo (`pipeline/analysis.py`)

- The reference prediction is computed once on the unperturbed `x_test`.
- For each draw `d`, the pipeline perturbs the input, predicts, and stores `years_draws[d]` and one 3×3 confusion matrix: rows are the true-activity risk, columns are the wearable-measured risk.
- `delta_years = years_draws − years_ref` keeps its sign and has shape `(n_draws, n_subjects)`. It is **never averaged before saving**, because positive and negative errors can cancel. Means and pooled matrices are only views on top of the full arrays.

### Step 10: reporting (`pipeline/report.py`, `pipeline/plots.py`)

This step writes summary CSVs, the paper tables, figures, and `RESULTS.md` with tables and figure descriptions. Every number is recomputed on each run.

---

## Outputs (`output/`)

### Arrays (per device)

| File | Shape | Content |
|---|---|---|
| `years_ref_<device>.npy` | `(2183,)` | Reference years-to-prediabetes |
| `years_draws_<device>.npy` | `(n_draws, 2183)` | Years-to-prediabetes per draw |
| `delta_years_<device>.npy` | `(n_draws, 2183)` | Signed `draw − reference` |
| `confusion_matrices_<device>.npy` | `(n_draws, 3, 3)` | One confusion matrix per draw (order: low, medium, high) |

### Tables

| File | Content |
|---|---|
| `summary_<device>.csv` | One row per subject, sorted by \|mean Δ\|. Columns: reference years and risk, modal measured risk, mean/std/min/max Δ, `flip_prob` (fraction of draws where the class changed) |
| `summary_delta_years.csv` | One row per device. Columns: Δ distribution (mean, std, median, 2.5/97.5 percentiles, min/max), fraction of negative/positive pairs, paired t-test, pooled flip rate |
| `table_activity_cells.csv` | Paper Table 2: reference vs perturbed intensity/duration at activity cells, % change, paired t-test, Cohen's d |
| `confusion_onset_lt7.csv` | Paper Table 3: confusion matrix for subjects with true onset < 7 years, where the measured class is the modal class across draws |
| `RESULTS.md` | Human-readable write-up of all of the above, plus figure descriptions |

### Figures

| File | Content |
|---|---|
| `error_violin_<device>.png`, `error_box_<device>.png` | Distribution of the combined-session error pool |
| `confusion_matrices_sample_<device>.png` | 6 evenly spaced per-draw confusion matrices |
| `confusion_matrix_pooled_<device>.png` | Sum of all per-draw matrices |
| `violin_ref_vs_<device>_<intensity\|duration>.png` | Activity-cell values, reference vs perturbed (draw 0) |

### Loading results

```python
import numpy as np
delta = np.load("output/delta_years_garmin.npy")        # (draws, subjects)
cms   = np.load("output/confusion_matrices_garmin.npy") # (draws, 3, 3)
print(delta.mean(axis=0))   # per-subject mean Δ years
print(cms.sum(axis=0))      # pooled confusion matrix
```

---

## Configuration (`constants.py`)

All tunables are in one file:

| Constant | Value | Used for |
|---|---|---|
| `DEFAULT_HR_THRESHOLDS` | 120/150/180 bpm | Zone fallback when `user.json` lacks thresholds |
| `VO2_FACTORS_REFERENCE`, `VO2_FACTORS_DEVICE` | 0 / 0.203 / 0.5469 / 0.8225 | Intensity per zone. The original scripts used 0.546 for Polar vigorous. The two dicts are kept separate for that reason, but both currently use the same values |
| `INDOOR_OUTDOOR_ACTIVITY_TYPES` | `OTHER_INDOOR`, `OTHER_OUTDOOR` | Sessions accumulated into a "day" |
| `HOURS_THRESHOLD_MINUTES` | 960 (16 h) | Target accumulated time per combination |
| `N_COMBOS` | 20,000 | Size of the precomputed error pool |
| `COL_INTENSITY`, `COL_DURATION` | 3, 4 | Perturbed columns of `x_test` |
| `WARM_STEPS` | 9 | LSTM warm-up steps discarded |
| `GBL_OUTPUT_IDX` | 10 | Glucose base-level output |
| `STARTING_VALUE`, `THRESHOLD_PREDIABETES` | 5.0, 5.6 | GBL baseline and prediabetes threshold |
| `CAP_YEARS` | 10 | Cap and "never" value for years-to-onset |
| `RISK_HIGH_MAX`, `RISK_MEDIUM_MAX` | 2, 7 years | Risk-class cut-offs |
| `DEFAULT_SEED` | 7 | Default RNG seed |

---

## Reproducibility

- Each device starts from a fresh `np.random.default_rng(seed)`. The results for one device therefore do not depend on which other devices are run.
- `torch.manual_seed(seed)` is set as well, although inference itself is deterministic.
- Rerunning with the same `--seed`, `--draws` and data produces identical outputs.
