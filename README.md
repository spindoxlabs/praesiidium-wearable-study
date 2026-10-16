# Introduction

This repository comes as supplementary material of the paper *Assessing the Impact of Wearable-Derived Physical Activity Measurement Errors on Personalised Prediabetes Risk Prediction*. The code allows you to reproduce the results described in the paper. The paper shows that the heart rate measured by commercial wrist-worn devices is sufficient to estimate the amount of physical activity performed by patients for prediabetes risk prediction. This README is divided into three sections:

1. How to reproduce the paper's results
2. Description of the input data
3. Description of the output data

## Reproduce paper's results

To run the code, you need the following prerequisites:

* [Taskfile](https://taskfile.dev/docs/installation)
* [Docker](https://docs.docker.com/engine/install)

Once they are installed, move to the root folder of the project and run `task setup` to build the docker image and download the input data. Than run `task up`to run the pipeline that reproduces the results, and stores the output in the `output` folder.

## Input data

The pipeline has two inputs: the synthetic dataset and the recording sessions collected with the wearable devices. Each is described in its own subsection.

### Synthetic dataset

A `float32` array of shape **`(2183, 540, 8)`**: subjects × time steps × features. Each time step is 1/3 of a day, so 540 steps cover 180 days. You can download it [here](https://zenodo.org/records/23344939/files/x_test.zip?download=1)

| Index | Feature |
|---|---|---|
| 0 | `carb` | diet |
| 1 | `prot` | diet |
| 2 | `fat` | diet |
| 3 | `intensity` |
| 4 | `duration` |
| 5 | `sex` | |
| 6 | `age` | |
| 7 | `BMI` | |

Note that the daily activity intensity and duration are summarized in the third time step of each day. The first and second time steps have these values set to zero.

### Recording sessions

Recording sessions can be downloaded [here](https://zenodo.org/records/23344939/files/raw_sessions_final.zip?download=1) and are grouped by participant:

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

In the paper, we only used the three `*_heart_rate` fields, which are the personal zone thresholds in bpm (beats per minute).

#### `records/polar_training-session-*.json`: one training session

There are 1,416 files in total. Each recording contains both the reference data (Polar) and the measured data (Garmin or Fitbit).

| Field | Type | Description |
|---|---|---|
| `activity_type` | str | e.g. `OTHER_INDOOR`, `WALKING`, `RUNNING`, `STRENGTH_TRAINING`, … (25 types) |
| `active` | bool | set to true if there is at least one reading greater than 130 bpm |
| `heart_rate_times` | list[str] | Polar (reference) timestamps, ISO-8601, ~1 Hz |
| `heart_rate_values` | list[number] | Polar HR in bpm |
| `associated_device_records.heart_rate_times` | list[str] | Garmin or Fitbit timestamps for the same window (note that these readings are sampled at less than ~1 Hz, so this list has fewer elements than `heart_rate_times`) |
| `associated_device_records.heart_rate_values` | list[number] | Garmin or Fitbit HR in bpm |
| `associated_device_name` | str | `"FitbitFile"`, `"GarminFile"` or `"NoneType"`. The wearable used during the recording session (Fitbit or Garmin; none if no wearable data is associated with the recording) |

#### `sessions_report.csv`, `aggregated_report.csv`, `activities_summary.csv`

These are per-user summaries created during data collection to check data completeness. They are not read by the pipeline.

## Output description

These are the outputs used to create the statistics, tables, and figures in the paper. They are described in three tables, grouped by format.

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
| `summary_<device>.csv` | One row per subject, sorted by \|mean Δ years to prediabetes onset of reference vs perturbed\|. Columns: reference years and risk class, modal measured risk, mean/std/min/max Δ years, `flip_prob` (fraction of draws where the risk class changed) |
| `summary_delta_years.csv` | One row per device. Columns: Δ years to prediabetes onset distribution (mean, std, median, 2.5/97.5 percentiles, min/max), fraction of negative/positive Δs, paired t-test, pooled flip rate |
| `table_activity_cells.csv` | reference vs perturbed intensity/duration, % change, paired t-test, Cohen's d |
| `confusion_onset_lt7.csv` | confusion matrix for subjects with true onset < 7 years, where the measured class is the modal class across draws |

### Figures

| File | Content |
|---|---|
| `error_violin_<device>.png`, `error_box_<device>.png` | Distribution of intensity and duration errors (reference vs perturbed) |
| `confusion_matrices_sample_<device>.png` | 6 per-draw confusion matrices of risk prediction, reference against perturbed |
| `confusion_matrix_pooled_<device>.png` | Sum of all per-draw matrices |
| `violin_ref_vs_<device>_<intensity\|duration>.png` | Physical activity values, reference vs perturbed |
