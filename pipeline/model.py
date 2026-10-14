"""Step 7: load the trained LSTM surrogate and run it -> years-to-prediabetes
per subject, then classify into a risk band.

The model itself (architecture + trained weights) is out of scope for this
rebuild -- reused as-is from che_fatica_praes/model_I2O.checkpoint, per the
"copy x_test.npy and the checkpoint in, don't retrain" decision.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn

from constants import (
    CAP_YEARS,
    GBL_OUTPUT_IDX,
    RISK_HIGH_MAX,
    RISK_MEDIUM_MAX,
    STARTING_VALUE,
    THRESHOLD_PREDIABETES,
    WARM_STEPS,
)


class LSTM_I2O(nn.Module):
    def __init__(self, input_size, n_hidden, n_layers, output_size, device):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.device = device
        self.lstm = nn.LSTM(input_size, n_hidden, n_layers, batch_first=True)
        self.linear = nn.Linear(n_hidden, output_size)
        self.to(device)

    def forward(self, x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.n_hidden, device=self.device)
        c0 = torch.zeros(self.n_layers, x.size(0), self.n_hidden, device=self.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.linear(out)


def load_surrogate(checkpoint_path: Path, torch_device: torch.device):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # sklearn scaler pickled with an older version
        ckpt = joblib.load(checkpoint_path)

    model = LSTM_I2O(
        ckpt["input_size"], ckpt["n_hidden"], ckpt["n_layers"], ckpt["output_size"], torch_device,
    ).to(torch_device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def _predict_glucose_baselevel(inputs, model, ckpt, torch_device, batch_size=512):
    x_scaler, y_scaler = ckpt["x_scaler"], ckpt["y_scaler"]
    samples, timesteps, n_features = inputs.shape
    x_scaled = x_scaler.transform(inputs.reshape(-1, n_features)).reshape(samples, timesteps, n_features)

    chunks = []
    with torch.no_grad():
        for start in range(0, samples, batch_size):
            chunk = x_scaled[start:start + batch_size]
            u = torch.tensor(chunk, dtype=torch.float32, device=torch_device)
            y = model(u).cpu().numpy()[:, WARM_STEPS:, :]  # drop warm-up
            b, steps, n_out = y.shape
            y_back = y_scaler.inverse_transform(y.reshape(-1, n_out)).reshape(b, steps, n_out)
            chunks.append(y_back[:, :, GBL_OUTPUT_IDX])
    return np.concatenate(chunks, axis=0)


def years_to_prediabetes(inputs, model, ckpt, torch_device, batch_size=512, cap_years=CAP_YEARS):
    """Fit glucose-baselevel trajectory to a line through the origin; years =
    (threshold - start) / slope. Never-onset and beyond-cap both map to cap_years."""
    gbl = _predict_glucose_baselevel(inputs, model, ckpt, torch_device, batch_size)
    steps = gbl.shape[1]

    n_total = steps + WARM_STEPS
    t_years = np.linspace(WARM_STEPS / 3, n_total / 3, steps) / 365
    t2_sum = np.sum(t_years ** 2)

    gbl_rel = gbl - STARTING_VALUE
    slopes = np.dot(gbl_rel, t_years) / t2_sum

    rise = THRESHOLD_PREDIABETES - STARTING_VALUE
    years = np.full(slopes.shape, np.nan)
    nonzero = np.abs(slopes) > 1e-12
    years[nonzero] = rise / slopes[nonzero]

    years[np.isnan(years)] = cap_years
    years[years < 0.0] = cap_years
    years[years > cap_years] = cap_years
    return years


def risk_class(years):
    years = np.asarray(years, dtype=float)
    out = np.empty(years.shape, dtype=object)
    out[:] = "low"
    out[years <= RISK_MEDIUM_MAX] = "medium"
    out[years <= RISK_HIGH_MAX] = "high"
    return out
