"""Training and evaluation utilities for TCN power forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# ============================================================================
# Training
# ============================================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
) -> float:
    """Train one epoch with MAE loss.

    The loader yields ``(X, y)`` or ``(X, y, X_local)`` tuples.  *X_local*
    is ignored during Global TCN training (Phase 2).

    Returns
    -------
    float
        Average epoch loss.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        X = batch[0].to(device)
        y = batch[1].to(device)

        optimizer.zero_grad()
        y_pred = model(X)                     # (B, pred_len)
        loss = nn.functional.l1_loss(y_pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ============================================================================
# Evaluation — rolling forecast
# ============================================================================

@torch.no_grad()
def evaluate(
    model: nn.Module,
    df: pd.DataFrame,
    seqs: list[str],
    public_cols: list[str],
    input_steps: int = 1440,
    pred_len: int = 336,
    stride: int = 48,
    train_ratio: float = 0.8,
    device: str = "cpu",
) -> dict:
    """Rolling-forecast evaluation over all sequences.

    For each sequence, starting at its test portion, repeatedly:
      1. Feed the most recent *input_steps* real data as input.
      2. Predict the next *pred_len* steps.
      3. Slide forward by *stride* steps.

    All (prediction, actual) pairs are collected and standard regression
    metrics are computed across the full test horizon.

    Parameters
    ----------
    model : nn.Module
        Global TCN model with signature ``(B, C, T_in) -> (B, T_out)``.
    df : DataFrame
        Preprocessed client data (from :func:`~fl_code.data_utils.preprocess`).
    seqs : list[str]
        Load column names.
    public_cols : list[str]
        Public feature column names.
    input_steps : int
    pred_len : int
    stride : int
    train_ratio : float

    Returns
    -------
    dict
        Keys: ``mae``, ``mse``, ``rmse``, ``r2``, ``predictions``, ``actuals``.
    """
    model.eval()

    pub_arr = df[public_cols].values.astype(np.float32)
    all_preds = []
    all_actuals = []

    for s in seqs:
        f = df[s].first_valid_index()
        l = df[s].last_valid_index()
        if f is None or l is None:
            continue

        load = df[s].values.astype(np.float32)
        valid_len = l - f + 1
        split = f + int(valid_len * train_ratio)

        pos = split
        while pos + input_steps + pred_len <= l + 1:
            # Build input
            X_pub = pub_arr[pos:pos + input_steps].T                     # (pub_dim, T_in)
            X_load = load[pos:pos + input_steps][np.newaxis, :]          # (1, T_in)
            X = np.concatenate([X_pub, X_load], axis=0)                  # (C, T_in)
            X_t = torch.from_numpy(X).unsqueeze(0).to(device)            # (1, C, T_in)

            pred = model(X_t).squeeze(0).cpu().numpy()                   # (T_out,)
            actual = load[pos + input_steps:pos + input_steps + pred_len]

            all_preds.append(pred)
            all_actuals.append(actual)

            pos += stride

    if not all_preds:
        return {"mae": float("nan"), "mse": float("nan"), "rmse": float("nan"),
                "r2": float("nan"), "predictions": None, "actuals": None}

    preds = np.concatenate(all_preds)
    actuals_arr = np.concatenate(all_actuals)

    valid = ~np.isnan(actuals_arr)
    preds = preds[valid]
    actuals_arr = actuals_arr[valid]

    mae = np.mean(np.abs(preds - actuals_arr))
    mse = np.mean((preds - actuals_arr) ** 2)
    rmse = np.sqrt(mse)
    ss_res = np.sum((actuals_arr - preds) ** 2)
    ss_tot = np.sum((actuals_arr - actuals_arr.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2),
        "predictions": preds,
        "actuals": actuals_arr,
    }


# ============================================================================
# Visualization
# ============================================================================

def plot_forecast(
    model: nn.Module,
    df: pd.DataFrame,
    seq_name: str,
    public_cols: list[str],
    input_steps: int = 1440,
    pred_len: int = 336,
    stride: int = 48,
    display_steps: int = 672,
    train_ratio: float = 0.8,
    device: str = "cpu",
    title: str | None = None,
    ax: plt.Axes | None = None,
):
    """Plot predicted vs actual for *display_steps* (default 14 days).

    Uses the same rolling-forecast protocol as :func:`evaluate`: at each
    step, predicts *pred_len* steps ahead, keeps only the first *stride*
    points (the "new" information), then slides forward by *stride*.  The
    resulting curve is stitch-free — each timestep is predicted exactly once.

    Parameters
    ----------
    model : nn.Module
    df : DataFrame
        Preprocessed client data.
    seq_name : str
        Single load column name to visualise.
    public_cols : list[str]
    input_steps : int
    pred_len : int
    stride : int
        Step size — must match the stride used in training / evaluation.
    display_steps : int
        Number of 30-min steps to show (default 672 = 14 days).
    train_ratio : float
    device : str
    title : str or None
    ax : matplotlib Axes or None

    Returns
    -------
    ax : matplotlib Axes
    """
    pub_arr = df[public_cols].values.astype(np.float32)
    load = df[seq_name].values.astype(np.float32)

    f = df[seq_name].first_valid_index()
    l = df[seq_name].last_valid_index()
    if f is None or l is None:
        raise ValueError(f"{seq_name} has no valid data")

    valid_len = l - f + 1
    split = f + int(valid_len * train_ratio)

    # Rolling forecast: slide by stride, keep first "stride" points each time
    pos = split
    pred_pieces = []
    while len(pred_pieces) < display_steps and pos + input_steps + pred_len <= l + 1:
        X_pub = pub_arr[pos:pos + input_steps].T
        X_load = load[pos:pos + input_steps][np.newaxis, :]
        X = np.concatenate([X_pub, X_load], axis=0)
        X_t = torch.from_numpy(X).unsqueeze(0).to(device)

        pred = model(X_t).squeeze(0).detach().cpu().numpy()  # (pred_len,)
        take = min(stride, display_steps - len(pred_pieces))
        pred_pieces.append(pred[:take])
        pos += stride

    preds = np.concatenate(pred_pieces)
    actuals = load[split:split + len(preds)]

    t = np.arange(len(preds)) * 0.5 / 24   # 30-min steps → days

    if ax is None:
        _, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t, actuals, label="Actual", linewidth=1.0, color="#1f77b4")
    ax.plot(t, preds, label="Predicted", linewidth=1.0, color="#ff7f0e", alpha=0.85)
    ax.set_xlabel("Days from test start")
    ax.set_ylabel("Normalised load")
    ax.set_title(title or f"{seq_name} — {display_steps // 48}-day forecast")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax
