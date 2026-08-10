"""Data utilities for baseline and client training.

Reusable across Phase 2 (baseline) and Phase 3 (FL client training).
Handles: CSV loading, sequence building, feature preprocessing,
dataset splitting, and lazy sliding-window generation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INPUT_STEPS = 336   # 7 days @30min
PRED_LEN = 48       # 1 day @30min
WINDOW = INPUT_STEPS + PRED_LEN  # 384
DEFAULT_STRIDE = 6  # 3 hours @30min (6 steps)
TRAIN_RATIO = 0.8

# ---------------------------------------------------------------------------
# Feature categorisation per dataset
# unbounded → log1p + StandardScaler
# bounded   → identity (keep as-is)
# ---------------------------------------------------------------------------
FEATURE_CONFIG: dict[str, dict[str, list[str]]] = {
    "steel_ind": {
        "unbounded": ["co2", "lagging_reactive_power"],
        "bounded": ["lagging_pf", "load_type"],
    },
    "tetouan_city": {
        "unbounded": ["temperature", "humidity", "wind_speed",
                      "general_diffuse_flow", "diffuse_flow"],
        "bounded": [],
    },
    "lcl_res": {"unbounded": [], "bounded": []},
    "eld_ind": {"unbounded": [], "bounded": []},
}

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class SequenceData:
    """One sequence ready for window slicing."""
    name: str
    # target (label) — y_log = log1p(raw), no z-score
    y_log: np.ndarray                # (valid_len,) float32
    # historical-load feature — log1p(raw) then z-scored (fit on train portion)
    load_z: np.ndarray               # (valid_len,) float32
    # local features — preprocessed arrays keyed by feature name
    local_feats: dict[str, np.ndarray] = field(default_factory=dict)
    # window-start row indices (absolute, into the CSV row space)
    train_starts: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    test_starts: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))


# ---------------------------------------------------------------------------
# Config / CSV loading
# ---------------------------------------------------------------------------
def load_client_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dataset_df(csv_path: Path, seq_cols: list[str],
                    pub_cols: list[str],
                    local_cols: list[str] | None = None) -> pd.DataFrame:
    """Read CSV with only the columns we need."""
    cols = ["datetime"] + seq_cols + pub_cols
    if local_cols:
        cols += local_cols
    df = pd.read_csv(csv_path, usecols=cols, low_memory=False)
    df["datetime"] = pd.to_datetime(df["datetime"])
    for c in pub_cols:
        assert df[c].notna().all(), f"{csv_path.name}: public feature '{c}' has NaN"
    return df


# ---------------------------------------------------------------------------
# Per-feature preprocessing helpers
# ---------------------------------------------------------------------------
def _fit_scaler_on_train(arr: np.ndarray, lo: int, t_star: int
                         ) -> tuple[float, float]:
    """Compute mean/std over the train portion [lo, t_star)."""
    train_slice = arr[lo:t_star]
    m = float(np.nanmean(train_slice))
    s = float(np.nanstd(train_slice))
    if s < 1e-8:
        s = 1.0  # constant feature → no-op scaling
    return m, s


def _preprocess_unbounded(arr: np.ndarray, lo: int, t_star: int, hi: int
                          ) -> np.ndarray:
    """log1p → fit scaler on train portion → z-score valid range [lo, hi)."""
    tformed = np.log1p(arr)  # handles [0, ∞)
    m, s = _fit_scaler_on_train(tformed, lo, t_star)
    z = tformed.copy()
    z[lo:hi] = (z[lo:hi] - m) / s
    return z.astype(np.float32)


def _preprocess_bounded(arr: np.ndarray) -> np.ndarray:
    """Identity — keep [0,1] features as-is."""
    return arr.astype(np.float32)


# ---------------------------------------------------------------------------
# Sequence building
# ---------------------------------------------------------------------------
def _find_valid_range(series: np.ndarray) -> tuple[int, int]:
    valid = np.where(np.isfinite(series))[0]
    if len(valid) == 0:
        return 0, 0
    return int(valid[0]), int(valid[-1]) + 1


def build_sequence_public(series: pd.Series, name: str,
                          stride: int = DEFAULT_STRIDE) -> SequenceData | None:
    """Build a sequence with only public features (incl. historical load).

    Preprocessing:
      - Label y:        log1p(raw)           → y_log
      - Historical load: log1p(raw) + z-score → load_z
    """
    arr = series.to_numpy(dtype=np.float32)
    lo, hi = _find_valid_range(arr)
    if hi - lo < WINDOW:
        return None

    # ffill within valid range
    arr[lo:hi] = pd.Series(arr[lo:hi]).ffill().to_numpy(dtype=np.float32)

    # label: log1p only
    y_log = np.log1p(arr)
    y_log[:lo] = np.nan
    y_log[hi:] = np.nan

    # historical-load feature: log1p + z-score
    t_star = lo + round(TRAIN_RATIO * (hi - lo))
    load_log = np.log1p(arr)
    lm, ls = _fit_scaler_on_train(load_log, lo, t_star)
    load_z = load_log.copy()
    load_z[lo:hi] = (load_z[lo:hi] - lm) / ls
    load_z[:lo] = np.nan
    load_z[hi:] = np.nan

    # window starts
    train_starts, test_starts = _make_window_starts(lo, hi, t_star, y_log, stride)

    return SequenceData(
        name=name, y_log=y_log.astype(np.float32),
        load_z=load_z.astype(np.float32),
        train_starts=train_starts, test_starts=test_starts,
    )


def build_sequence_all(series: pd.Series, name: str, ds_id: str,
                       local_df: pd.DataFrame,
                       stride: int = DEFAULT_STRIDE) -> SequenceData | None:
    """Build a sequence with public features + local features.

    Same as build_sequence_public but also preprocesses local features
    according to FEATURE_CONFIG.
    """
    sd = build_sequence_public(series, name, stride)
    if sd is None:
        return None

    # Determine t_star from the original array for per-feature split
    arr = series.to_numpy(dtype=np.float32)
    lo, hi = _find_valid_range(arr)
    t_star = lo + round(TRAIN_RATIO * (hi - lo))

    fc = FEATURE_CONFIG.get(ds_id, {"unbounded": [], "bounded": []})

    for feat_name in fc["unbounded"]:
        if feat_name in local_df.columns:
            feat_arr = local_df[feat_name].to_numpy(dtype=np.float32)
            feat_arr[lo:hi] = pd.Series(feat_arr[lo:hi]).ffill().to_numpy(dtype=np.float32)
            sd.local_feats[feat_name] = _preprocess_unbounded(feat_arr, lo, t_star, hi)

    for feat_name in fc["bounded"]:
        if feat_name in local_df.columns:
            feat_arr = local_df[feat_name].to_numpy(dtype=np.float32)
            feat_arr[lo:hi] = pd.Series(feat_arr[lo:hi]).ffill().to_numpy(dtype=np.float32)
            sd.local_feats[feat_name] = _preprocess_bounded(feat_arr)

    return sd


# ---------------------------------------------------------------------------
# Window start generation
# ---------------------------------------------------------------------------
def _make_window_starts(lo: int, hi: int, t_star: int,
                        y_log: np.ndarray,
                        stride: int = DEFAULT_STRIDE) -> tuple[np.ndarray, np.ndarray]:
    """Generate train/test window start indices, dropping any that touch NaN."""
    train_starts = np.arange(lo, t_star - WINDOW + 1, stride, dtype=np.int32)
    test_starts = np.arange(max(lo, t_star - INPUT_STEPS),
                            hi - WINDOW + 1, stride, dtype=np.int32)

    def _ok(starts: np.ndarray) -> np.ndarray:
        mask = np.ones(len(starts), dtype=bool)
        for i, s in enumerate(starts):
            if np.isnan(y_log[s:s + WINDOW]).any():
                mask[i] = False
        return starts[mask]

    return _ok(train_starts), _ok(test_starts)


# ---------------------------------------------------------------------------
# Dataset splitting
# ---------------------------------------------------------------------------
def split_chronological(seqs: list[SequenceData], ratio: float = TRAIN_RATIO
                        ) -> tuple[list[SequenceData], list[SequenceData]]:
    """For narrow datasets: each sequence is already split internally.
    Just return all seqs for training (their .train_starts are used) and
    all seqs for testing (their .test_starts are used)."""
    return seqs, seqs  # same sequences, different window sets


def split_by_sequence(seqs: list[SequenceData], ratio: float = TRAIN_RATIO,
                      seed: int = 42) -> tuple[list[SequenceData], list[SequenceData]]:
    """For wide datasets: random 80/20 split of *sequences* themselves."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(seqs))
    n_train = round(len(seqs) * ratio)
    train_seqs = [seqs[i] for i in idx[:n_train]]
    test_seqs = [seqs[i] for i in idx[n_train:]]
    return train_seqs, test_seqs


# ---------------------------------------------------------------------------
# Lazy WindowDataset
# ---------------------------------------------------------------------------
class WindowDataset(Dataset):
    """Lazily slices windows from a list of SequenceData + public-feature matrix.

    Parameters
    ----------
    seqs : list[SequenceData]
        Only train_starts are used (training mode).
    pub : np.ndarray (N_rows, 8)
        Shared public time-feature matrix (sin/cos/category_id).
    in_channels : int
        9 for public mode, 9+D_local for all mode.
    local_feat_names : list[str]
        Ordered list of local feature names (determines channel order).
    """

    def __init__(self, seqs: list[SequenceData], pub: np.ndarray,
                 in_channels: int = 9,
                 local_feat_names: list[str] | None = None):
        self.seqs = seqs
        self.pub = pub.astype(np.float32)
        self.in_channels = in_channels
        self.local_feat_names = local_feat_names or []
        self.cumlen = np.cumsum([len(s.train_starts) for s in seqs],
                                dtype=np.int64)

    def __len__(self) -> int:
        return int(self.cumlen[-1]) if len(self.cumlen) > 0 else 0

    def __getitem__(self, idx: int):
        si = int(np.searchsorted(self.cumlen, idx, side="right"))
        j = idx - int(self.cumlen[si - 1]) if si > 0 else idx
        s = self.seqs[si]
        start = int(s.train_starts[j])

        # Channel layout: [public_features (8), historical_load (1), local_feat_0, ...]
        pub_slice = self.pub[start:start + INPUT_STEPS]          # (336, 8)
        load_slice = s.load_z[start:start + INPUT_STEPS]         # (336,)
        load_slice = np.nan_to_num(load_slice, nan=0.0)

        channels = [pub_slice, load_slice.reshape(-1, 1)]
        for fn in self.local_feat_names:
            feat = s.local_feats.get(fn)
            if feat is not None:
                sl = feat[start:start + INPUT_STEPS]
                sl = np.nan_to_num(sl, nan=0.0)
                channels.append(sl.reshape(-1, 1))
            else:
                channels.append(np.zeros((INPUT_STEPS, 1), dtype=np.float32))

        x = np.concatenate(channels, axis=1).T  # (in_channels, 336)
        y = s.y_log[start + INPUT_STEPS:start + WINDOW].copy()  # (48,)

        return (torch.from_numpy(np.ascontiguousarray(x)),
                torch.from_numpy(y))
