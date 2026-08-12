"""Client data loader — load raw client data from processed CSVs.

No truncation or alignment — returns the full time range with leading/trailing
NaN preserved.  Each client's sequence-length distribution is reported.
"""

from __future__ import annotations

import yaml
import numpy as np
import pandas as pd
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "fl_code" / "models" / "client_config.yaml"

try:
    from fl_code.config import INPUT_STEPS, PRED_LEN, STRIDE, TRAIN_RATIO
except ImportError:  # direct script execution: python fl_code/data_utils.py
    import sys as _sys
    _sys.path.insert(0, str(ROOT))
    from fl_code.config import INPUT_STEPS, PRED_LEN, STRIDE, TRAIN_RATIO


def load_client_data(client_id: str) -> tuple[pd.DataFrame, dict]:
    """Load a client's raw data.

    1. Look up the client in ``client_config.yaml``.
    2. Load the corresponding processed CSV.
    3. Return the full time range (no truncation).  Leading / trailing NaN
       are preserved — they represent different start/end times across
       sequences.

    Parameters
    ----------
    client_id : str
        e.g. ``"lcl_res_0"``, ``"eld_ind_2"``, ``"steel_ind_0"``.

    Returns
    -------
    df : DataFrame
        Columns: datetime + load sequences + public features + local features.
    info : dict
        Metadata including per-sequence length distribution.
    """
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    dataset_id, client_cfg = _find_client(config, client_id)
    if dataset_id is None:
        raise KeyError(f"Client '{client_id}' not found in {CONFIG_PATH}")

    df = pd.read_csv(
        ROOT / "data" / "processed" / f"{dataset_id}.csv",
        parse_dates=["datetime"],
    )

    seqs = client_cfg["sequences"]
    public_cols = list(config[dataset_id]["public_features"])
    local_cols = list(config[dataset_id].get("local_features", []))

    # If config uses one-hot cat cols but CSV still has category_id,
    # temporarily load category_id for expansion
    _needs_onehot = (
        "cat_residential" in public_cols
        and "category_id" in df.columns
    )
    if _needs_onehot:
        csv_public = [c for c in public_cols if not c.startswith("cat_")] + ["category_id"]
    else:
        csv_public = public_cols

    keep_cols = ["datetime"] + seqs + csv_public + local_cols
    df = df[[c for c in keep_cols if c in df.columns]]

    # Expand category_id → one-hot
    if _needs_onehot:
        cat = df["category_id"].astype(int)
        df["cat_residential"] = (cat == 0).astype(float)
        df["cat_transformer"] = (cat == 1).astype(float)
        df["cat_industrial"]  = (cat == 2).astype(float)
        df = df.drop(columns=["category_id"])

    # Per-sequence length stats
    stats = _seq_length_stats(df, seqs)

    info = {
        "dataset": dataset_id,
        "client_id": client_id,
        "num_sequences": len(seqs),
        "total_rows": len(df),
        "time_range": (str(df["datetime"].iloc[0]), str(df["datetime"].iloc[-1])),
        "public_features": public_cols,
        "local_features": local_cols,
        **stats,
    }
    return df, info


def _seq_length_stats(df: pd.DataFrame, seqs: list[str]) -> dict:
    """Compute per-sequence length distribution."""
    valid_steps = {}
    first_valid_dt = {}
    last_valid_dt = {}
    for s in seqs:
        n = df[s].notna().sum()
        valid_steps[s] = int(n)
        first_idx = df[s].first_valid_index()
        last_idx = df[s].last_valid_index()
        first_valid_dt[s] = str(df.loc[first_idx, "datetime"]) if first_idx is not None else None
        last_valid_dt[s] = str(df.loc[last_idx, "datetime"]) if last_idx is not None else None

    steps = list(valid_steps.values())

    return {
        "valid_steps": {
            "min": int(np.min(steps)),
            "max": int(np.max(steps)),
            "mean": float(np.mean(steps)),
            "median": float(np.median(steps)),
            "p10": float(np.percentile(steps, 10)),
            "p90": float(np.percentile(steps, 90)),
        },
        "valid_days": {
            "min": np.min(steps) / 48,
            "max": np.max(steps) / 48,
            "mean": np.mean(steps) / 48,
        },
        "first_valid_range": (min(first_valid_dt.values()), max(first_valid_dt.values())),
        "last_valid_range": (min(last_valid_dt.values()), max(last_valid_dt.values())),
    }


def preprocess(df: pd.DataFrame, seqs: list[str],
               local_cols: list[str] | None = None,
               train_ratio: float = TRAIN_RATIO) -> tuple[pd.DataFrame, dict]:
    """Normalise load sequences and local features.

    Load (power) columns — per-sequence, per-column:
        1. log1p  →  2. (y - mean) / std.
        μ/σ are computed **only on the training portion** (first
        ``train_ratio`` of valid steps), then applied to all valid steps.

    Local features — per-column:
        (x - mean) / std, likewise μ/σ from training portion only.

    Parameters
    ----------
    df : DataFrame
        From :func:`load_client_data`.
    seqs : list[str]
        Load column names.
    local_cols : list[str] or None
        Local feature column names (can be empty list).
    train_ratio : float
        Fraction of valid steps to use for computing μ/σ (default 0.8).

    Returns
    -------
    df_norm : DataFrame
        Normalised copy of *df* (datetime column unchanged).
    params : dict
        ``{column_name: {"log1p": bool, "mean": float, "std": float}}``.
        Use :func:`inverse_preprocess` to recover original values.
    """
    df_norm = df.copy()
    params = {}
    local_cols = local_cols or []

    # ---- load columns (per-sequence) ----
    for s in seqs:
        valid = df[s].notna()
        if valid.sum() == 0:
            params[s] = {"log1p": True, "mean": 0.0, "std": 1.0}
            continue
        df_norm[s] = df_norm[s].astype(float)

        f = df[s].first_valid_index()
        l = df[s].last_valid_index()
        split = f + int((l - f + 1) * train_ratio)

        # μ/σ from training portion only
        train_mask = valid & (df.index >= f) & (df.index < split)
        y_train = df.loc[train_mask, s].values.astype(float)
        y_train_log = np.log1p(y_train)
        mu, sigma = y_train_log.mean(), y_train_log.std(ddof=0)
        if sigma < 1e-9:
            sigma = 1.0

        # Apply to all valid values (train + test)
        y_all = df.loc[valid, s].values.astype(float)
        y_all_log = np.log1p(y_all)
        df_norm.loc[valid, s] = (y_all_log - mu) / sigma
        params[s] = {"log1p": True, "mean": float(mu), "std": float(sigma)}

    # ---- local feature columns (per-column, μ/σ from first train_ratio of valid rows) ----
    for c in local_cols:
        valid = df[c].notna()
        if valid.sum() == 0:
            params[c] = {"log1p": False, "mean": 0.0, "std": 1.0}
            continue
        df_norm[c] = df_norm[c].astype(float)

        valid_idx = df.index[valid]
        n_train = int(len(valid_idx) * train_ratio)
        train_idx = valid_idx[:n_train]

        x_train = df.loc[train_idx, c].values.astype(float)
        mu, sigma = x_train.mean(), x_train.std(ddof=0)
        if sigma < 1e-9:
            sigma = 1.0

        x_all = df.loc[valid, c].values.astype(float)
        df_norm.loc[valid, c] = (x_all - mu) / sigma
        params[c] = {"log1p": False, "mean": float(mu), "std": float(sigma)}

    return df_norm, params


def inverse_preprocess(df_norm: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Reverse the normalisation applied by :func:`preprocess`."""
    df_out = df_norm.copy()
    for col, p in params.items():
        if col not in df_out.columns:
            continue
        valid = df_out[col].notna()
        if valid.sum() == 0:
            continue
        y = df_out.loc[valid, col].values.astype(float) * p["std"] + p["mean"]
        if p["log1p"]:
            y = np.expm1(y)
        df_out.loc[valid, col] = y
    return df_out


def _find_client(config: dict, client_id: str) -> tuple[str | None, dict | None]:
    for ds_id, ds_cfg in config.items():
        for cid, ccfg in ds_cfg["clients"].items():
            if cid == client_id:
                return ds_id, ccfg
    return None, None


# ============================================================================
# Sliding-window sample generation
# ============================================================================

def make_sliding_windows(
    df: pd.DataFrame,
    seqs: list[str],
    public_cols: list[str],
    input_steps: int = INPUT_STEPS,
    pred_len: int = PRED_LEN,
    stride: int = STRIDE,
    train: bool = True,
    train_ratio: float = TRAIN_RATIO,
    local_cols: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, list[dict]]:
    """Generate {X, y} samples via sliding window over each sequence.

    For each load sequence, a window of ``input_steps + pred_len`` slides
    across its valid (non-NaN) range at ``stride`` intervals.  Windows
    containing any NaN in the load or feature columns are skipped.

    Parameters
    ----------
    df : DataFrame
        From :func:`load_client_data` (optionally after :func:`preprocess`).
    seqs : list[str]
        Load column names.
    public_cols : list[str]
        Public feature column names.
    input_steps : int
        Input window length (default 144 = 3 days @ 30 min).
    pred_len : int
        Forecast horizon (default 6 = 3 hours @ 30 min).
    stride : int
        Step size between consecutive windows.
    train : bool
        True → samples from train portion only; False → test portion.
    train_ratio : float
        Fraction of each sequence's valid steps to use for training.
    local_cols : list[str] or None
        Local feature column names (used by Residual Corrector).

    Returns
    -------
    X : np.ndarray, shape ``(N, public_dim + 1, input_steps)``
        Public features + historical load for the input window.
    y : np.ndarray, shape ``(N, pred_len)``
        Target load for the forecast horizon.
    X_local : np.ndarray or None, shape ``(N, pred_len, local_dim)``
        Local features aligned with the output window (None if no local cols).
    meta : list[dict]
        Per-sample metadata: ``seq_name``, ``window_start`` (row index).
    """
    local_cols = local_cols or []

    # Pre-compute feature matrices for fast slicing
    pub_arr = df[public_cols].values.astype(np.float32)          # (T, pub_dim)
    loc_arr = df[local_cols].values.astype(np.float32) if local_cols else None  # (T, loc_dim)

    X_list, y_list, loc_list, meta = [], [], [], []

    for s in seqs:
        f = df[s].first_valid_index()
        l = df[s].last_valid_index()
        if f is None or l is None:
            continue

        load = df[s].values.astype(np.float32)                   # (T,)
        valid_len = l - f + 1
        split = f + int(valid_len * train_ratio)
        total = input_steps + pred_len

        if train:
            win_start = f
            win_end = split - total + 1      # entire window before split
        else:
            win_start = split
            win_end = l - total + 2          # entire window within [split, l]

        for i in range(max(win_start, 0), max(win_end, 0), stride):
            in_end = i + input_steps
            out_end = in_end + pred_len

            # Skip if load has NaN in window
            if np.isnan(load[i:out_end]).any():
                continue
            # Skip if public features have NaN in input window
            if np.isnan(pub_arr[i:in_end]).any():
                continue
            # Skip if local features have NaN in output window
            if loc_arr is not None and np.isnan(loc_arr[in_end:out_end]).any():
                continue

            X_pub = pub_arr[i:in_end].T                               # (pub_dim, input_steps)
            X_load = load[i:in_end][np.newaxis, :]                     # (1, input_steps)
            X_list.append(np.concatenate([X_pub, X_load], axis=0))     # (pub_dim+1, input_steps)
            y_list.append(load[in_end:out_end])                        # (pred_len,)

            if loc_arr is not None:
                loc_list.append(loc_arr[in_end:out_end])               # (pred_len, loc_dim)

            meta.append({"seq": s, "window_start": i})

    X = np.stack(X_list, axis=0) if X_list else np.empty((0, len(public_cols) + 1, input_steps), dtype=np.float32)
    y = np.stack(y_list, axis=0) if y_list else np.empty((0, pred_len), dtype=np.float32)
    X_local = np.stack(loc_list, axis=0) if loc_list else None

    return X, y, X_local, meta


class PowerDataset:
    """PyTorch Dataset wrapping sliding-window samples.

    Usage::

        X, y, X_local, meta = make_sliding_windows(...)
        ds = PowerDataset(X, y, X_local)
        loader = DataLoader(ds, batch_size=64, shuffle=True)
        for batch_X, batch_y in loader:
            ...

    Parameters
    ----------
    X : np.ndarray, shape ``(N, C, T_in)``
    y : np.ndarray, shape ``(N, T_out)``
    X_local : np.ndarray or None, shape ``(N, T_out, D_local)``
    """

    def __init__(self, X: np.ndarray, y: np.ndarray,
                 X_local: np.ndarray | None = None):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.X_local = torch.from_numpy(X_local) if X_local is not None else None

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        if self.X_local is not None:
            return self.X[idx], self.y[idx], self.X_local[idx]
        return self.X[idx], self.y[idx]


class LazySlidingWindowDataset:
    """Memory-efficient sliding-window Dataset for clients with many sequences.

    Stores only window positions (a few MB) and generates X/y arrays on
    :meth:`__getitem__`.  Suitable for clients like ``lcl_res_0`` (174
    sequences → ~98K windows with stride=48).

    Parameters
    ----------
    df : DataFrame
        Normalised client data (from :func:`preprocess`).
    seqs : list[str]
        Load column names.
    public_cols : list[str]
        Public feature column names.
    input_steps : int
    pred_len : int
    stride : int
    train : bool
    train_ratio : float
    """

    def __init__(self, df: pd.DataFrame, seqs: list[str],
                 public_cols: list[str], input_steps: int = INPUT_STEPS,
                 pred_len: int = PRED_LEN, stride: int = STRIDE,
                 train: bool = True, train_ratio: float = TRAIN_RATIO):
        self.pub_arr = df[public_cols].values.astype(np.float32)
        # Store all load columns as a single 2D array for fast slicing
        self.load_arr = df[seqs].values.astype(np.float32)  # (T, num_seqs)
        self.seq_idx_map = {s: i for i, s in enumerate(seqs)}
        self.input_steps = input_steps
        self.pred_len = pred_len
        self.total = input_steps + pred_len

        # (seq_idx, start_pos) — numpy 存储，比 list[tuple] 省 ~85% 内存
        # （429 万个窗口：tuple 列表 ~480MB → int32 数组 ~69MB）
        window_list: list[tuple[int, int]] = []
        n_skipped = 0

        for si, s in enumerate(seqs):
            col = self.load_arr[:, si]
            f = df[s].first_valid_index()
            l = df[s].last_valid_index()
            if f is None or l is None:
                continue

            valid_len = l - f + 1
            split = f + int(valid_len * train_ratio)

            if train:
                win_start = f
                win_end = split - self.total + 1
            else:
                win_start = split
                win_end = l - self.total + 2

            for i in range(max(win_start, 0), max(win_end, 0), stride):
                in_end = i + input_steps
                out_end = in_end + pred_len
                if np.isnan(col[i:out_end]).any():
                    n_skipped += 1
                    continue
                if np.isnan(self.pub_arr[i:in_end]).any():
                    n_skipped += 1
                    continue
                window_list.append((si, i))

        self.windows = np.asarray(window_list, dtype=np.int32).reshape(-1, 2)

        if n_skipped:
            import sys
            print(f"[LazySlidingWindowDataset] skipped {n_skipped} windows with NaN "
                  f"({len(self.windows)} valid, train={train})", file=sys.stderr)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        si, start = self.windows[idx]
        in_end = start + self.input_steps
        out_end = in_end + self.pred_len

        col = self.load_arr[:, si]
        X_pub = self.pub_arr[start:in_end].T.copy()           # (pub_dim, T_in)
        X_load = col[start:in_end][np.newaxis, :]              # (1, T_in)
        X = np.concatenate([X_pub, X_load], axis=0)            # (C, T_in)
        y = col[in_end:out_end].copy()                         # (T_out,)

        return torch.from_numpy(X), torch.from_numpy(y)


# ============================================================================
# Train / test split
# ============================================================================

def split_train_test(df: pd.DataFrame, seqs: list[str],
                     train_ratio: float = TRAIN_RATIO) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-sequence chronological 80/20 split.

    Each sequence is split independently at ``train_ratio`` of its own valid
    (non-NaN) range.  The two returned DataFrames have the same shape and
    columns as ``df``:

    - **train_df**: test-portion values masked to NaN
    - **test_df**:  train-portion values masked to NaN

    Datetime and feature columns are preserved in both.

    Parameters
    ----------
    df : DataFrame
        From :func:`load_client_data`.
    seqs : list[str]
        Load column names.
    train_ratio : float
        Fraction of each sequence's valid steps to use for training.

    Returns
    -------
    train_df : DataFrame
    test_df : DataFrame
    """
    train_df = df.copy()
    test_df = df.copy()

    for s in seqs:
        f = df[s].first_valid_index()
        l = df[s].last_valid_index()
        if f is None or l is None:
            continue
        valid_len = l - f + 1
        split = f + int(valid_len * train_ratio)

        train_df.loc[split:l, s] = float("nan")
        test_df.loc[f:split - 1, s] = float("nan")

    return train_df, test_df


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python data_utils.py <client_id>")
        print("Example: python data_utils.py lcl_res_0")
        sys.exit(1)

    client_id = sys.argv[1]
    df, info = load_client_data(client_id)

    print(f"Client: {info['client_id']}")
    print(f"Dataset: {info['dataset']}")
    print(f"Total rows: {info['total_rows']}")
    print(f"Time range: {info['time_range'][0]} ~ {info['time_range'][1]}")
    print(f"Sequences: {info['num_sequences']}")
    print(f"Shape: {df.shape}")
    print(f"Public features: {info['public_features']}")
    print(f"Local features: {info['local_features']}")
    print()
    print("--- Valid steps per sequence ---")
    for k, v in info["valid_steps"].items():
        print(f"  {k}: {v:.0f}" if isinstance(v, float) else f"  {k}: {v}")
    print()
    print("--- Valid days per sequence ---")
    for k, v in info["valid_days"].items():
        print(f"  {k}: {v:.1f}")
    print()
    print(f"First valid range: {info['first_valid_range'][0]} ~ {info['first_valid_range'][1]}")
    print(f"Last valid range:  {info['last_valid_range'][0]} ~ {info['last_valid_range'][1]}")
    print()
    print(df.head(5).to_string())
    print()

    # Preprocessing
    seq_cols = [c for c in df.columns if c not in ("datetime",)
                and c not in info["public_features"]
                and c not in info["local_features"]]
    df_norm, norm_params = preprocess(df, seq_cols, info["local_features"])
    # Roundtrip check (only on non-NaN cells)
    df_rt = inverse_preprocess(df_norm, norm_params)
    rt_err = (df[seq_cols] - df_rt[seq_cols]).abs().max().max()

    print("--- Preprocessing ---")
    print(f"Roundtrip max error (should be ~0): {rt_err:.2e}")
    load_params = {k: v for k, v in norm_params.items() if v["log1p"]}
    print(f"Load sequences normalised: {len(load_params)} (log1p + mean/std)")
    feat_params = {k: v for k, v in norm_params.items() if not v["log1p"]}
    if feat_params:
        print(f"Local features normalised: {len(feat_params)} (mean/std)")
        for k, v in feat_params.items():
            print(f"  {k}: mean={v['mean']:.3f}, std={v['std']:.3f}")
    print()
    print("Normalised head:")
    print(df_norm.head(5).to_string())
    print()

    # Sliding-window sample generation
    X_train, y_train, _, meta_train = make_sliding_windows(
        df_norm, seq_cols, info["public_features"],
        stride=48, train=True, local_cols=info["local_features"],
    )
    X_test, y_test, _, meta_test = make_sliding_windows(
        df_norm, seq_cols, info["public_features"],
        stride=48, train=False, local_cols=info["local_features"],
    )
    print("--- Sliding windows (stride=48) ---")
    print(f"Train samples: {len(X_train)}, X shape: {X_train.shape}, y shape: {y_train.shape}")
    print(f"Test samples:  {len(X_test)}, X shape: {X_test.shape}, y shape: {y_test.shape}")
