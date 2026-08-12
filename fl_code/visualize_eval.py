"""Interactive rolling-forecast visualiser (tkinter GUI).

Workflow:
  1. pick a client (from ``client_config.yaml``)
  2. pick one or more load sequences
  3. pick a model — all available checkpoints are auto-discovered:
       - Global TCN only (Phase 2)
       - Global + Corrector (Phase 3, non-DP)
       - Global + DP Corrector σ=... (Phase 4, one option per noise level)
  4. rolling forecast; predictions are **de-normalised** back to physical
     units and plotted against the actuals for the first 7 days of the test
     split

Usage::

    python -m fl_code.visualize_eval
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml

import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from fl_code.data_utils import load_client_data, preprocess
from fl_code.models import (
    TCNConfig, CorrectorConfig,
    build_tcn, build_corrector,
)
from fl_code.config import INPUT_STEPS, PRED_LEN, STRIDE, DISPLAY_STEPS

ROOT = Path(__file__).resolve().parents[1]
CLIENT_CONFIG_PATH = ROOT / "fl_code" / "models" / "client_config.yaml"
BASELINE_DIR = ROOT / "fl_code" / "baseline_outputs"
PERSONALIZED_DIR = ROOT / "fl_code" / "personalized_outputs"
DP_DIR = ROOT / "fl_code" / "dp_outputs"
GLOBAL_MODEL_PATH = BASELINE_DIR / "best_global_tcn.pt"


# ---------------------------------------------------------------------------
# Metrics (computed in raw physical units, GUI-independent)
# ---------------------------------------------------------------------------

def _metrics(actuals: np.ndarray, preds: np.ndarray) -> dict | None:
    """MAE / MSE / RMSE / R² on valid (non-NaN) pairs, in raw units."""
    valid = ~np.isnan(actuals) & ~np.isnan(preds)
    a, p = actuals[valid], preds[valid]
    if len(a) == 0:
        return None
    mae = float(np.mean(np.abs(p - a)))
    mse = float(np.mean((p - a) ** 2))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum((a - p) ** 2))
    ss_tot = float(np.sum((a - a.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}


# ---------------------------------------------------------------------------
# Rolling forecast (GUI-independent, testable)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _rolling_forecast(model, corrector, df_norm, seq, public_cols, local_cols,
                      start, n_steps, p: dict, device="cpu") -> np.ndarray:
    """Continuous rolling forecast over ``[start, start + n_steps)``.

    The first prediction consumes the 144-step window ending at *start*,
    so predictions are time-aligned with the actuals from *start* onward.
    Every *STRIDE* steps the model re-predicts the next *PRED_LEN* steps
    (window slides by 6 steps → fully covered, stitch-free curve).

    Returns de-normalised predictions (raw physical units), NaN where no
    prediction is available (e.g. invalid data).
    """
    pub_arr = df_norm[public_cols].values.astype(np.float32)
    loc_arr = df_norm[local_cols].values.astype(np.float32) if local_cols else None
    load = df_norm[seq].values.astype(np.float32)

    preds_norm = np.full(n_steps, np.nan, dtype=np.float32)
    prev_residual = np.zeros(PRED_LEN, dtype=np.float32)
    local_dim = len(local_cols)

    pos = start - INPUT_STEPS
    while pos + INPUT_STEPS + PRED_LEN <= start + n_steps:
        win = load[pos:pos + INPUT_STEPS]
        if np.isnan(win).any():
            break

        X_pub = pub_arr[pos:pos + INPUT_STEPS].T
        X_load = win[np.newaxis, :]
        X = np.concatenate([X_pub, X_load], axis=0)
        X_t = torch.from_numpy(X).unsqueeze(0).to(device)
        y_pre = model(X_t).squeeze(0).cpu().numpy()          # (PRED_LEN,)

        if corrector is not None:
            res_t = torch.from_numpy(prev_residual).unsqueeze(0).to(device)
            if local_dim > 0:
                x_loc = loc_arr[pos + INPUT_STEPS:pos + INPUT_STEPS + PRED_LEN]
                x_loc_t = torch.from_numpy(x_loc).unsqueeze(0).to(device)
            else:
                x_loc_t = None
            yp_t = torch.from_numpy(y_pre).unsqueeze(0).to(device)
            e = corrector(yp_t, res_t, x_loc_t).squeeze(0).cpu().numpy()  # (PRED_LEN, 3)
            y_final = y_pre + e[:, 1]                         # P50
        else:
            y_final = y_pre

        out_idx = pos + INPUT_STEPS - start
        take = min(PRED_LEN, n_steps - out_idx)
        if take <= 0:
            break
        preds_norm[out_idx:out_idx + take] = y_final[:take]

        actual = load[pos + INPUT_STEPS:pos + INPUT_STEPS + PRED_LEN]
        prev_residual = actual - y_pre
        pos += STRIDE

    # De-normalise (load sequences always use log1p + z-score)
    return np.expm1(preds_norm * p["std"] + p["mean"])


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class EvalVisualizer:
    """tkinter GUI: client picker → sequence picker → rolling forecast plot."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.global_model = None
        self.current: dict | None = None
        self.model_map: dict[str, Path | None] = {}

        root.title("Rolling-Forecast Visualiser — Global TCN ± Corrector")
        root.geometry("1180x820")

        with open(CLIENT_CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        self.client_ids = [cid for ds in cfg.values() for cid in ds["clients"]]

        self._build_ui()
        self.status_var.set(f"{len(self.client_ids)} clients | device={self.device}")

    # ---- UI construction ----

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill="x")

        ttk.Label(top, text="Client:").pack(side="left")
        self.client_var = tk.StringVar()
        self.client_cb = ttk.Combobox(
            top, textvariable=self.client_var, values=self.client_ids,
            state="readonly", width=24,
        )
        self.client_cb.pack(side="left", padx=6)
        self.client_cb.bind("<<ComboboxSelected>>", self._on_client_selected)

        self.status_var = tk.StringVar(value="Select a client")
        ttk.Label(top, textvariable=self.status_var,
                  foreground="#555555").pack(side="left", padx=12)

        mid = ttk.Frame(self.root, padding=8)
        mid.pack(fill="x")

        ttk.Label(mid, text="Model:").pack(side="left")
        self.model_var = tk.StringVar()
        self.model_cb = ttk.Combobox(mid, textvariable=self.model_var,
                                     state="readonly", width=34)
        self.model_cb.pack(side="left", padx=6)

        ttk.Label(mid, text="Sequences (multi-select):").pack(side="left")
        self.seq_list = tk.Listbox(mid, selectmode="extended",
                                   height=6, width=52)
        self.seq_list.pack(side="left", padx=6)

        ttk.Button(mid, text="Plot 7-day forecast",
                   command=self._plot).pack(side="left", padx=8)

        self.fig = Figure(figsize=(11, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True,
                                         padx=8, pady=(0, 8))

    # ---- model detection ----

    @staticmethod
    def _detect_models(cid: str) -> list[tuple[str, Path | None]]:
        """Auto-discover available models for a client.

        Returns ``[(display_label, corrector_path_or_None)]`` — the first
        entry is always the Global TCN alone.  Scans Phase 2 (baseline),
        Phase 3 (personalized) and Phase 4 (DP) output directories.
        """
        options = [("Global TCN (Phase 2)", None)]

        p3 = PERSONALIZED_DIR / f"corrector_{cid}.pt"
        if p3.exists():
            options.append((f"Global + Corrector (Phase 3)", p3))

        if DP_DIR.exists():
            nodp = DP_DIR / f"corrector_{cid}_nodp.pt"
            if nodp.exists():
                options.append(("Global + Non-DP Corrector (Phase 4)", nodp))
            for p in sorted(DP_DIR.glob(f"corrector_{cid}_dp_sigma*.pt")):
                sigma = p.stem.rsplit("sigma", 1)[1].replace("_", ".")
                options.append(
                    (f"Global + DP Corrector σ={sigma} (Phase 4)", p))

        return options

    @staticmethod
    def _infer_rc_type(state_dict: dict) -> str:
        """Infer Corrector architecture from its state_dict keys."""
        keys = state_dict.keys()
        if any(k.startswith("corrector.") for k in keys):
            return "tcn"
        if any(k.startswith("lstm.") for k in keys):
            return "lstm"
        return "mlp"

    def _load_corrector(self, path: Path, local_dim: int):
        """Load a Corrector checkpoint (Phase 3 or 4) with rc_type auto-detect.

        Also strips an eventual ``_module.`` prefix from Opacus
        GradSampleModule state_dicts.
        """
        sd = torch.load(path, map_location="cpu", weights_only=True)
        if any(k.startswith("_module.") for k in sd):
            sd = {k.replace("_module.", "", 1): v for k, v in sd.items()}
        rc_type = self._infer_rc_type(sd)
        corrector = build_corrector(
            CorrectorConfig(rc_type=rc_type, local_feat_dim=local_dim))
        corrector.load_state_dict(sd)
        corrector.eval()
        return corrector.to(self.device)

    # ---- events ----

    def _on_client_selected(self, _event=None):
        cid = self.client_var.get()
        if not cid:
            return
        self.status_var.set(f"Loading {cid} ...")
        self.root.update_idletasks()
        try:
            df, info = load_client_data(cid)
            feat = set(info["public_features"] + info["local_features"])
            seqs = [c for c in df.columns if c not in feat and c != "datetime"]
            df_norm, params = preprocess(df, seqs, info["local_features"])

            self.current = {
                "df": df,
                "df_norm": df_norm,
                "params": params,
                "seqs": seqs,
                "public_cols": info["public_features"],
                "local_cols": info["local_features"],
            }
            self.seq_list.delete(0, "end")
            for s in seqs:
                self.seq_list.insert("end", s)

            self._ensure_global_model()

            # auto-detect Phase 2/3/4 models
            self.model_map = dict(self._detect_models(cid))
            labels = list(self.model_map.keys())
            self.model_cb["values"] = labels
            self.model_var.set(labels[0])

            n_models = len(labels) - 1
            self.status_var.set(
                f"{cid}: {len(seqs)} sequences, {n_models} Corrector(s) "
                f"found | {info.get('valid_days', '?')} valid days")
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))
            self.status_var.set("Load failed")

    def _ensure_global_model(self):
        if self.global_model is not None:
            return
        if not GLOBAL_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Global TCN not found: {GLOBAL_MODEL_PATH}\n"
                f"Run Phase 2 first: python -m fl_code.train_baseline")
        model = build_tcn(TCNConfig()).to(self.device)
        model.load_state_dict(torch.load(GLOBAL_MODEL_PATH, map_location=self.device,
                                         weights_only=True))
        model.eval()
        self.global_model = model

    # ---- plotting ----

    def _plot(self):
        if self.current is None:
            messagebox.showwarning("No client", "Select a client first.")
            return
        sel = [self.current["seqs"][i] for i in self.seq_list.curselection()]
        if not sel:
            messagebox.showwarning("No sequence",
                                   "Select at least one sequence.")
            return

        label = self.model_var.get()
        path = self.model_map.get(label) if label else None
        corrector = (self._load_corrector(path, len(self.current["local_cols"]))
                     if path else None)

        self.status_var.set(f"Forecasting {len(sel)} sequence(s) ...")
        self.root.update_idletasks()

        self.fig.clear()
        axes = self.fig.subplots(len(sel), 1, sharex=True, squeeze=False)[:, 0]
        self.fig.suptitle(
            f"{self.client_var.get()} — rolling forecast, first 7 days of test split",
            fontsize=13)

        for ax, seq in zip(axes, sel):
            df = self.current["df"]
            f = df[seq].first_valid_index()
            l = df[seq].last_valid_index()
            split = f + int((l - f + 1) * 0.8)
            start_dt = str(df["datetime"].iloc[split])[:19]

            preds = _rolling_forecast(
                self.global_model, corrector, self.current["df_norm"], seq,
                self.current["public_cols"], self.current["local_cols"],
                split, DISPLAY_STEPS, self.current["params"][seq], self.device,
            )
            actuals = df[seq].values[split:split + DISPLAY_STEPS].astype(float)
            t = np.arange(DISPLAY_STEPS) * 0.5 / 24

            ax.plot(t, actuals, label="Actual", lw=1.2, color="#1f77b4")
            ax.plot(t, preds, label="Predicted", lw=1.2,
                    color="#ff7f0e", alpha=0.85)

            m = _metrics(actuals, preds)
            metrics_txt = (f"MAE={m['mae']:.2f}  RMSE={m['rmse']:.2f}  "
                           f"MSE={m['mse']:.1f}  R²={m['r2']:.4f}" if m
                           else "no valid data")
            ax.set_title(f"{seq}  (test starts {start_dt})\n{metrics_txt}",
                         fontsize=10)
            ax.set_ylabel("Power (raw)")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)

        axes[0].set_xlabel("Days from test start")
        self.fig.tight_layout(rect=[0, 0, 1, 0.96])
        self.canvas.draw_idle()
        self.status_var.set("Done.")

    def run(self):
        self.root.mainloop()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    root = tk.Tk()
    EvalVisualizer(root).run()


if __name__ == "__main__":
    main()
