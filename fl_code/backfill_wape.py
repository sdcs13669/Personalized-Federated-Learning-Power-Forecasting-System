"""Re-evaluate the models recorded in personalized_results.json and refresh
all evaluation metrics in place — no retraining.

Re-loads the frozen Global TCN and each client's Corrector checkpoint
recorded in the JSON, re-runs the exact rolling-forecast evaluation of
train_personalized, and overwrites per-client + aggregate metrics so the
whole file comes from a single evaluation pass:

  - ``results[cid]``: mae / rmse / r2 / wape (baseline + personalized),
    improvement_mae_pct
  - ``final_metrics``: avg_mae_baseline / avg_mae_personalized /
    avg_improvement_mae_pct / wape_baseline / wape_personalized /
    client_metrics

Use it to add WAPE to older Phase 3 outputs, or to refresh stale eval
numbers against the current local data. The JSON is only overwritten after
every client evaluates successfully. Model / Corrector paths come from the
JSON itself; when the recorded ``output_dir`` does not exist on this
machine (JSON copied from the server), Corrector checkpoints fall back to
the JSON file's own directory.

Usage::

    python -m fl_code.backfill_wape
    python -m fl_code.backfill_wape --json path/to/personalized_results.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from fl_code.config import STRIDE
from fl_code.models import (
    TCNConfig, CorrectorConfig, build_tcn, build_corrector,
)
from fl_code.train_personalized import (
    _evaluate_personalized, _load_client_data_cached,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = ROOT / "fl_code" / "personalized_outputs" / "personalized_results.json"


def _parse_int(value: str | None) -> int | None:
    if value is None or str(value).strip().lower() in ("none", ""):
        return None
    return int(value)


def backfill(results: dict, json_path: Path, device: str) -> dict:
    args = results["args"]
    stride = _parse_int(args.get("stride")) or STRIDE
    max_seqs = _parse_int(args.get("max_seqs"))
    eval_seqs = _parse_int(args.get("eval_seqs"))
    output_dir = Path(args["output_dir"])

    global_path = Path(results["global_model"])
    if not global_path.exists():
        global_path = ROOT / global_path
    global_model = build_tcn(TCNConfig()).to(device)
    global_model.load_state_dict(
        torch.load(global_path, map_location=device, weights_only=True))
    global_model.eval()

    wape_base_num = wape_pers_num = wape_denom = 0.0
    for cid, r in results["results"].items():
        data = _load_client_data_cached(cid, stride, max_seqs)
        seqs = data["seqs"]
        if eval_seqs and len(seqs) > eval_seqs:
            seqs = seqs[:eval_seqs]

        corrector = build_corrector(CorrectorConfig(
            rc_type=r["corrector_type"],
            local_feat_dim=r["local_dim"])).to(device)
        ckpt = output_dir / f"corrector_{cid}.pt"
        if not ckpt.exists():
            ckpt = json_path.parent / f"corrector_{cid}.pt"
        corrector.load_state_dict(
            torch.load(ckpt, map_location=device, weights_only=True))
        corrector.eval()

        m_base, m_pers, sums = _evaluate_personalized(
            global_model, corrector, data["df_norm"], seqs,
            data["public_cols"], data["local_cols"], stride, device)

        # 同一评估的 mae/rmse/r2/wape 一起覆盖，保证 JSON 内指标自洽
        r["mae_baseline"] = m_base["mae"]
        r["rmse_baseline"] = m_base["rmse"]
        r["r2_baseline"] = m_base["r2"]
        r["mae_personalized"] = m_pers["mae"]
        r["rmse_personalized"] = m_pers["rmse"]
        r["r2_personalized"] = m_pers["r2"]
        gain = None
        if not np.isnan(m_base["mae"]) and not np.isnan(m_pers["mae"]):
            gain = (m_base["mae"] - m_pers["mae"]) / m_base["mae"] * 100
        r["improvement_mae_pct"] = round(gain, 2) if gain is not None else None
        r["wape_baseline"] = m_base["wape"]
        r["wape_personalized"] = m_pers["wape"]
        wape_base_num += sums[0]
        wape_pers_num += sums[1]
        wape_denom += sums[2]
        print(f"  {cid:<20s} mae_pers={m_pers['mae']:.4f}  "
              f"wape_base={m_base['wape']:.4f}  wape_pers={m_pers['wape']:.4f}")

    # 重建聚合指标（与 train_personalized 相同的口径）
    valid_base = [r["mae_baseline"] for r in results["results"].values()
                  if not np.isnan(r["mae_baseline"])]
    valid_pers = [r["mae_personalized"] for r in results["results"].values()
                  if not np.isnan(r["mae_personalized"])]
    gains = [r["improvement_mae_pct"] for r in results["results"].values()
             if r["improvement_mae_pct"] is not None]
    client_metrics = {
        cid: {"mae": r["mae_personalized"],
              "rmse": r["rmse_personalized"],
              "r2": r["r2_personalized"],
              "n_train": r["n_windows"]}
        for cid, r in results["results"].items()
    }
    results["final_metrics"] = {
        "avg_mae_baseline": (float(np.mean(valid_base))
                             if valid_base else float("nan")),
        "avg_mae_personalized": (float(np.mean(valid_pers))
                                 if valid_pers else float("nan")),
        "avg_improvement_mae_pct": (round(float(np.mean(gains)), 2)
                                    if gains else None),
        "wape_baseline": (float(wape_base_num / wape_denom)
                          if wape_denom > 0 else float("nan")),
        "wape_personalized": (float(wape_pers_num / wape_denom)
                              if wape_denom > 0 else float("nan")),
        "client_metrics": client_metrics,
    }
    return results


def main(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    json_path = Path(args.json)
    with open(json_path) as f:
        results = json.load(f)
    print(f"Backfilling WAPE for {len(results['results'])} clients "
          f"(global model: {results['global_model']}) ...")

    results = backfill(results, json_path, device)

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    fm = results["final_metrics"]
    print(f"\nWrote {json_path}")
    print(f"  final_metrics.wape_baseline     = {fm['wape_baseline']:.4f}")
    print(f"  final_metrics.wape_personalized = {fm['wape_personalized']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Re-evaluate trained models and refresh all eval metrics "
                    "in personalized_results.json (no retraining)")
    parser.add_argument("--json", type=str, default=str(DEFAULT_JSON),
                        help=f"Path to personalized_results.json "
                             f"(default: {DEFAULT_JSON})")
    main(parser.parse_args())
