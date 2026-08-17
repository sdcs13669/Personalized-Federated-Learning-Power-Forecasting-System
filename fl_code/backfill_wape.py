"""Backfill WAPE into an existing personalized_results.json without retraining.

Re-loads the frozen Global TCN and each client's Corrector checkpoint
recorded in the JSON, re-runs the exact rolling-forecast evaluation of
train_personalized, and writes per-client + aggregate WAPE back into the
JSON in place:

  - ``results[cid].wape_baseline`` / ``wape_personalized``
  - ``final_metrics.wape_baseline`` / ``wape_personalized``

The JSON is only overwritten after every client evaluates successfully.
Model / Corrector paths come from the JSON itself; when the recorded
``output_dir`` does not exist on this machine (JSON copied from the
server), Corrector checkpoints fall back to the JSON file's own directory.

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

        stored = r.get("mae_personalized")
        if stored is not None and abs(stored - m_pers["mae"]) > 1e-4:
            print(f"  WARNING {cid}: recomputed mae_personalized "
                  f"{m_pers['mae']:.4f} != stored {stored:.4f}")

        r["wape_baseline"] = m_base["wape"]
        r["wape_personalized"] = m_pers["wape"]
        wape_base_num += sums[0]
        wape_pers_num += sums[1]
        wape_denom += sums[2]
        print(f"  {cid:<20s} wape_base={m_base['wape']:.4f}  "
              f"wape_pers={m_pers['wape']:.4f}")

    results["final_metrics"]["wape_baseline"] = (
        float(wape_base_num / wape_denom) if wape_denom > 0 else float("nan"))
    results["final_metrics"]["wape_personalized"] = (
        float(wape_pers_num / wape_denom) if wape_denom > 0 else float("nan"))
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
        description="Backfill WAPE into personalized_results.json using the "
                    "already-trained models (no retraining)")
    parser.add_argument("--json", type=str, default=str(DEFAULT_JSON),
                        help=f"Path to personalized_results.json "
                             f"(default: {DEFAULT_JSON})")
    main(parser.parse_args())
