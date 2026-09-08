"""客户端阶段3：本地测试集滚动预测 → 生成展示数据（need.md）。

加载全局 TCN 与本地残差修正器（阶段2产物），对本地测试集做滚动预测，
输出：真实 / 全局TCN预测 / 全局+RC P50 / 区间下上界 曲线，以及
WAPE(global / +RC)、WAPE 下降比例、PINAW、区间覆盖率、修正器架构。

用法: python -m fl_code.app_stage_eval \
        --global-model <g.pt> --corrector <c.pt> --rc-type tcn \
        --cid steel_ind_0 --data-dir app/data --out out.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]


def _load_data(data_dir: str, client_id: str) -> dict:
    """镜像 train_personalized._load_client_data_cached 的 App 数据分支。"""
    import yaml
    from fl_code.data_utils import preprocess, _find_client
    cfg_path = Path(__file__).resolve().parent / "models" / "client_config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    dataset_id, client_cfg = _find_client(config, client_id)
    ddir = Path(data_dir)
    csvs = sorted(ddir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"data-dir 下没有 csv: {ddir}")
    df = pd.read_csv(csvs[0], parse_dates=["datetime"])
    seqs = client_cfg["sequences"]
    public_cols = list(config[dataset_id]["public_features"])
    local_cols = list(config[dataset_id].get("local_features", []))
    keep = ["datetime"] + seqs + \
        [c for c in public_cols + local_cols if c in df.columns]
    if "category_id" in df.columns:
        keep.append("category_id")  # 保留以展开 cat_residential/...，否则 public 列 KeyError
    df = df[keep]
    if "category_id" in df.columns:
        cat = df["category_id"].astype(int)
        df["cat_residential"] = (cat == 0).astype(float)
        df["cat_transformer"] = (cat == 1).astype(float)
        df["cat_industrial"] = (cat == 2).astype(float)
        df = df.drop(columns=["category_id"])
    df_raw = df.copy()
    df_norm, params = preprocess(df, seqs, local_cols)
    return {"df_raw": df_raw, "df_norm": df_norm, "seqs": seqs,
            "public_cols": public_cols, "local_cols": local_cols,
            "params": params}


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--global-model", required=True)
    ap.add_argument("--corrector", required=True)
    ap.add_argument("--rc-type", default="tcn")
    ap.add_argument("--cid", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=48)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) 模型
    from fl_code.models import TCNConfig, CorrectorConfig, build_tcn, build_corrector
    global_tcn = build_tcn(TCNConfig()).to(device)
    global_tcn.load_state_dict(torch.load(args.global_model, map_location=device,
                                          weights_only=True))
    global_tcn.eval()
    for p in global_tcn.parameters():
        p.requires_grad = False

    # 2) 数据
    data = _load_data(args.data_dir, args.cid)
    df_norm = data["df_norm"]
    df_raw = data["df_raw"]
    params = data["params"]
    seqs = data["seqs"]
    public_cols = data["public_cols"]
    local_cols = data["local_cols"]

    # 3) 修正器
    corr_cfg = CorrectorConfig(rc_type=args.rc_type,
                               local_feat_dim=len(local_cols))
    corrector = build_corrector(corr_cfg).to(device)
    corrector.load_state_dict(torch.load(args.corrector, map_location=device,
                                         weights_only=True))
    corrector.eval()

    pub_arr = df_norm[public_cols].values.astype(np.float32)
    loc_arr = df_norm[local_cols].values.astype(np.float32) if local_cols else None
    input_steps, pred_len = 144, 6

    # 4) 滚动预测（镜像 train_personalized._evaluate_personalized）
    series_real, series_global, series_rc, series_lo, series_hi = [], [], [], [], []
    all_actual, all_base, all_rcp50 = [], [], []
    all_lo, all_hi = [], []
    best_seq = None   # 记录最后一个有效序列，用于"预测未来"

    for s in seqs:
        load = df_norm[s].values.astype(np.float32)      # 归一化负荷（模型输入）
        load_raw = df_raw[s].values.astype(np.float32)   # 原始负荷（真实功率）
        p = params.get(s) or {"log1p": True, "mean": 0.0, "std": 1.0}

        def _to_power(x):
            """逆变换 log1p→z-score：还原真实功率。"""
            x = np.asarray(x, dtype=np.float64) * p["std"] + p["mean"]
            return np.expm1(x) if p.get("log1p") else x

        f = df_norm[s].first_valid_index()
        l = df_norm[s].last_valid_index()
        if f is None or l is None:
            continue
        valid_len = l - f + 1
        split = f + int(valid_len * 0.8)

        prev_residual = np.zeros(pred_len, dtype=np.float32)
        pos = split
        while pos + input_steps + pred_len <= l + 1:
            X_pub = pub_arr[pos:pos + input_steps].T
            X_load = load[pos:pos + input_steps][np.newaxis, :]
            X = np.concatenate([X_pub, X_load], axis=0)
            X_t = torch.from_numpy(X).unsqueeze(0).to(device)
            y_pre = global_tcn(X_t).squeeze(0).cpu().numpy()   # (pred_len,) 归一化

            X_rc = X
            if loc_arr is not None:
                loc_win = np.nan_to_num(
                    loc_arr[pos:pos + input_steps], nan=0.0).T
                X_rc = np.concatenate([X_rc, loc_win], axis=0)
            X_rc_t = torch.from_numpy(X_rc).unsqueeze(0).to(device)
            residual_t = torch.from_numpy(prev_residual).unsqueeze(0).to(device)
            e_corr = corrector(torch.from_numpy(y_pre).unsqueeze(0).to(device),
                               residual_t, X_rc_t).squeeze(0).cpu().numpy()  # (T,3)
            y_final = y_pre[:, np.newaxis] + e_corr                 # (T,3) 归一化

            actual_norm = load[pos + input_steps:pos + input_steps + pred_len]
            actual = load_raw[pos + input_steps:pos + input_steps + pred_len]  # 真实功率

            all_actual.append(actual)
            all_base.append(_to_power(y_pre))
            all_rcp50.append(_to_power(y_final[:, 1]))
            all_lo.append(_to_power(y_final[:, 0]))
            all_hi.append(_to_power(y_final[:, 2]))
            prev_residual = actual_norm - y_pre
            pos += args.stride

        # 记录最后一个（最大的 last_valid_index）序列，用于“最后一个窗口预测未来”
        if best_seq is None or l > best_seq[2]:
            best_seq = (load, f, l, _to_power)

    if not all_actual:
        raise RuntimeError("测试集没有可预测的窗口")

    # ---- 最后一个窗口预测未来（没有真实值，只给预测曲线/区间）----
    future = {"global": [], "rc": [], "lower": [], "upper": []}
    if best_seq is not None:
        load_b, f_b, l_b, _to_power = best_seq
        pos_f = l_b + 1 - input_steps          # 最后 input_steps 点作为输入
        if pos_f >= f_b and pos_f + input_steps <= l_b + 1:
            X_pub = pub_arr[pos_f:pos_f + input_steps].T
            X_load = load_b[pos_f:pos_f + input_steps][np.newaxis, :]
            X = np.concatenate([X_pub, X_load], axis=0)
            X_t = torch.from_numpy(X).unsqueeze(0).to(device)
            y_pre = global_tcn(X_t).squeeze(0).cpu().numpy()
            X_rc = X
            if loc_arr is not None:
                loc_win = np.nan_to_num(
                    loc_arr[pos_f:pos_f + input_steps], nan=0.0).T
                X_rc = np.concatenate([X_rc, loc_win], axis=0)
            e_corr = corrector(
                torch.from_numpy(y_pre).unsqueeze(0).to(device),
                torch.zeros(1, pred_len).to(device),
                torch.from_numpy(X_rc).unsqueeze(0).to(device),
            ).squeeze(0).cpu().numpy()
            y_final = y_pre[:, np.newaxis] + e_corr
            future = {"global": _to_power(y_pre).tolist(),
                      "rc": _to_power(y_final[:, 1]).tolist(),
                      "lower": _to_power(y_final[:, 0]).tolist(),
                      "upper": _to_power(y_final[:, 2]).tolist()}

    actuals = np.concatenate(all_actual)
    valid = ~np.isnan(actuals)
    base_preds = np.concatenate(all_base)[valid]
    rc_preds = np.concatenate(all_rcp50)[valid]
    lo_preds = np.concatenate(all_lo)[valid]
    hi_preds = np.concatenate(all_hi)[valid]
    actuals = actuals[valid]

    def wape(p):
        denom = float(np.sum(np.abs(actuals)))
        return float(np.sum(np.abs(p - actuals)) / denom * 100) if denom > 0 else float("nan")

    wape_global = wape(base_preds)
    wape_rc = wape(rc_preds)
    wape_drop_pct = ((wape_global - wape_rc) / wape_global * 100
                     if wape_global and not np.isnan(wape_global) else None)

    # 区间指标（归一化空间）
    in_range = (actuals >= lo_preds) & (actuals <= hi_preds)
    coverage = float(np.mean(in_range)) * 100
    width = float(np.mean(hi_preds - lo_preds))
    arange = float(np.max(actuals) - np.min(actuals))
    pinaw = width / arange if arange > 0 else float("nan")

    # 5) 动态播放用：把每个窗口的预测段顺次拼接（真实/全局/RC/区间）
    #    只拼“有效（有真实）”的窗口，末尾留一个用最后窗口预测未来
    def flat(arrs, filt):
        out = []
        for a, ok in zip(arrs, filt):
            if ok:
                out.extend(a.tolist())
        return out
    ok = [True] * len(all_actual)
    sr = flat(all_actual, ok)
    sg = flat(all_base, ok)
    src = flat(all_rcp50, ok)
    slo = flat(all_lo, ok)
    shi = flat(all_hi, ok)

    # 把“预测未来”追加到播放序列尾部：真实值无（null），只给预测曲线/区间
    nf = len(future.get("global", []))
    if nf:
        sr = sr + [None] * nf
        sg = sg + future["global"]
        src = src + future["rc"]
        slo = slo + future["lower"]
        shi = shi + future["upper"]

    out = {
        "client_id": args.cid,
        "arch": args.rc_type,
        "wape_global": round(wape_global, 3),
        "wape_rc": round(wape_rc, 3),
        "wape_drop_pct": round(wape_drop_pct, 2) if wape_drop_pct is not None else None,
        "pinaw": round(pinaw, 4) if not np.isnan(pinaw) else None,
        "coverage": round(coverage, 2),
        "n_points": int(len(sr)),
        "future": future,
        "series": {"real": sr, "global": sg, "rc": src,
                   "lower": slo, "upper": shi},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print("OK stage3 ->", args.out)


if __name__ == "__main__":
    main()
