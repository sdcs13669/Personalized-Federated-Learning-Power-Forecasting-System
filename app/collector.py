"""数据采集：从 GitHub URL 下载 zip → 解压到本地数据目录 → 校验。"""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

REQUIRED_COLS = ["datetime"]


def collect_dataset(dataset_id: str, url: str, data_dir: Path,
                    client_id: str) -> dict:
    """下载、解压、校验。返回 info dict，失败抛异常。"""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(url, timeout=30) as resp:
        if resp.status != 200:
            raise RuntimeError(f"下载失败: HTTP {resp.status}")
        raw = resp.read()

    zip_bytes = io.BytesIO(raw)
    with zipfile.ZipFile(zip_bytes) as zf:
        names = zf.namelist()
        csv_name = next((n for n in names if n.endswith(".csv")), None)
        if csv_name is None:
            raise RuntimeError("压缩包内没有 csv 文件")
        with zf.open(csv_name) as f:
            csv_bytes = f.read()

    out_csv = data_dir / Path(csv_name).name
    out_csv.write_bytes(csv_bytes)

    df = pd.read_csv(out_csv, parse_dates=["datetime"])
    for col in REQUIRED_COLS:
        if col not in df.columns:
            raise RuntimeError(f"数据缺少列: {col}")

    usage_cols = [c for c in df.columns if c not in ("datetime",)]
    missing = df[usage_cols].isna().sum().sum() if usage_cols else 0
    total = len(df) * max(len(usage_cols), 1)
    missing_rate = round(missing / max(total, 1), 4)

    (data_dir / "dataset_id.txt").write_text(dataset_id, encoding="utf-8")

    return {
        "ok": True,
        "dataset_id": dataset_id,
        "client_id": client_id,
        "csv_path": str(out_csv),
        "rows": int(len(df)),
        "time_range": (str(df["datetime"].iloc[0]),
                       str(df["datetime"].iloc[-1])),
        "missing_rate": missing_rate,
    }
