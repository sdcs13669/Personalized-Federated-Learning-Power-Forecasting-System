"""切分并打包演示数据集（steel 整份 + tetouan 按 zone 三份），输出 4 个 zip。

用法: D:\\anoconda\\envs\\fl\\python.exe data/app_datasets/prepare_app_datasets.py
"""
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "data" / "app_datasets"
OUT.mkdir(parents=True, exist_ok=True)

# (zip 名, 源 csv, 序列列)；seq_col=None 表示整份原样打包
SPECS = [
    ("steel_ind_0.zip", PROCESSED / "steel_ind.csv", None),
    ("tetouan_0.zip", PROCESSED / "tetouan_city.csv", "load_zone1"),
    ("tetouan_1.zip", PROCESSED / "tetouan_city.csv", "load_zone2"),
    ("tetouan_2.zip", PROCESSED / "tetouan_city.csv", "load_zone3"),
]

ZONE_LOADS = ("load_zone1", "load_zone2", "load_zone3")


def main() -> None:
    for name, src, seq_col in SPECS:
        df = pd.read_csv(src, parse_dates=["datetime"])
        if seq_col is not None:
            # 只保留该 zone 的负荷列，其余列（datetime/特征/category_id）原样
            df = df[[c for c in df.columns
                     if c == seq_col or c not in ZONE_LOADS]]
        csv_name = name.replace(".zip", ".csv")
        tmp_csv = OUT / csv_name
        df.to_csv(tmp_csv, index=False)
        with zipfile.ZipFile(OUT / name, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(tmp_csv, arcname=csv_name)
        tmp_csv.unlink()
        print(f"OK {name}: {len(df)} rows -> {csv_name} ({len(df.columns)} cols)")


if __name__ == "__main__":
    main()
