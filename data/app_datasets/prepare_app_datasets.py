"""切分并打包 9 个客户端的演示数据集，输出 9 个 zip。

- steel_ind 整份（steel_ind_0）
- tetouan_city 按 zone 三份（tetouan_city_0/1/2）
- lcl_res 按社区组两份（comm_000~005 / comm_100~104，30 户相邻求和已在 processed 阶段完成）
- eld_ind 按 MT 组三份（30/30/35 条 MT，入网期外的数据保留为 NaN，与实验线
  load_client_data 的"全时间范围、不截断"行为一致）

序列列清单以 fl_code/models/client_config.yaml 为唯一真源；zip 内只保留
本客户端的序列列，其余列（datetime/公共时间特征/category_id/本地特征）原样。

用法: D:\\anoconda\\envs\\fl\\python.exe data/app_datasets/prepare_app_datasets.py
"""
import zipfile
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
CONFIG = ROOT / "fl_code" / "models" / "client_config.yaml"
OUT = ROOT / "data" / "app_datasets"
OUT.mkdir(parents=True, exist_ok=True)

ZONE_LOADS = ("load_zone1", "load_zone2", "load_zone3")

# 各数据集"序列列"的识别前缀——zip 内只保留本客户端的序列列，
# 同前缀的其余列（其他客户端及未配置的 MT）全部剔除
SEQ_PREFIX = {"eld_ind": "MT_", "lcl_res": "comm_", "tetouan_city": "load_zone"}

# (zip 名, 源 csv, 序列列列表)；seq_cols=None 表示整份原样打包
SPECS = [
    ("steel_ind_0.zip", PROCESSED / "steel_ind.csv", None),
    ("tetouan_0.zip", PROCESSED / "tetouan_city.csv", ["load_zone1"]),
    ("tetouan_1.zip", PROCESSED / "tetouan_city.csv", ["load_zone2"]),
    ("tetouan_2.zip", PROCESSED / "tetouan_city.csv", ["load_zone3"]),
]

# 走 client_config.yaml 的客户端：(client_id, zip 名, 源 csv)
CONFIG_CLIENTS = [
    ("lcl_res_0", "lcl_res_0.zip", PROCESSED / "lcl_res.csv"),
    ("lcl_res_1", "lcl_res_1.zip", PROCESSED / "lcl_res.csv"),
    ("eld_ind_0", "eld_ind_0.zip", PROCESSED / "eld_ind.csv"),
    ("eld_ind_1", "eld_ind_1.zip", PROCESSED / "eld_ind.csv"),
    ("eld_ind_2", "eld_ind_2.zip", PROCESSED / "eld_ind.csv"),
]


def _config_specs() -> list[tuple[str, Path, list[str], str]]:
    """从 client_config.yaml 生成剩余客户端的切分规格。

    返回 (zip 名, 源 csv, 本客户端序列列, 数据集 id)——
    同前缀的其他序列列从 zip 中剔除，其余列原样保留。
    """
    with open(CONFIG, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    specs = []
    for cid, zname, src in CONFIG_CLIENTS:
        dataset_id = cid.rsplit("_", 1)[0]
        seqs = list(config[dataset_id]["clients"][cid]["sequences"])
        specs.append((zname, src, seqs, dataset_id))
    return specs


def main() -> None:
    for name, src, seq_cols in SPECS:
        df = pd.read_csv(src, parse_dates=["datetime"])
        if seq_cols is not None:
            # 只保留该 zone 的负荷列，其余列（datetime/特征/category_id）原样
            df = df[[c for c in df.columns
                     if c in seq_cols or c not in ZONE_LOADS]]
        _pack(name, df)

    for name, src, seqs, dataset_id in _config_specs():
        prefix = SEQ_PREFIX[dataset_id]
        df = pd.read_csv(src, parse_dates=["datetime"])
        df = df[[c for c in df.columns
                 if c in seqs or not c.startswith(prefix)]]
        _pack(name, df)


def _pack(name: str, df: pd.DataFrame) -> None:
    csv_name = name.replace(".zip", ".csv")
    tmp_csv = OUT / csv_name
    df.to_csv(tmp_csv, index=False)
    with zipfile.ZipFile(OUT / name, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(tmp_csv, arcname=csv_name)
    size_mb = (OUT / name).stat().st_size / 1e6
    tmp_csv.unlink()
    print(f"OK {name}: {len(df)} rows, {len(df.columns)} cols -> {csv_name} "
          f"({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
