"""Client data loading shared by the experiment line and the App line."""
from __future__ import annotations

from fl_code.data_utils import (
    load_client_data, preprocess, LazySlidingWindowDataset)


def load_client_cache(client_id: str, stride: int,
                      max_seqs: int | None = None) -> dict:
    df, info = load_client_data(client_id)
    feat_names = set(info["public_features"] + info["local_features"])
    seqs = [c for c in df.columns if c not in feat_names and c != "datetime"]
    if max_seqs and len(seqs) > max_seqs:
        seqs = seqs[:max_seqs]
    df_norm, _ = preprocess(df, seqs, info["local_features"])
    train_ds = LazySlidingWindowDataset(
        df_norm, seqs, info["public_features"], stride=stride, train=True)
    return {
        "df_norm": df_norm,
        "seqs": seqs,
        "public_cols": info["public_features"],
        "local_cols": info["local_features"],
        "train_ds": train_ds,
        "n_train": len(train_ds),
    }
