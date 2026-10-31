"""Exploratory unsupervised analysis: scaling + K-Means + PCA on continuous clinical measurements.

Run from the command line with:  python -m arohan_clinical.unsupervised   (from the `src` folder)
The analysis is exploratory and uses all rows (there is no held-out set for unsupervised structure).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from . import config
from .data_loader import DatasetError, Roles, detect_roles, find_dataset_file, load_dataset
from .preprocessing import infer_feature_types


def select_features(df: pd.DataFrame, roles: Roles) -> list[str]:
    """Continuous measurements only: Euclidean distance on category codes (sex, chest-pain type...) is not meaningful."""
    numeric, _ = infer_feature_types(df, roles.feature_cols)
    return numeric


def run_analysis(df: pd.DataFrame, roles: Roles, k: int | None = None, k_range=range(2, 9),
                 seed: int = config.RANDOM_STATE) -> dict:
    features = select_features(df, roles)
    if len(features) < 2:
        raise ValueError("Need at least two continuous features for K-Means/PCA.")
    X = df[features].apply(pd.to_numeric, errors="coerce")
    Xs = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(X))

    rows = []
    for kk in k_range:
        if kk >= len(df):
            continue
        km = KMeans(n_clusters=kk, n_init=10, random_state=seed).fit(Xs)
        rows.append({"k": kk, "silhouette": float(silhouette_score(Xs, km.labels_)), "inertia": float(km.inertia_)})
    scores = pd.DataFrame(rows)
    if scores.empty:
        raise ValueError("Not enough rows to evaluate K-Means.")
    best_k = int(scores.loc[scores["silhouette"].idxmax(), "k"])
    k = int(k) if k else best_k
    labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Xs)

    pca = PCA(n_components=min(len(features), 5), random_state=seed).fit(Xs)
    coords = pca.transform(Xs)[:, :2]
    explained = pd.DataFrame({"component": [f"PC{i + 1}" for i in range(len(pca.explained_variance_ratio_))],
                              "explained_variance": pca.explained_variance_ratio_})
    explained["cumulative"] = explained["explained_variance"].cumsum()

    cluster = pd.Series(labels, name="cluster").map(lambda v: f"Cluster {v + 1}")
    pca_df = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1], "cluster": cluster.values})
    sizes = cluster.value_counts().sort_index().rename_axis("cluster").reset_index(name="rows")
    sizes["percent"] = (sizes["rows"] / len(df) * 100).round(1)
    summary = X.assign(cluster=cluster.values).groupby("cluster").agg(["mean"]).droplevel(1, axis=1).round(2)

    crosstab = None
    if roles.target_col is not None:
        crosstab = pd.crosstab(cluster.values, df[roles.target_col].astype(str).values)
        crosstab.index.name = "cluster"
        crosstab.columns.name = f"{roles.target_col} (external label, not used for clustering)"
        pca_df["target"] = df[roles.target_col].astype(str).values

    return {"features": features, "k": k, "best_k": best_k, "scores": scores, "sizes": sizes,
            "summary": summary.reset_index(), "pca": pca_df, "explained": explained, "crosstab": crosstab,
            "settings": {"scaler": "StandardScaler (mean 0, sd 1)", "imputer": "median", "kmeans_n_init": 10,
                         "seed": seed, "k_selection": "highest silhouette over k=2..8 unless chosen by the user"}}


def save_outputs(paths: config.Paths, result: dict) -> list[str]:
    paths.ensure()
    written = []
    for key in ("scores", "sizes", "summary", "explained"):
        p = paths.tables / f"unsupervised_{key}.csv"
        result[key].to_csv(p, index=False)
        written.append(p.name)
    return written


if __name__ == "__main__":
    paths = config.default_paths()
    csv = find_dataset_file(paths.raw)
    if csv is None:
        raise SystemExit(f"No CSV found in {paths.raw}")
    frame = load_dataset(csv)
    out = run_analysis(frame, detect_roles(frame))
    print("features:", out["features"], "k:", out["k"])
    print(out["sizes"])
    print("saved:", save_outputs(paths, out))
