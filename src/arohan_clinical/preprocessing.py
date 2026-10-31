"""Feature typing, target preparation, train/test split and the sklearn preprocessor.

All imputation / scaling / encoding lives inside an sklearn ColumnTransformer that is part of the
model Pipeline, so it is fitted on training data only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config
from .data_loader import Roles


def infer_feature_types(df: pd.DataFrame, feature_cols: list[str], max_categories: int = 6):
    """Split features into (numeric, categorical). Dictionary kind wins; otherwise use dtype heuristics."""
    numeric, categorical = [], []
    for c in feature_cols:
        spec = config.spec_for(c)
        is_num = pd.api.types.is_numeric_dtype(df[c])
        kind = spec["kind"] if spec else None
        if kind in ("categorical", "binary"):
            categorical.append(c)
        elif kind == "numeric" and is_num:
            numeric.append(c)
        elif not is_num:
            categorical.append(c)
        elif pd.api.types.is_integer_dtype(df[c]) and df[c].nunique(dropna=True) <= max_categories:
            categorical.append(c)
        else:
            numeric.append(c)
    return numeric, categorical


def coerce_features(X: pd.DataFrame, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    """Numeric columns -> float; categorical columns -> string labels (so '1' and 1.0 agree)."""
    X = X.copy()
    for c in numeric:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
    for c in categorical:
        col = pd.to_numeric(X[c], errors="coerce")
        if col.notna().all() or (col.notna() == X[c].notna()).all():
            X[c] = col.map(lambda v: None if pd.isna(v) else (str(int(v)) if float(v).is_integer() else str(v)))
        else:
            X[c] = X[c].astype("object").where(X[c].notna(), None).map(lambda v: None if v is None else str(v))
    return X


class KMeansDistanceFeatures(BaseEstimator, TransformerMixin):
    """Unsupervised step of the hybrid model: K-Means distances to each cluster centre.

    The target is never used. It is fitted inside the Pipeline, so cluster centres (and k, when chosen by
    silhouette) come from the TRAINING rows only; also inside every cross-validation fold.
    """

    def __init__(self, n_clusters=None, k_range=config.HYBRID_K_RANGE, random_state=config.RANDOM_STATE):
        self.n_clusters = n_clusters
        self.k_range = k_range
        self.random_state = random_state

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if self.n_clusters:
            ks = [int(self.n_clusters)]
        else:
            ks = [k for k in range(self.k_range[0], self.k_range[1] + 1) if k < len(X)]
        if not ks:
            raise ValueError("Not enough rows for K-Means.")
        best, best_score = None, -np.inf
        for k in ks:
            km = KMeans(n_clusters=k, n_init=10, random_state=self.random_state).fit(X)
            score = silhouette_score(X, km.labels_) if len(ks) > 1 else 0.0
            if score > best_score:
                best, best_score = km, score
        self.kmeans_ = best
        self.n_clusters_ = best.n_clusters
        return self

    def transform(self, X):
        return self.kmeans_.transform(np.asarray(X, dtype=float))

    def get_feature_names_out(self, input_features=None):
        return np.array([f"distance_to_cluster_{i + 1}" for i in range(self.n_clusters_)])


def build_preprocessor(numeric: list[str], categorical: list[str], add_clusters: bool = False,
                       n_clusters: int | None = None) -> ColumnTransformer:
    parts = []
    if numeric:
        parts.append(("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                                       ("scale", StandardScaler())]), numeric))
    if categorical:
        parts.append(("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                       ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical))
    if add_clusters:
        if len(numeric) < 2:
            raise ValueError("Cluster features need at least two continuous features.")
        parts.append(("cluster", Pipeline([("impute", SimpleImputer(strategy="median")),
                                           ("scale", StandardScaler()),
                                           ("kmeans", KMeansDistanceFeatures(n_clusters=n_clusters))]), numeric))
    if not parts:
        raise ValueError("No feature columns available for preprocessing.")
    return ColumnTransformer(parts, remainder="drop")


def prepare_target(series: pd.Series):
    """Return (y in {0,1}, kept_mask, classes). Rows with missing target are dropped."""
    kept = series.notna()
    values = series[kept]
    classes = sorted(values.unique().tolist())
    if len(classes) != 2:
        raise ValueError(f"Target must be binary; found {len(classes)} distinct values: {classes[:10]}")
    y = (values == classes[1]).astype(int)
    return y, kept, classes


def split_data(df: pd.DataFrame, roles: Roles, test_size: float = config.TEST_SIZE, seed: int = config.RANDOM_STATE):
    """Stratified (when possible) train/test split. Identifier and target never enter X."""
    if roles.target_col is None:
        raise ValueError("No target column detected. Set AROHAN_TARGET to the label column name.")
    y, kept, classes = prepare_target(df[roles.target_col])
    X = df.loc[kept, roles.feature_cols]
    numeric, categorical = infer_feature_types(X, roles.feature_cols)
    X = coerce_features(X, numeric, categorical)
    stratify = y if y.value_counts().min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=stratify)
    return X_tr, X_te, y_tr, y_te, numeric, categorical, classes
