"""Laboratory reports module (NidaanKosha 100k Indian lab reports dataset).

Provides:
- Data-driven discovery of lab tests, units, and LOINC codes.
- Cardiometabolic biomarker extraction (lipids, glucose, renal markers).
- Safe unit conversion (umol/L to mg/dL for creatinine) and physiological anomaly screening.
- Wide-format per-report feature matrix construction with explicit missingness handling.
- Unsupervised learning: median imputation + StandardScaler, silhouette k-selection, K-Means clustering and PCA.
- Strict diagnostic honesty: no circular threshold-based labels; unsupervised/descriptive analysis only.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .config import Paths

# Standard cardiometabolic biomarkers with physiological plausible ranges (mg/dL unless specified)
CARDIOMETABOLIC_SPECS: dict[str, dict[str, Any]] = {
    "total_cholesterol": {
        "label": "Total Cholesterol",
        "unit": "mg/dL",
        "plausible": (50.0, 600.0),
        "aliases": [
            "total cholesterol", "cholesterol total", "cholesterol - total", "serum cholesterol",
            "cholesterol", "cholesterol, total"
        ],
    },
    "triglycerides": {
        "label": "Triglycerides",
        "unit": "mg/dL",
        "plausible": (20.0, 1500.0),
        "aliases": [
            "triglycerides", "triglycerides serum", "triglyceride", "serum triglycerides"
        ],
    },
    "hdl_cholesterol": {
        "label": "HDL Cholesterol",
        "unit": "mg/dL",
        "plausible": (10.0, 150.0),
        "aliases": [
            "hdl cholesterol", "hdl cholesterol direct", "hdl-cholesterol", "cholesterol hdl",
            "high density lipoprotein"
        ],
    },
    "ldl_cholesterol": {
        "label": "LDL Cholesterol",
        "unit": "mg/dL",
        "plausible": (20.0, 400.0),
        "aliases": [
            "ldl cholesterol", "ldl cholesterol direct", "ldl-cholesterol", "cholesterol ldl",
            "low density lipoprotein"
        ],
    },
    "vldl_cholesterol": {
        "label": "VLDL Cholesterol",
        "unit": "mg/dL",
        "plausible": (2.0, 200.0),
        "aliases": [
            "vldl cholesterol", "vldl", "vldl - cholesterol", "very low density lipoprotein"
        ],
    },
    "glucose": {
        "label": "Blood Glucose",
        "unit": "mg/dL",
        "plausible": (30.0, 600.0),
        "aliases": [
            "glucose", "glucose - fasting", "glucose fasting", "fasting blood sugar",
            "fbs", "blood sugar fasting", "glucose, plasma"
        ],
    },
    "creatinine": {
        "label": "Serum Creatinine",
        "unit": "mg/dL",
        "plausible": (0.1, 25.0),
        "aliases": [
            "creatinine", "creatinine serum", "creatinine - serum", "serum creatinine"
        ],
    },
    "urea": {
        "label": "Blood Urea",
        "unit": "mg/dL",
        "plausible": (5.0, 250.0),
        "aliases": [
            "urea", "blood urea", "blood urea nitrogen", "bun", "urea serum"
        ],
    },
    "uric_acid": {
        "label": "Uric Acid",
        "unit": "mg/dL",
        "plausible": (1.0, 25.0),
        "aliases": [
            "uric acid", "uric acid serum", "serum uric acid"
        ],
    },
}


def get_cardiometabolic_mapping() -> dict[str, str]:
    """Map lowercase raw test names to standardized canonical biomarker keys."""
    mapping = {}
    for canon_key, spec in CARDIOMETABOLIC_SPECS.items():
        for alias in spec["aliases"]:
            mapping[alias.lower().strip()] = canon_key
    return mapping


def find_nidaankosha_files(paths: Paths) -> list[Path]:
    """Locate NidaanKosha parquet files in data/raw/labs/."""
    candidates = []
    data_dir = paths.raw_labs / "data"
    if data_dir.exists():
        candidates.extend(sorted(data_dir.glob("*.parquet")))
    if not candidates and paths.raw_labs.exists():
        candidates.extend(sorted(paths.raw_labs.glob("*.parquet")))
    return candidates


def get_active_labs_source(paths: Paths) -> tuple[Path | None, bool]:
    """Return the active labs parquet file and whether it is the derived demo sample.

    Returns (path, is_demo). Uses full local files if present; otherwise falls back
    to data/demo/labs_cardiometabolic_sample.parquet.
    """
    full_files = find_nidaankosha_files(paths)
    if full_files:
        return full_files[0], False
    if hasattr(paths, "demo_labs") and paths.demo_labs.exists():
        return paths.demo_labs, True
    return None, False


def discover_lab_tests(parquet_path: Path, max_rows: int = 50000) -> pd.DataFrame:
    """Scan real parquet file and return data-driven summary of test names, counts, units, and LOINCs."""
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path, columns=["test_name", "unit", "loinc", "value"])
    if table.num_rows > max_rows:
        table = table.slice(0, max_rows)
    df = table.to_pandas()

    records = []
    for test_name, group in df.groupby("test_name"):
        units = [u for u in group["unit"].dropna().unique() if str(u).strip() != ""][:3]
        loincs = [l for l in group["loinc"].dropna().unique() if str(l).strip() != ""][:2]
        records.append({
            "test_name": test_name,
            "readings_count": len(group),
            "common_units": ", ".join(units) if units else "-",
            "loinc_codes": ", ".join(loincs) if loincs else "-",
        })

    summary = pd.DataFrame(records).sort_values("readings_count", ascending=False).reset_index(drop=True)
    return summary


def clean_and_normalize_lab_readings(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clean lab readings, safely convert units, filter physiological anomalies, and log stats."""
    mapping = get_cardiometabolic_mapping()
    cleaned = df.copy()

    # Map raw test name to canonical biomarker
    cleaned["test_clean"] = cleaned["test_name"].astype(str).str.lower().str.strip()
    cleaned["canon_biomarker"] = cleaned["test_clean"].map(mapping)

    # Filter to cardiometabolic tests
    cardio_df = cleaned[cleaned["canon_biomarker"].notna()].copy()

    # Extract float value
    cardio_df["numeric_value"] = pd.to_numeric(cardio_df["value"], errors="coerce")

    stats = {
        "total_cardio_rows": len(cardio_df),
        "non_numeric_values": int(cardio_df["numeric_value"].isna().sum()),
        "unit_conversions_applied": 0,
        "anomalies_filtered": 0,
    }

    # Safe unit conversion: creatinine from umol/L to mg/dL (1 mg/dL = 88.4 umol/L)
    is_creat = cardio_df["canon_biomarker"] == "creatinine"
    unit_str = cardio_df["unit"].astype(str).str.lower().str.strip()
    is_umol = is_creat & unit_str.str.contains("umol", na=False)
    if is_umol.any():
        num_conv = int(is_umol.sum())
        cardio_df.loc[is_umol, "numeric_value"] = cardio_df.loc[is_umol, "numeric_value"] / 88.4
        stats["unit_conversions_applied"] += num_conv

    # Physiological plausible range check
    valid_mask = cardio_df["numeric_value"].notna()
    for canon_key, spec in CARDIOMETABOLIC_SPECS.items():
        low, high = spec["plausible"]
        key_mask = (cardio_df["canon_biomarker"] == canon_key) & valid_mask
        out_of_range = key_mask & ((cardio_df["numeric_value"] < low) | (cardio_df["numeric_value"] > high))
        if out_of_range.any():
            count_anom = int(out_of_range.sum())
            stats["anomalies_filtered"] += count_anom
            cardio_df.loc[out_of_range, "numeric_value"] = np.nan

    return cardio_df, stats


def build_lab_feature_matrix(
    cleaned_df: pd.DataFrame,
    min_tests_per_doc: int = 1,
    max_docs: int | None = None,
) -> pd.DataFrame:
    """Pivot long-format lab readings into a wide per-document feature matrix."""
    valid = cleaned_df[cleaned_df["numeric_value"].notna()].copy()
    if valid.empty:
        return pd.DataFrame()

    # Keep demographic metadata if present
    meta_cols = [c for c in ["age", "gender"] if c in valid.columns]
    group_cols = ["document_id"] + meta_cols

    # Count tests per document
    doc_test_counts = valid.groupby("document_id")["canon_biomarker"].nunique()
    eligible_docs = doc_test_counts[doc_test_counts >= min_tests_per_doc].index
    if max_docs and len(eligible_docs) > max_docs:
        eligible_docs = eligible_docs[:max_docs]

    filtered = valid[valid["document_id"].isin(eligible_docs)]

    # Pivot: mean value per document for each biomarker
    matrix = filtered.pivot_table(
        index=group_cols,
        columns="canon_biomarker",
        values="numeric_value",
        aggfunc="mean",
    ).reset_index()

    return matrix


def run_lab_unsupervised(
    matrix_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    n_clusters: int | None = None,
    k_range: tuple[int, int] = (2, 6),
    random_state: int = 42,
) -> dict[str, Any]:
    """Perform unsupervised clustering and PCA on the lab feature matrix.

    Preprocessing (median imputation + StandardScaler) is fitted inside this analysis.
    """
    if matrix_df.empty:
        raise ValueError("Cannot perform unsupervised analysis on empty matrix.")

    if feature_cols is None:
        biomarker_keys = list(CARDIOMETABOLIC_SPECS.keys())
        feature_cols = [c for c in matrix_df.columns if c in biomarker_keys]

    if not feature_cols:
        raise ValueError("No cardiometabolic feature columns found in matrix.")

    X_raw = matrix_df[feature_cols].copy()

    # Fit imputer and scaler internally
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_raw)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    # Determine k: silhouette search or override
    silhouette_scores: dict[int, float] = {}
    chosen_k = n_clusters
    if chosen_k is None:
        best_k = k_range[0]
        best_sil = -1.0
        # If dataset is large, sample up to 2500 rows for silhouette scoring to maintain fast responsiveness
        n_rows = len(X_scaled)
        if n_rows > 2500:
            rng = np.random.default_rng(random_state)
            sample_idx = rng.choice(n_rows, size=2500, replace=False)
            X_sil = X_scaled[sample_idx]
        else:
            X_sil = X_scaled

        for k in range(k_range[0], k_range[1] + 1):
            km = KMeans(n_clusters=k, random_state=random_state, n_init=5)
            labels = km.fit_predict(X_scaled)
            if n_rows > 2500:
                sil = float(silhouette_score(X_sil, labels[sample_idx]))
            else:
                sil = float(silhouette_score(X_scaled, labels))
            silhouette_scores[k] = round(sil, 4)
            if sil > best_sil:
                best_sil = sil
                best_k = k
        chosen_k = best_k

    # Fit final KMeans
    kmeans = KMeans(n_clusters=chosen_k, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # Fit PCA (2D for scatter plot, plus all components for variance)
    n_pca_components = min(X_scaled.shape[1], 2)
    pca = PCA(n_components=n_pca_components, random_state=random_state)
    coords = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame({
        "PC1": coords[:, 0],
        "PC2": coords[:, 1] if n_pca_components > 1 else np.zeros(len(coords)),
        "cluster": [f"Cluster {c + 1}" for c in cluster_labels],
    })
    if "document_id" in matrix_df.columns:
        pca_df["document_id"] = matrix_df["document_id"].values

    # Cluster summaries (biomarker medians/means per cluster)
    res_df = matrix_df.copy()
    res_df["cluster"] = [f"Cluster {c + 1}" for c in cluster_labels]
    cluster_means = res_df.groupby("cluster")[feature_cols].mean().round(2)
    cluster_counts = res_df.groupby("cluster")["cluster"].count().rename("Lab Reports")
    cluster_summaries = pd.concat([cluster_counts, cluster_means], axis=1).reset_index()

    # Explained variance for full PCA
    full_pca = PCA(random_state=random_state)
    full_pca.fit(X_scaled)

    return {
        "k": chosen_k,
        "silhouette_scores": silhouette_scores,
        "cluster_labels": cluster_labels,
        "cluster_summaries": cluster_summaries,
        "pca_df": pca_df,
        "explained_variance_ratio": [float(r) for r in full_pca.explained_variance_ratio_],
        "feature_cols": feature_cols,
        "models": {
            "imputer": imputer,
            "scaler": scaler,
            "kmeans": kmeans,
            "pca": pca,
        },
    }
