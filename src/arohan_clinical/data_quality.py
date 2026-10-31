"""Data-quality checks. These only *report*; they never change the raw data."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .data_loader import Roles
from .preprocessing import infer_feature_types


def dtype_table(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"column": df.columns, "dtype": [str(t) for t in df.dtypes],
                         "unique_values": [int(df[c].nunique(dropna=True)) for c in df.columns]})


def missing_values(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().sum()
    return pd.DataFrame({"column": miss.index, "missing": miss.values.astype(int),
                         "percent": (miss.values / max(len(df), 1) * 100).round(2)})


def duplicate_checks(df: pd.DataFrame, id_col: str | None) -> dict:
    feats = df.drop(columns=[id_col]) if id_col else df
    out = {"full_row_duplicates": int(df.duplicated().sum()),
           "duplicates_excluding_id": int(feats.duplicated().sum()),
           "id_column": id_col, "duplicate_ids": None}
    if id_col:
        out["duplicate_ids"] = int(df[id_col].duplicated().sum())
    return out


def unexpected_categories(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        spec = config.spec_for(c)
        if not spec or not spec.get("expected"):
            continue
        num = pd.to_numeric(df[c], errors="coerce")
        bad = df[c].notna() & ~num.isin(spec["expected"])
        if bad.any():
            rows.append({"column": c, "expected_values": ", ".join(map(str, spec["expected"])),
                         "unexpected_values": ", ".join(sorted(map(str, df.loc[bad, c].unique())))[:200],
                         "rows_affected": int(bad.sum())})
    return pd.DataFrame(rows, columns=["column", "expected_values", "unexpected_values", "rows_affected"])


def range_checks(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        spec = config.spec_for(c)
        if not spec or not spec.get("plausible"):
            continue
        num = pd.to_numeric(df[c], errors="coerce")
        lo, hi = spec["plausible"]
        rows.append({"column": c, "observed_min": num.min(), "observed_max": num.max(),
                     "plausible_range": f"{lo} to {hi}", "rows_outside": int(((num < lo) | (num > hi)).sum())})
    return pd.DataFrame(rows, columns=["column", "observed_min", "observed_max", "plausible_range", "rows_outside"])


def zero_value_report(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """Count zeros in continuous columns. Zeros are NOT assumed to be missing."""
    rows = []
    for c in numeric_cols:
        num = pd.to_numeric(df[c], errors="coerce")
        n = int((num == 0).sum())
        rows.append({"column": c, "zero_values": n, "percent": round(n / max(len(df), 1) * 100, 2)})
    return pd.DataFrame(rows, columns=["column", "zero_values", "percent"])


def outlier_summary(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in numeric_cols:
        num = pd.to_numeric(df[c], errors="coerce").dropna()
        if num.empty:
            continue
        q1, q3 = num.quantile(0.25), num.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        rows.append({"column": c, "q1": q1, "q3": q3, "lower_fence": lo, "upper_fence": hi,
                     "outliers": int(((num < lo) | (num > hi)).sum())})
    return pd.DataFrame(rows, columns=["column", "q1", "q3", "lower_fence", "upper_fence", "outliers"])


def target_distribution(df: pd.DataFrame, target_col: str | None) -> pd.DataFrame:
    if not target_col:
        return pd.DataFrame(columns=["class", "count", "percent"])
    vc = df[target_col].value_counts(dropna=False).sort_index()
    return pd.DataFrame({"class": vc.index.astype(str), "count": vc.values.astype(int),
                         "percent": (vc.values / len(df) * 100).round(2)})


def correlation_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    num = df[cols].apply(pd.to_numeric, errors="coerce")
    return num.corr()


def build_quality_report(df: pd.DataFrame, roles: Roles) -> dict:
    numeric, categorical = infer_feature_types(df, roles.feature_cols)
    miss = missing_values(df)
    dups = duplicate_checks(df, roles.id_col)
    unexpected = unexpected_categories(df)
    ranges = range_checks(df)
    zeros = zero_value_report(df, numeric)
    outliers = outlier_summary(df, numeric)

    issues, warnings = [], []
    if roles.target_col is None:
        issues.append("No target column detected.")
    if int(miss["missing"].sum()):
        issues.append(f"{int(miss['missing'].sum())} missing cells.")
    if dups["duplicate_ids"]:
        issues.append(f"{dups['duplicate_ids']} duplicate patient IDs.")
    if len(unexpected):
        issues.append(f"Unexpected category codes in {len(unexpected)} column(s).")
    if dups["duplicates_excluding_id"]:
        warnings.append(f"{dups['duplicates_excluding_id']} rows duplicate another row on all non-ID columns.")
    for _, r in zeros[zeros["zero_values"] > 0].iterrows():
        warnings.append(f"'{r['column']}' has {r['zero_values']} zero values (not auto-treated as missing; verify).")
    if len(ranges) and int(ranges["rows_outside"].sum()):
        warnings.append(f"{int(ranges['rows_outside'].sum())} values outside generic plausible ranges.")
    status = "Issues found" if issues else ("Warnings" if warnings else "Passed")
    return {"status": status, "issues": issues, "warnings": warnings, "missing": miss, "duplicates": dups,
            "unexpected": unexpected, "ranges": ranges, "zeros": zeros, "outliers": outliers,
            "numeric": numeric, "categorical": categorical,
            "target_distribution": target_distribution(df, roles.target_col)}
