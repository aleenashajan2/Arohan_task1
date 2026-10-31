"""Locate, load, describe and fingerprint the raw dataset. The raw file is never modified."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import config
from .utils import read_json, sha256_file, utc_now, write_json


class DatasetError(Exception):
    """Raised for problems with the dataset that the user can act on."""


@dataclass
class Roles:
    id_col: str | None
    target_col: str | None
    feature_cols: list[str] = field(default_factory=list)


def find_dataset_file(raw_dir: Path) -> Path | None:
    if not raw_dir.exists():
        return None
    csvs = sorted(p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv")
    if not csvs:
        return None
    preferred = [p for p in csvs if "cardio" in p.name.lower()]
    return (preferred or csvs)[0]


def load_dataset(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise DatasetError(f"Could not read '{Path(path).name}' as CSV: {exc}") from exc
    df.columns = [str(c).strip() for c in df.columns]
    if df.empty or df.shape[1] < 2:
        raise DatasetError(f"'{Path(path).name}' has no usable rows/columns.")
    return df


def detect_roles(df: pd.DataFrame, target_override: str | None = None) -> Roles:
    override = target_override if target_override is not None else config.TARGET_OVERRIDE
    id_col = target_col = None
    for c in df.columns:
        key = config.canonical_key(c)
        if key == "patientid" and id_col is None:
            id_col = c
        if key == "target" and target_col is None:
            target_col = c
    if override and override in df.columns:
        target_col = override
    features = [c for c in df.columns if c not in (id_col, target_col)]
    return Roles(id_col, target_col, features)


def validate_required_columns(df: pd.DataFrame) -> list[str]:
    """Return expected dictionary keys that have no matching column (empty list = all present)."""
    present = {config.canonical_key(c) for c in df.columns}
    return [k for k in config.REQUIRED_KEYS if k not in present]


def build_dataset_info(path: Path, df: pd.DataFrame, roles: Roles) -> dict:
    stat = path.stat()
    return {
        "dataset_name": config.DATASET_NAME,
        "source_url": config.SOURCE_URL,
        "doi": config.DOI,
        "version": config.DATASET_VERSION,
        "licence": config.LICENCE,
        "published": config.PUBLISHED,
        "publisher_description": config.PUBLISHER_DESCRIPTION,
        "filename": path.name,
        "size_bytes": stat.st_size,
        "sha256": sha256_file(path),
        "file_modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "id_column": roles.id_col,
        "target_column": roles.target_col,
        "missing_expected_columns": validate_required_columns(df),
    }


def write_dataset_info(paths: config.Paths, info: dict) -> dict:
    """Write outputs/reports/dataset_info.json. Keeps the first-recorded time if the checksum is unchanged."""
    target = paths.reports / "dataset_info.json"
    previous = read_json(target)
    if previous and previous.get("sha256") == info["sha256"] and previous.get("first_recorded_utc"):
        info["first_recorded_utc"] = previous["first_recorded_utc"]
    else:
        info["first_recorded_utc"] = utc_now()
    write_json(target, info)
    return info
