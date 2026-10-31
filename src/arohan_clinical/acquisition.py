"""Data acquisition and registry management for AROHAN Clinical AI.

Manages data/raw/ subfolders, SHA-256 checksums, dataset_registry.json, and file discovery.
Only local file operations and the documented Hugging Face snapshot_download helper are executed.
"""
from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

from .config import Paths

RAW_SUBDIRS = ["labs", "notes", "history", "report_images", "echo_reports"]


def sha256_file(path: Path) -> str:
    """Compute hex-encoded SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def ensure_data_folders(paths: Paths) -> dict[str, Path]:
    """Ensure standard raw subfolders exist under data/raw/."""
    folders = {}
    for sub in RAW_SUBDIRS:
        p = paths.raw / sub
        p.mkdir(parents=True, exist_ok=True)
        folders[sub] = p
    return folders


def registry_path(paths: Paths) -> Path:
    """Return Path to outputs/reports/dataset_registry.json."""
    return paths.reports / "dataset_registry.json"


def load_dataset_registry(paths: Paths) -> list[dict[str, Any]]:
    """Load the dataset registry JSON, or return empty list if not yet created."""
    rp = registry_path(paths)
    if not rp.exists():
        return []
    try:
        with open(rp, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []


def save_dataset_registry(paths: Paths, registry: list[dict[str, Any]]) -> Path:
    """Write the dataset registry JSON."""
    rp = registry_path(paths)
    rp.parent.mkdir(parents=True, exist_ok=True)
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)
    return rp


def register_dataset(paths: Paths, info: dict[str, Any]) -> list[dict[str, Any]]:
    """Append or update a dataset record in dataset_registry.json."""
    registry = load_dataset_registry(paths)
    name = info.get("name")
    # Replace existing entry by name if present, else append
    existing_idx = next((i for i, d in enumerate(registry) if d.get("name") == name), None)
    record = {
        **info,
        "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    if existing_idx is not None:
        registry[existing_idx] = record
    else:
        registry.append(record)
    save_dataset_registry(paths, registry)
    return registry


def discover_raw_files(paths: Paths) -> dict[str, list[Path]]:
    """Discover existing files in each data/raw/ subfolder."""
    result: dict[str, list[Path]] = {
        "structured": sorted(p for p in paths.raw.glob("*.csv") if p.is_file()),
    }
    for sub in RAW_SUBDIRS:
        folder = paths.raw / sub
        if folder.exists():
            files = sorted(p for p in folder.rglob("*") if p.is_file() and not p.name.startswith("."))
            result[sub] = files
        else:
            result[sub] = []
    return result


def download_hf_dataset(repo_id: str, local_dir: Path, repo_type: str = "dataset") -> Path:
    """Download a Hugging Face dataset via snapshot_download into local_dir."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as err:
        raise ImportError("huggingface_hub is required for downloading Hugging Face datasets.") from err

    local_dir.mkdir(parents=True, exist_ok=True)
    path_str = snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=str(local_dir),
    )
    return Path(path_str)
