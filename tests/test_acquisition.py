"""Tests for dataset acquisition, folder management, and registry handling."""
from pathlib import Path

from arohan_clinical.acquisition import (
    RAW_SUBDIRS,
    discover_raw_files,
    ensure_data_folders,
    load_dataset_registry,
    register_dataset,
    sha256_file,
)
from arohan_clinical.config import Paths


def test_ensure_data_folders(tmp_path):
    paths = Paths(tmp_path)
    folders = ensure_data_folders(paths)
    for sub in RAW_SUBDIRS:
        assert (paths.raw / sub).exists()
        assert (paths.raw / sub).is_dir()
        assert sub in folders


def test_sha256_file(tmp_path):
    f = tmp_path / "sample.txt"
    f.write_text("Antigravity arohan clinical test content", encoding="utf-8")
    csum = sha256_file(f)
    assert isinstance(csum, str)
    assert len(csum) == 64
    # Check deterministic
    assert sha256_file(f) == csum


def test_registry_lifecycle(tmp_path):
    paths = Paths(tmp_path)
    paths.ensure()
    initial = load_dataset_registry(paths)
    assert initial == []

    record = {
        "name": "Test Mock Dataset",
        "modality": "Test modality",
        "source_url": "https://example.com/test",
        "licence": "MIT",
        "record_count": 42,
    }
    reg = register_dataset(paths, record)
    assert len(reg) == 1
    assert reg[0]["name"] == "Test Mock Dataset"
    assert "updated_at" in reg[0]

    # Loading from disk should match
    loaded = load_dataset_registry(paths)
    assert len(loaded) == 1
    assert loaded[0]["record_count"] == 42


def test_discover_raw_files(tmp_path):
    paths = Paths(tmp_path)
    paths.ensure()
    (paths.raw / "test_meas.csv").write_text("a,b\n1,2", encoding="utf-8")
    (paths.raw_labs / "lab1.parquet").write_bytes(b"PAR1mock")
    (paths.raw_notes / "note1.txt").write_text("patient note", encoding="utf-8")

    discovered = discover_raw_files(paths)
    assert len(discovered["structured"]) == 1
    assert len(discovered["labs"]) == 1
    assert len(discovered["notes"]) == 1
    assert len(discovered["echo_reports"]) == 0
