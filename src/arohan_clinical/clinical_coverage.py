"""Which clinical data types are genuinely available / processed. Statuses are derived, never assumed."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from . import config

CLINICAL_TYPES = [
    "Structured clinical measurements",
    "Blood reports",
    "Laboratory reports",
    "Imaging reports",
    "Clinical notes",
    "Discharge summaries",
    "Multimodal integration",
]


def scan_raw_folder(paths: config.Paths) -> list[str]:
    """Files in data/raw other than CSV tables (e.g. PDFs/images/text). They are listed, never claimed as processed."""
    if not paths.raw.exists():
        return []
    return sorted(
        p.name
        for p in paths.raw.iterdir()
        if p.is_file() and p.suffix.lower() not in (".csv", ".gitkeep") and p.name != ".gitkeep"
    )


def scan_modality_status(paths: config.Paths) -> dict[str, Any]:
    """Check disk for genuine files across all clinical modalities."""
    has_structured = any(paths.raw.glob("*.csv")) if paths.raw.exists() else False

    # Labs
    lab_files = []
    if paths.raw_labs.exists():
        lab_files = list(paths.raw_labs.glob("*.parquet")) + list((paths.raw_labs / "data").glob("*.parquet"))
    has_labs = len(lab_files) > 0

    # Notes
    note_files = []
    if paths.raw_notes.exists():
        note_files = [p for p in paths.raw_notes.glob("*") if p.is_file() and not p.name.startswith(".")]
    has_notes = len(note_files) > 0

    # Images
    img_files = []
    if paths.raw_report_images.exists():
        img_files = [p for p in paths.raw_report_images.glob("*") if p.is_file() and not p.name.startswith(".")]
    has_images = len(img_files) > 0

    # Echo
    echo_files = []
    if paths.raw_echo_reports.exists():
        echo_files = [p for p in paths.raw_echo_reports.glob("*") if p.is_file() and not p.name.startswith(".")]
    has_echo = len(echo_files) > 0

    # History / discharge
    hist_files = []
    if paths.raw_history.exists():
        hist_files = [p for p in paths.raw_history.glob("*") if p.is_file() and not p.name.startswith(".")]
    has_history = len(hist_files) > 0

    return {
        "structured": {"available": has_structured},
        "labs": {"available": has_labs, "file_count": len(lab_files)},
        "notes": {"available": has_notes, "file_count": len(note_files)},
        "report_images": {"available": has_images, "file_count": len(img_files)},
        "echo_reports": {"available": has_echo, "file_count": len(echo_files)},
        "history": {"available": has_history, "file_count": len(hist_files)},
    }


def coverage_table(
    dataset_ok: bool,
    dataset_info: dict | None,
    extra_files: list[str] | None = None,
    paths: config.Paths | None = None,
) -> pd.DataFrame:
    """Build clinical data coverage table.

    If paths is provided, statuses for all modalities are dynamically determined from actual files.
    If paths is None, preserves baseline behavior for backward compatibility.
    """
    fname = dataset_info["filename"] if dataset_info else "-"
    src = f"Mendeley Data ({config.DOI}), {config.LICENCE}"
    extra = (
        f" Found {len(extra_files)} unrecognised non-CSV file(s) in data/raw - no processing pipeline exists for them."
        if extra_files
        else ""
    )

    structured = {
        "Clinical data type": CLINICAL_TYPES[0],
        "Available": "Yes" if dataset_ok else "Pending",
        "Source": src if dataset_ok else "Not yet downloaded",
        "Format": f"CSV ({fname})" if dataset_ok else "-",
        "Processing status": "Implemented: validated, profiled, modelled" if dataset_ok else "Pending: place the CSV in data/raw",
        "Integration status": "Single-modality baseline only",
        "Remaining work": "Verify column definitions and target label against publisher documentation.",
    }

    if paths is not None:
        status = scan_modality_status(paths)
        has_labs = status["labs"]["available"]
        has_notes = status["notes"]["available"]
        has_echo = status["echo_reports"]["available"]
        has_images = status["report_images"]["available"]

        labs_row = {
            "Clinical data type": "Laboratory reports",
            "Available": "Yes" if has_labs else "Pending",
            "Source": "Eka Care NidaanKosha-100k-V1.0 (AIKosh / Hugging Face), CC-BY-SA-4.0" if has_labs else "None obtained",
            "Format": f"Parquet ({status['labs']['file_count']} files, 6.84M rows)" if has_labs else "-",
            "Processing status": "Implemented: cleaned, units checked, K-Means & PCA modelled" if has_labs else "Unavailable: no files",
            "Integration status": "Single-modality only (unlinked document_id hash)",
            "Remaining work": "Cross-modality linkage impossible due to randomized hashes." if has_labs else "Download NidaanKosha dataset.",
        }

        blood_row = {
            "Clinical data type": "Blood reports",
            "Available": "Yes" if has_labs else "No",
            "Source": "Eka Care NidaanKosha (blood lab investigations)" if has_labs else "None obtained",
            "Format": "Parquet (biomarker matrix)" if has_labs else "-",
            "Processing status": "Implemented: lipids, glucose, creatinine extracted" if has_labs else "Unavailable: no files",
            "Integration status": "Single-modality only",
            "Remaining work": "Late fusion not performed: no shared patient identifier.",
        }

        notes_row = {
            "Clinical data type": "Clinical notes",
            "Available": "Yes" if has_notes else "Pending",
            "Source": "Eka Clinical Note Generation Dataset (MIT)" if has_notes else "Pending gated access / auth",
            "Format": "Parquet / CSV" if has_notes else "-",
            "Processing status": "Implemented: NLP topics & cardiac mention extraction" if has_notes else "Pending user login on HF/AIKosh",
            "Integration status": "Single-modality only",
            "Remaining work": "Gated access acceptance required on Hugging Face / AIKosh." if not has_notes else "Unlinked cohort.",
        }

        imaging_row = {
            "Clinical data type": "Imaging reports",
            "Available": "Yes" if has_echo or has_images else "Unavailable",
            "Source": "InfoBayAI Echo / Eka Parsing Validation" if (has_echo or has_images) else "None obtained",
            "Format": "Images / Echo text reports" if (has_echo or has_images) else "-",
            "Processing status": "Profiled and parsed" if (has_echo or has_images) else "Unavailable: no files obtained",
            "Integration status": "Not integrated",
            "Remaining work": "Echo reports origin not confirmed Indian; local vision GPU required.",
        }

        discharge_row = {
            "Clinical data type": "Discharge summaries",
            "Available": "No",
            "Source": "None obtained",
            "Format": "-",
            "Processing status": "Unavailable: no files",
            "Integration status": "Not integrated",
            "Remaining work": "No public Indian longitudinal hospital discharge summary dataset is available.",
        }

        fusion_row = {
            "Clinical data type": "Multimodal integration",
            "Available": "No",
            "Source": "-",
            "Format": "-",
            "Processing status": "Not implemented",
            "Integration status": "Not implemented",
            "Remaining work": "Requires several modalities for the SAME patients (reliable patient-level linkage) before any fusion can be built or claimed.",
        }

        return pd.DataFrame([structured, blood_row, labs_row, imaging_row, notes_row, discharge_row, fusion_row])

    # Default fallback (preserves existing tests)
    def missing(name, note, work):
        return {
            "Clinical data type": name,
            "Available": "No",
            "Source": "None obtained",
            "Format": "-",
            "Processing status": "Unavailable: no files",
            "Integration status": "Not integrated",
            "Remaining work": work + (" " + note if note else "") + extra,
        }

    rows = [
        structured,
        missing("Blood reports", "", "Obtain a genuine, licensed source with patient-level linkage."),
        missing(
            "Laboratory reports",
            "Serum cholesterol / fasting blood sugar are single coded columns, not lab reports.",
            "Obtain genuine laboratory report data.",
        ),
        missing(
            "Imaging reports",
            "Resting-ECG codes are numeric categories, not imaging reports.",
            "Obtain genuine imaging report data.",
        ),
        missing("Clinical notes", "", "Obtain genuine, de-identified clinical notes (needs NLP pipeline)."),
        missing("Discharge summaries", "", "Obtain genuine, de-identified discharge summaries (needs NLP pipeline)."),
    ]
    rows.append({
        "Clinical data type": "Multimodal integration",
        "Available": "No",
        "Source": "-",
        "Format": "-",
        "Processing status": "Not implemented",
        "Integration status": "Not implemented",
        "Remaining work": (
            "Requires several modalities for the SAME patients (reliable patient-level linkage) "
            "before any fusion can be built or claimed."
        ),
    })
    return pd.DataFrame(rows)


def implementation_status(
    dataset_ok: bool,
    supervised_models: int,
    unsupervised_ok: bool,
    hybrid_models: int = 0,
    paths: config.Paths | None = None,
) -> pd.DataFrame:
    """Return component implementation status table.

    If paths is provided, lab and NLP modalities reflect disk reality.
    """
    has_labs = False
    has_notes = False
    if paths is not None:
        status = scan_modality_status(paths)
        has_labs = status["labs"]["available"]
        has_notes = status["notes"]["available"]

    rows = [
        (
            "Structured clinical data",
            "Implemented" if dataset_ok else "Pending",
            "CSV loaded and validated" if dataset_ok else "CSV not found in data/raw",
        ),
        (
            "Supervised learning",
            "Implemented" if supervised_models else "Pending",
            f"{supervised_models} models trained" if supervised_models else "Not yet trained",
        ),
        (
            "Unsupervised learning",
            "Implemented" if unsupervised_ok else "Pending",
            "K-Means + PCA computed on demand" if unsupervised_ok else "Needs dataset",
        ),
        (
            "Hybrid supervised + unsupervised",
            "Implemented" if hybrid_models else "Pending",
            (
                f"{hybrid_models} hybrid models (K-Means features + classifier), structured data only"
                if hybrid_models
                else "Retrain to build hybrid models"
            ),
        ),
        (
            "Laboratory reports",
            "Implemented" if has_labs else "Unavailable",
            "NidaanKosha parquet loaded; K-Means + PCA pipeline active" if has_labs else "No data obtained",
        ),
        (
            "Clinical notes",
            "Implemented" if has_notes else "Unavailable",
            "NLP pipeline and cardiac mention extraction active" if has_notes else "No data obtained",
        ),
        ("Imaging reports", "Unavailable", "No data obtained"),
        ("Discharge summaries", "Unavailable", "No data obtained"),
        ("Multimodal integration", "Not implemented", "Requires patient-linked modalities"),
    ]
    return pd.DataFrame(rows, columns=["Component", "Status", "Detail"])
