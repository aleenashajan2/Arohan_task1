"""Imaging reports and parsing validation module (lab-report images and echo reports).

Provides:
- Profiling of raw lab-report images (file counts, format distribution, size statistics).
- Regex-based structured extraction of echocardiography measurements (LVEF, LVEDD, etc.).
- Transparent model execution tracking (Parrotlet-V-lite-4B model status).
- Multimodal fusion readiness matrix and unified export schema definition.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Paths


def profile_report_images(paths: Paths) -> dict[str, Any]:
    """Profile lab-report images in data/raw/report_images/ (supports image files and parquet image datasets)."""
    folder = paths.raw_report_images
    if not folder.exists():
        return {
            "available": False,
            "total_images": 0,
            "formats": {},
            "total_size_bytes": 0,
            "sample_images": [],
            "note": "No image files found in data/raw/report_images/.",
        }

    image_extensions = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".pdf"}
    files = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in image_extensions]
    parquet_files = [p for p in folder.rglob("*.parquet") if p.is_file()]

    if not files and not parquet_files:
        return {
            "available": False,
            "total_images": 0,
            "formats": {},
            "total_size_bytes": 0,
            "sample_images": [],
            "note": "Folder exists but contains no image or parquet files.",
        }

    formats: dict[str, int] = {}
    total_size = 0
    samples: list[dict[str, Any]] = []

    # Standalone image files
    for f in files:
        ext = f.suffix.lower()
        formats[ext] = formats.get(ext, 0) + 1
        total_size += f.stat().st_size

    # Parquet image datasets (e.g. Hugging Face ekacare/medical_records_parsing_validation_set)
    parquet_image_count = 0
    for pf in parquet_files:
        try:
            df_pq = pd.read_parquet(pf)
            if "image" in df_pq.columns:
                parquet_image_count += len(df_pq)
                doc_col = "document_type" if "document_type" in df_pq.columns else None
                for idx, row in df_pq.iterrows():
                    val = row["image"]
                    b = val if isinstance(val, bytes) else (val.get("bytes", b"") if isinstance(val, dict) else b"")
                    size = len(b) if isinstance(b, (bytes, bytearray)) else 0
                    total_size += size
                    dtype = str(row[doc_col]) if doc_col else "Medical image"
                    formats[dtype] = formats.get(dtype, 0) + 1
                    if len(samples) < 6 and isinstance(b, (bytes, bytearray)) and len(b) > 0:
                        samples.append({
                            "bytes": b,
                            "type": dtype,
                            "id": str(row.get("md5_image", f"img_{idx}"))[:12]
                        })
        except Exception:
            continue

    total_count = len(files) + parquet_image_count
    if total_count == 0:
        return {
            "available": False,
            "total_images": 0,
            "formats": {},
            "total_size_bytes": 0,
            "sample_images": [],
            "note": "Folder contains parquet files but no valid image records found.",
        }

    return {
        "available": True,
        "total_images": total_count,
        "formats": formats,
        "total_size_bytes": total_size,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "avg_size_kb": round((total_size / total_count) / 1024, 1) if total_count else 0,
        "sample_images": samples,
        "note": f"Found {total_count} images across formats/document types {list(formats.keys())}.",
    }


# Standard echo report extraction regexes
ECHO_MEASUREMENT_PATTERNS = {
    "lvef": {
        "label": "Left Ventricular Ejection Fraction (LVEF)",
        "unit": "%",
        "pattern": r"(?:lvef|ejection\s*fraction|ef)(?:[\s:=]|of|is)*([0-9]{1,3}(?:\.[0-9]+)?)\s*%",
        "plausible": (10.0, 85.0),
    },
    "lvedd": {
        "label": "LV End-Diastolic Dimension (LVEDD)",
        "unit": "mm",
        "pattern": r"(?:lvedd|lvid[d]?|lv\s*internal\s*diameter\s*diastole)(?:[\s:=]|of|is)*([0-9]{1,3}(?:\.[0-9]+)?)\s*(?:mm|cm)",
        "plausible": (20.0, 80.0),
    },
    "la_diameter": {
        "label": "Left Atrial (LA) Dimension",
        "unit": "mm",
        "pattern": r"(?:la\s*dimension|la\s*diameter|left\s*atrium)(?:[\s:=]|of|is)*([0-9]{1,3}(?:\.[0-9]+)?)\s*(?:mm|cm)",
        "plausible": (15.0, 70.0),
    },
}


def extract_echo_measurements(text: str) -> dict[str, Any]:
    """Extract clinical echocardiography metrics from text using validated regex patterns."""
    text_lower = text.lower()
    measurements: dict[str, Any] = {}
    for metric, spec in ECHO_MEASUREMENT_PATTERNS.items():
        match = re.search(spec["pattern"], text_lower)
        if match:
            try:
                val = float(match.group(1))
                low, high = spec["plausible"]
                # Convert cm to mm if needed
                if "cm" in match.group(0) and val < 10.0:
                    val = val * 10.0
                is_valid = low <= val <= high
                measurements[metric] = {
                    "value": val,
                    "unit": spec["unit"],
                    "label": spec["label"],
                    "plausible": is_valid,
                }
            except Exception:
                measurements[metric] = None
        else:
            measurements[metric] = None
    return measurements


def profile_echo_reports(paths: Paths) -> dict[str, Any]:
    """Profile echocardiography reports in data/raw/echo_reports/."""
    folder = paths.raw_echo_reports
    if not folder.exists():
        return {
            "available": False,
            "report_count": 0,
            "extracted_metrics": [],
            "note": "No reports found in data/raw/echo_reports/.",
            "limitation": "Country of origin is not confirmed as Indian (InfoBayAI dataset).",
        }

    files = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in (".txt", ".json", ".csv", ".parquet")]
    if not files:
        return {
            "available": False,
            "report_count": 0,
            "extracted_metrics": [],
            "note": "Folder exists but contains no recognized text or data reports.",
            "limitation": "Country of origin is not confirmed as Indian.",
        }

    return {
        "available": True,
        "report_count": len(files),
        "note": f"Found {len(files)} echocardiography report files.",
        "limitation": "Origin not confirmed as Indian; manual research verification required.",
    }


def get_parrotlet_status() -> dict[str, Any]:
    """Return local status and execution justification for Parrotlet-V-lite-4B vision LLM."""
    return {
        "model_name": "Parrotlet-V-lite-4B (Eka Care / IndiaAI)",
        "status": "Model not run",
        "reason": (
            "Vision LLM parsing requires dedicated local GPU hardware (minimum 12 GB VRAM) "
            "and local model weights. To preserve strict clinical research integrity, "
            "no mocked parsing outputs are generated, and no external hosted APIs are contacted."
        ),
        "compliance": "Fully compliant with local-only, zero external API transmission rule.",
    }


def get_fusion_readiness_table(paths: Paths) -> pd.DataFrame:
    """Return a transparent table of modality-specific outputs and cross-modality linkage status."""
    rows = [
        {
            "Modality": "Structured clinical measurements",
            "Dataset": "Cardiovascular Disease Dataset (Mendeley)",
            "Identifier": "patientid (synthetic/sequential 1..1000)",
            "Output Type": "Binary classifier probability & K-Means distance features",
            "Cross-Modality Linkage": "Unlinked: no shared patient identifier with other datasets",
            "Fusion Feasibility": "Late fusion not performed: no shared patient identifier",
        },
        {
            "Modality": "Laboratory reports",
            "Dataset": "NidaanKosha 100k (Eka Care / AIKosh)",
            "Identifier": "document_id (randomized 32-char hex hash)",
            "Output Type": "K-Means cluster IDs & PCA 2D coordinates",
            "Cross-Modality Linkage": "Unlinked: anonymized hash prevents any external linkage",
            "Fusion Feasibility": "Late fusion not performed: no shared patient identifier",
        },
        {
            "Modality": "Clinical notes",
            "Dataset": "Clinical Note Generation Dataset (Eka Care)",
            "Identifier": "session_id (consultation identifier)",
            "Output Type": "TF-IDF / TruncatedSVD topics & weak-label predictions",
            "Cross-Modality Linkage": "Unlinked: separate consultation sample",
            "Fusion Feasibility": "Late fusion not performed: no shared patient identifier",
        },
        {
            "Modality": "Imaging / Echocardiography reports",
            "Dataset": "InfoBayAI Echo / Eka Parsing Validation",
            "Identifier": "image_id / filename",
            "Output Type": "Structured measurement extraction (LVEF, dimensions)",
            "Cross-Modality Linkage": "Unlinked: distinct clinical cohorts",
            "Fusion Feasibility": "Late fusion not performed: no shared patient identifier",
        },
    ]
    return pd.DataFrame(rows)


EXPORT_SCHEMA = {
    "record_id": "string — Unique identifier within the source modality",
    "modality": "string — One of: structured, labs, notes, imaging, echo",
    "score_or_embedding": "list[float] | float — Model score, risk probability, or latent vector",
    "cluster_id": "string | int — Unsupervised cluster assignment (e.g. 'Cluster 1')",
    "source_dataset": "string — Name and DOI/URL of originating dataset",
    "patient_linkage_key": "null — Explicitly null until unified patient ID registry exists",
}
