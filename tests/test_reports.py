"""Tests for image profiling, echocardiography extraction, and multimodal fusion readiness."""
import pandas as pd

from arohan_clinical.config import Paths
from arohan_clinical.reports import (
    EXPORT_SCHEMA,
    extract_echo_measurements,
    get_fusion_readiness_table,
    get_parrotlet_status,
    profile_echo_reports,
    profile_report_images,
)


def test_profile_report_images_empty(tmp_path):
    paths = Paths(tmp_path)
    paths.ensure()
    res = profile_report_images(paths)
    assert res["available"] is False
    assert res["total_images"] == 0


def test_profile_report_images_with_files(tmp_path):
    paths = Paths(tmp_path)
    paths.ensure()
    (paths.raw_report_images / "lab_report_1.png").write_bytes(b"\x89PNG\r\n\x1a\nmock")
    (paths.raw_report_images / "lab_report_2.jpg").write_bytes(b"\xff\xd8\xffmock")

    res = profile_report_images(paths)
    assert res["available"] is True
    assert res["total_images"] == 2
    assert ".png" in res["formats"]
    assert ".jpg" in res["formats"]


def test_profile_report_images_with_parquet(tmp_path):
    paths = Paths(tmp_path)
    paths.ensure()
    df = pd.DataFrame({
        "image": [b"\xff\xd8\xffmock1", b"\xff\xd8\xffmock2"],
        "document_type": ["Lab-report", "Prescription"],
        "md5_image": ["md5_1", "md5_2"]
    })
    df.to_parquet(paths.raw_report_images / "sample.parquet")

    res = profile_report_images(paths)
    assert res["available"] is True
    assert res["total_images"] == 2
    assert "Lab-report" in res["formats"]
    assert "Prescription" in res["formats"]
    assert len(res["sample_images"]) == 2


def test_extract_echo_measurements():
    sample = "2D Echocardiogram shows LVEF: 52%, LVEDD: 46 mm and LA dimension of 35 mm. Normal valves."
    meas = extract_echo_measurements(sample)
    assert meas["lvef"] is not None
    assert meas["lvef"]["value"] == 52.0
    assert meas["lvef"]["plausible"] is True

    assert meas["lvedd"] is not None
    assert meas["lvedd"]["value"] == 46.0
    assert meas["lvedd"]["plausible"] is True

    assert meas["la_diameter"] is not None
    assert meas["la_diameter"]["value"] == 35.0
    assert meas["la_diameter"]["plausible"] is True


def test_extract_echo_extreme_anomaly():
    sample = "LVEF: 98% with LVEDD: 120 mm"
    meas = extract_echo_measurements(sample)
    # LVEF > 85% is clinically extreme / implausible
    assert meas["lvef"]["plausible"] is False
    # LVEDD > 80 mm is clinically extreme
    assert meas["lvedd"]["plausible"] is False


def test_profile_echo_reports_empty(tmp_path):
    paths = Paths(tmp_path)
    paths.ensure()
    res = profile_echo_reports(paths)
    assert res["available"] is False
    assert res["report_count"] == 0
    assert "not confirmed as Indian" in res["limitation"]


def test_get_parrotlet_status():
    stat = get_parrotlet_status()
    assert stat["status"] == "Model not run"
    assert "GPU" in stat["reason"]
    assert "external API" in stat["compliance"]


def test_fusion_readiness_table(tmp_path):
    paths = Paths(tmp_path)
    paths.ensure()
    df = get_fusion_readiness_table(paths)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
    for _, row in df.iterrows():
        assert "Late fusion not performed: no shared patient identifier" in row["Fusion Feasibility"]
    for k in ("record_id", "modality", "score_or_embedding", "cluster_id", "source_dataset", "patient_linkage_key"):
        assert k in EXPORT_SCHEMA
