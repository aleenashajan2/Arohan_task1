"""Tests for laboratory reports module (NidaanKosha lab investigations).

All fixtures are generated strictly inside tmp_path (test-only, never in data/raw).
"""
import numpy as np
import pandas as pd
import pytest

from arohan_clinical.labs import (
    CARDIOMETABOLIC_SPECS,
    build_lab_feature_matrix,
    clean_and_normalize_lab_readings,
    discover_lab_tests,
    get_cardiometabolic_mapping,
    run_lab_unsupervised,
)


@pytest.fixture
def mock_lab_df():
    """Create a small, test-only table matching NidaanKosha schema."""
    return pd.DataFrame([
        # Doc 1: Normal lipid profile + creatinine in umol/L
        {"document_id": "doc1", "age": 45, "gender": "male", "test_name": "Total Cholesterol", "value": "180.0", "unit": "mg/dL", "loinc": "2093-3"},
        {"document_id": "doc1", "age": 45, "gender": "male", "test_name": "Triglycerides", "value": "120.0", "unit": "mg/dL", "loinc": "2571-8"},
        {"document_id": "doc1", "age": 45, "gender": "male", "test_name": "Creatinine", "value": "88.4", "unit": "umol/L", "loinc": "2160-0"},
        # Doc 2: Elevated lipids + extreme anomaly creatinine
        {"document_id": "doc2", "age": 62, "gender": "female", "test_name": "Total Cholesterol", "value": "290.0", "unit": "mg/dL", "loinc": "2093-3"},
        {"document_id": "doc2", "age": 62, "gender": "female", "test_name": "Triglycerides", "value": "350.0", "unit": "mg/dL", "loinc": "2571-8"},
        {"document_id": "doc2", "age": 62, "gender": "female", "test_name": "Creatinine", "value": "999.0", "unit": "mg/dL", "loinc": "2160-0"}, # Outlier > 25
        # Doc 3: Non-numeric value noise
        {"document_id": "doc3", "age": 55, "gender": "male", "test_name": "Total Cholesterol", "value": "<100", "unit": "mg/dL", "loinc": "2093-3"},
        {"document_id": "doc3", "age": 55, "gender": "male", "test_name": "Glucose", "value": "95.0", "unit": "mg/dL", "loinc": "2345-7"},
        # Doc 4: Valid
        {"document_id": "doc4", "age": 50, "gender": "female", "test_name": "Total Cholesterol", "value": "210.0", "unit": "mg/dL", "loinc": "2093-3"},
        {"document_id": "doc4", "age": 50, "gender": "female", "test_name": "Triglycerides", "value": "160.0", "unit": "mg/dL", "loinc": "2571-8"},
        {"document_id": "doc4", "age": 50, "gender": "female", "test_name": "Creatinine", "value": "1.1", "unit": "mg/dL", "loinc": "2160-0"},
    ])


def test_cardiometabolic_mapping():
    mapping = get_cardiometabolic_mapping()
    assert "total cholesterol" in mapping
    assert mapping["total cholesterol"] == "total_cholesterol"
    assert "triglycerides" in mapping
    assert mapping["triglycerides"] == "triglycerides"
    assert "creatinine" in mapping
    assert mapping["creatinine"] == "creatinine"


def test_clean_and_normalize_creatinine_unit_conversion(mock_lab_df):
    cleaned, stats = clean_and_normalize_lab_readings(mock_lab_df)
    assert stats["unit_conversions_applied"] == 1
    # 88.4 umol/L / 88.4 = 1.0 mg/dL
    doc1_creat = cleaned[(cleaned["document_id"] == "doc1") & (cleaned["canon_biomarker"] == "creatinine")]["numeric_value"].iloc[0]
    assert np.isclose(doc1_creat, 1.0, atol=0.01)


def test_clean_and_normalize_anomaly_screening(mock_lab_df):
    cleaned, stats = clean_and_normalize_lab_readings(mock_lab_df)
    assert stats["anomalies_filtered"] == 1
    # doc2 creatinine = 999.0 should be set to NaN
    doc2_creat = cleaned[(cleaned["document_id"] == "doc2") & (cleaned["canon_biomarker"] == "creatinine")]["numeric_value"].iloc[0]
    assert np.isnan(doc2_creat)


def test_clean_and_normalize_non_numeric_handling(mock_lab_df):
    cleaned, stats = clean_and_normalize_lab_readings(mock_lab_df)
    assert stats["non_numeric_values"] == 1
    # doc3 '<100' should fail numeric coercion to NaN
    doc3_chol = cleaned[(cleaned["document_id"] == "doc3") & (cleaned["canon_biomarker"] == "total_cholesterol")]["numeric_value"].iloc[0]
    assert np.isnan(doc3_chol)


def test_build_lab_feature_matrix(mock_lab_df):
    cleaned, _ = clean_and_normalize_lab_readings(mock_lab_df)
    matrix = build_lab_feature_matrix(cleaned, min_tests_per_doc=2)
    assert not matrix.empty
    assert "document_id" in matrix.columns
    assert "total_cholesterol" in matrix.columns
    assert "triglycerides" in matrix.columns
    # doc1 and doc4 should have valid total_cholesterol and triglycerides
    assert len(matrix) >= 2


def test_run_lab_unsupervised(tmp_path):
    # Create test matrix with 20 rows
    rng = np.random.default_rng(42)
    rows = []
    for i in range(25):
        rows.append({
            "document_id": f"doc_{i}",
            "age": rng.integers(30, 75),
            "gender": "male" if i % 2 == 0 else "female",
            "total_cholesterol": float(rng.normal(200, 30)),
            "triglycerides": float(rng.normal(150, 40)),
            "creatinine": float(rng.normal(1.0, 0.2)),
            "glucose": float(rng.normal(100, 15)) if i % 3 != 0 else np.nan,  # missing values
        })
    df_matrix = pd.DataFrame(rows)

    res = run_lab_unsupervised(df_matrix, k_range=(2, 4), random_state=42)
    assert res["k"] in (2, 3, 4)
    assert len(res["cluster_labels"]) == 25
    assert "PC1" in res["pca_df"].columns
    assert "PC2" in res["pca_df"].columns
    assert len(res["cluster_summaries"]) == res["k"]
    assert "Lab Reports" in res["cluster_summaries"].columns

    # Reproducibility check with fixed seed
    res2 = run_lab_unsupervised(df_matrix, n_clusters=res["k"], random_state=42)
    np.testing.assert_array_equal(res["cluster_labels"], res2["cluster_labels"])


def test_discover_lab_tests(tmp_path, mock_lab_df):
    pq_path = tmp_path / "test_nidaankosha.parquet"
    mock_lab_df.to_parquet(pq_path)
    discovery = discover_lab_tests(pq_path, max_rows=100)
    assert isinstance(discovery, pd.DataFrame)
    assert "test_name" in discovery.columns
    assert "readings_count" in discovery.columns
    assert "Total Cholesterol" in discovery["test_name"].values
