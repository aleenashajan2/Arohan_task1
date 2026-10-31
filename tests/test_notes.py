"""Tests for clinical notes NLP, mention extraction, and weak-label modeling."""
import pandas as pd
import pytest

from arohan_clinical.notes import (
    clean_clinical_text,
    derive_weak_cardiac_label,
    extract_cardiac_mentions,
    run_notes_supervised,
    run_notes_unsupervised,
)


def test_clean_clinical_text():
    raw = "Patient  presents with CHEST PAIN! & breathlessness...   (bp: 140/90)"
    cleaned = clean_clinical_text(raw)
    assert "chest pain" in cleaned
    assert "breathlessness" in cleaned
    assert "  " not in cleaned


def test_extract_cardiac_mentions():
    note = "Patient with acute coronary artery disease, breathlessness, and diabetes. Started on aspirin."
    mentions = extract_cardiac_mentions(note)
    cats = mentions["mentions_by_category"]
    assert "breathlessness" in cats["symptoms"]
    assert "diabetes" in cats["risk_factors"]
    assert "coronary artery disease" in cats["cardiac_conditions"]
    assert "aspirin" in cats["medications"]
    assert mentions["has_cardiac_condition"] is True
    assert mentions["total_cardiac_mentions"] >= 4


def test_derive_weak_cardiac_label():
    cardiac_row = {"diagnosis": "acute myocardial infarction", "text": "severe angina"}
    lbl, name = derive_weak_cardiac_label(cardiac_row)
    assert lbl == 1
    assert "Cardiovascular disease" in name

    healthy_row = {"diagnosis": "allergic rhinitis", "text": "sneezing and runny nose"}
    lbl, name = derive_weak_cardiac_label(healthy_row)
    assert lbl == 0
    assert name == "Healthy"


@pytest.fixture
def test_notes_df():
    data = [
        {"session_id": f"s_{i}", "text": f"Patient reports chest pain and breathlessness. CAD condition {i}", "diagnosis": "coronary artery disease"}
        for i in range(12)
    ] + [
        {"session_id": f"s_{i+12}", "text": f"Routine annual checkup, seasonal cold, runny nose {i}", "diagnosis": "viral infection"}
        for i in range(12)
    ]
    df = pd.DataFrame(data)
    df["weak_label"] = [derive_weak_cardiac_label(r)[0] for _, r in df.iterrows()]
    return df


def test_run_notes_unsupervised(test_notes_df):
    res = run_notes_unsupervised(test_notes_df["text"], n_topics=2, n_clusters=2)
    assert res["n_topics"] == 2
    assert res["n_clusters"] == 2
    assert len(res["cluster_labels"]) == len(test_notes_df)
    assert "top_terms" in res["topics"].columns


def test_run_notes_supervised_leakage_guard(test_notes_df):
    # Leakage guard: passing diagnosis as input text feature must raise ValueError
    with pytest.raises(ValueError, match="Data leakage guard violation"):
        run_notes_supervised(test_notes_df, text_col="diagnosis", label_col="weak_label", raw_source_field="diagnosis")


def test_run_notes_supervised_pipeline(test_notes_df):
    res = run_notes_supervised(
        test_notes_df,
        text_col="text",
        label_col="weak_label",
        raw_source_field="diagnosis",
        test_size=0.25,
        random_state=42,
    )
    assert res["small_sample_warning"] is True
    assert "baseline_tfidf" in res["held_out_test_metrics"]
    assert "hybrid_svd" in res["held_out_test_metrics"]
    assert "train_cross_validation_accuracy" in res
    assert res["n_train"] + res["n_test"] == len(test_notes_df)
    assert "weak_label_disclaimer" in res
