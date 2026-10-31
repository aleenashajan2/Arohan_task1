import numpy as np
import pytest

from arohan_clinical.data_loader import detect_roles
from arohan_clinical.unsupervised import run_analysis, save_outputs, select_features


def test_features_exclude_id_target_and_codes(frame):
    feats = select_features(frame, detect_roles(frame))
    assert set(feats) == {"age", "restingBP", "serumcholestrol", "maxheartrate", "oldpeak"}


def test_analysis_outputs(frame):
    r = run_analysis(frame, detect_roles(frame))
    assert r["sizes"]["rows"].sum() == len(frame)
    assert r["k"] == r["best_k"] and r["scores"]["silhouette"].between(-1, 1).all()
    assert r["explained"]["explained_variance"].sum() <= 1.0 + 1e-9
    assert len(r["pca"]) == len(frame) and {"PC1", "PC2", "cluster"} <= set(r["pca"].columns)


def test_reproducible_and_k_override(frame):
    roles = detect_roles(frame)
    a, b = run_analysis(frame, roles, k=3), run_analysis(frame, roles, k=3)
    assert a["k"] == 3 and a["pca"]["cluster"].equals(b["pca"]["cluster"])


def test_needs_two_features(frame):
    roles = detect_roles(frame[["patientid", "age", "target"]])
    with pytest.raises(ValueError):
        run_analysis(frame[["patientid", "age", "target"]], roles)


def test_save_outputs(project, frame):
    r = run_analysis(frame, detect_roles(frame))
    assert len(save_outputs(project, r)) == 4 and (project.tables / "unsupervised_sizes.csv").exists()
