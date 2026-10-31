import numpy as np
import pandas as pd

from arohan_clinical.data_loader import detect_roles
from arohan_clinical.preprocessing import (build_preprocessor, coerce_features, infer_feature_types, prepare_target,
                                           split_data)


def test_feature_types(frame):
    roles = detect_roles(frame)
    num, cat = infer_feature_types(frame, roles.feature_cols)
    assert set(num) == {"age", "restingBP", "serumcholestrol", "maxheartrate", "oldpeak"}
    assert "patientid" not in num + cat and "target" not in num + cat


def test_split_is_disjoint_stratified_and_reproducible(frame):
    roles = detect_roles(frame)
    a = split_data(frame, roles, 0.2, 7)
    b = split_data(frame, roles, 0.2, 7)
    X_tr, X_te, y_tr, y_te = a[:4]
    assert set(X_tr.index).isdisjoint(X_te.index)
    assert len(X_tr) + len(X_te) == len(frame)
    assert abs(y_tr.mean() - y_te.mean()) < 0.03
    assert X_tr.index.equals(b[0].index)
    assert "patientid" not in X_tr.columns and "target" not in X_tr.columns


def test_preprocessor_fit_on_training_only(frame):
    roles = detect_roles(frame)
    X_tr, X_te, y_tr, y_te, num, cat, _ = split_data(frame, roles)
    prep = build_preprocessor(num, cat).fit(X_tr)
    scaler = prep.named_transformers_["num"].named_steps["scale"]
    np.testing.assert_allclose(scaler.mean_, X_tr[num].mean().values)
    assert not np.allclose(scaler.mean_, frame[num].mean().values)


def test_inference_consistent_and_shapes(frame):
    roles = detect_roles(frame)
    X_tr, X_te, y_tr, y_te, num, cat, _ = split_data(frame, roles)
    prep = build_preprocessor(num, cat).fit(X_tr)
    batch = prep.transform(X_te)
    one = prep.transform(X_te.iloc[[3]])
    dense = lambda m: m.toarray() if hasattr(m, "toarray") else m
    np.testing.assert_allclose(dense(batch)[3], dense(one)[0])
    expected = len(num) + sum(X_tr[c].nunique() for c in cat)
    assert dense(batch).shape == (len(X_te), expected)


def test_unknown_category_does_not_crash(frame):
    roles = detect_roles(frame)
    X_tr, X_te, y_tr, y_te, num, cat, _ = split_data(frame, roles)
    prep = build_preprocessor(num, cat).fit(X_tr)
    row = X_te.iloc[[0]].copy()
    row["chestpain"] = "9"
    assert prep.transform(row).shape[0] == 1


def test_prepare_target_requires_binary():
    y, kept, classes = prepare_target(pd.Series([0, 1, 1, 0, np.nan]))
    assert classes == [0.0, 1.0] and kept.sum() == 4 and set(y) == {0, 1}
    import pytest
    with pytest.raises(ValueError):
        prepare_target(pd.Series([0, 1, 2]))


def test_cluster_features_fit_on_training_only_and_shape(frame):
    from arohan_clinical.preprocessing import KMeansDistanceFeatures
    roles = detect_roles(frame)
    X_tr, X_te, y_tr, y_te, num, cat, _ = split_data(frame, roles)
    prep = build_preprocessor(num, cat, add_clusters=True, n_clusters=4).fit(X_tr)
    km = prep.named_transformers_["cluster"].named_steps["kmeans"]
    assert km.kmeans_.labels_.shape[0] == len(X_tr)           # fitted on training rows only
    out = prep.transform(X_te)
    dense = out.toarray() if hasattr(out, "toarray") else out
    base = build_preprocessor(num, cat).fit(X_tr).transform(X_te)
    base = base.toarray() if hasattr(base, "toarray") else base
    assert dense.shape[1] == base.shape[1] + 4
    assert (dense[:, -4:] >= 0).all()                          # distances


def test_cluster_features_ignore_target(frame):
    roles = detect_roles(frame)
    X_tr, X_te, y_tr, y_te, num, cat, _ = split_data(frame, roles)
    a = build_preprocessor(num, cat, add_clusters=True, n_clusters=3).fit(X_tr, y_tr)
    b = build_preprocessor(num, cat, add_clusters=True, n_clusters=3).fit(X_tr, 1 - y_tr)
    np.testing.assert_allclose(a.named_transformers_["cluster"].named_steps["kmeans"].kmeans_.cluster_centers_,
                               b.named_transformers_["cluster"].named_steps["kmeans"].kmeans_.cluster_centers_)
