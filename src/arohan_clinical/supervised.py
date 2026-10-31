"""Baseline supervised pipeline: Logistic Regression, Decision Tree, Random Forest.

Run from the command line with:  python -m arohan_clinical.supervised   (from the `src` folder)
"""
from __future__ import annotations

import platform
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from . import config
from .data_loader import DatasetError, Roles, detect_roles, find_dataset_file, load_dataset
from .evaluation import classification_metrics
from .preprocessing import build_preprocessor, coerce_features, split_data
from .utils import read_json, sha256_file, utc_now, write_json


def model_factories(seed: int = config.RANDOM_STATE) -> dict:
    """Fixed baseline hyper-parameters (not tuned)."""
    return {
        "Logistic Regression": lambda: LogisticRegression(max_iter=1000, random_state=seed),
        "Decision Tree": lambda: DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=seed),
        "Random Forest": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=1),
    }


def model_specs(seed: int = config.RANDOM_STATE) -> dict:
    """name -> (family, factory, uses_cluster_features). Hybrid = same classifier + K-Means distance features."""
    specs = {name: ("baseline", f, False) for name, f in model_factories(seed).items()}
    specs.update({f"Hybrid {name}": ("hybrid", f, True) for name, f in model_factories(seed).items()})
    return specs


def slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def model_path(paths: config.Paths, name: str):
    return paths.models / f"{slug(name)}.joblib"


def results_path(paths: config.Paths):
    return paths.reports / "supervised_results.json"


def _univariate_screen(X_tr: pd.DataFrame, y_tr: pd.Series, numeric: list[str], categorical: list[str]) -> list[dict]:
    """Single-feature AUC on TRAINING data only (in-sample screen for suspiciously strong features)."""
    rows = []
    for c in numeric:
        v = X_tr[c].fillna(X_tr[c].median())
        auc = roc_auc_score(y_tr, v) if v.nunique() > 1 else 0.5
        rows.append({"feature": c, "type": "numeric", "univariate_auc": float(max(auc, 1 - auc))})
    for c in categorical:
        rate = y_tr.groupby(X_tr[c].fillna("missing")).transform("mean")
        auc = roc_auc_score(y_tr, rate) if rate.nunique() > 1 else 0.5
        rows.append({"feature": c, "type": "categorical", "univariate_auc": float(max(auc, 1 - auc))})
    rows.sort(key=lambda r: r["univariate_auc"], reverse=True)
    for r in rows:
        r["flagged"] = bool(r["univariate_auc"] >= config.LEAKAGE_AUC_FLAG)
    return rows


def _input_spec(X_tr: pd.DataFrame, numeric: list[str], categorical: list[str]) -> dict:
    spec = {}
    for c in numeric:
        s = X_tr[c].dropna()
        d = config.spec_for(c) or {}
        spec[c] = {"kind": "numeric", "unit": d.get("unit", ""), "label": d.get("label", c),
                   "train_min": float(s.min()), "train_max": float(s.max()), "default": float(s.median()),
                   "plausible": list(d["plausible"]) if d.get("plausible") else None}
    for c in categorical:
        d = config.spec_for(c) or {}
        opts = sorted(X_tr[c].dropna().unique().tolist(), key=lambda v: (len(str(v)), str(v)))
        spec[c] = {"kind": "categorical", "options": [str(o) for o in opts], "label": d.get("label", c)}
    return spec


def run_training(paths: config.Paths, df: pd.DataFrame | None = None, roles=None,
                 seed: int = config.RANDOM_STATE, test_size: float = config.TEST_SIZE,
                 cv_folds: int = config.CV_FOLDS) -> dict:
    paths.ensure()
    csv = find_dataset_file(paths.raw)
    if df is None:
        if csv is None:
            raise DatasetError(f"No CSV found in {paths.raw}. See the README for the download steps.")
        df = load_dataset(csv)
    roles = roles or detect_roles(df)
    if roles.target_col is None:
        raise DatasetError("No target column detected. Columns found: " + ", ".join(df.columns) +
                           ". Set the AROHAN_TARGET environment variable to the label column.")

    X_tr, X_te, y_tr, y_te, numeric, categorical, classes = split_data(df, roles, test_size, seed)
    overlap = int(pd.util.hash_pandas_object(X_te, index=False).isin(pd.util.hash_pandas_object(X_tr, index=False)).sum())

    # processed data saved separately from raw (identifier is not included)
    pd.concat([X_tr, y_tr.rename("target")], axis=1).to_csv(paths.processed / "train.csv", index=False)
    pd.concat([X_te, y_te.rename("target")], axis=1).to_csv(paths.processed / "test.csv", index=False)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    models = {}
    for name, (family, factory, hybrid) in model_specs(seed).items():
        if hybrid and len(numeric) < 2:
            continue
        pipe = Pipeline([("prep", build_preprocessor(numeric, categorical, hybrid, config.HYBRID_K)), ("clf", factory())])
        cv_res = cross_validate(pipe, X_tr, y_tr, cv=cv, scoring=["accuracy", "recall", "f1", "roc_auc"], n_jobs=1)
        pipe.fit(X_tr, y_tr)
        n_cl = int(pipe.named_steps["prep"].named_transformers_["cluster"].named_steps["kmeans"].n_clusters_) if hybrid else None
        y_pred = pipe.predict(X_te)
        y_score = pipe.predict_proba(X_te)[:, 1]
        joblib.dump(pipe, model_path(paths, name))
        models[name] = {
            "family": family, "n_clusters": n_cl,
            "test_metrics": classification_metrics(y_te, y_pred, y_score),
            "cv": {m: {"mean": float(np.mean(cv_res[f"test_{m}"])), "std": float(np.std(cv_res[f"test_{m}"]))}
                   for m in ("accuracy", "recall", "f1", "roc_auc")},
            "y_pred": [int(v) for v in y_pred], "y_score": [float(v) for v in y_score],
            "model_file": model_path(paths, name).name,
        }

    results = {
        "trained_utc": utc_now(),
        "dataset_sha256": sha256_file(csv) if csv else None,
        "dataset_file": csv.name if csv else None,
        "target_column": roles.target_col, "classes": classes, "positive_class": classes[1],
        "id_column_excluded": roles.id_col, "features": roles.feature_cols,
        "numeric_features": numeric, "categorical_features": categorical,
        "split": {"test_size": test_size, "seed": seed, "stratified": bool(y_tr.value_counts().min() >= 2),
                  "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
                  "train_positive": int(y_tr.sum()), "test_positive": int(y_te.sum()),
                  "exact_feature_duplicates_between_train_and_test": overlap},
        "cv_folds": cv_folds, "y_test": [int(v) for v in y_te],
        "models": models,
        "hybrid_description": "Hybrid = the same classifier plus K-Means distance-to-cluster features computed from the "
                              "continuous measurements. Clusters are fitted on training rows only (target unused). "
                              "Structured data only.",
        "leakage_screen": _univariate_screen(X_tr, y_tr, numeric, categorical),
        "input_spec": _input_spec(X_tr, numeric, categorical),
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        "scikit_learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
    }
    write_json(results_path(paths), results)
    from .evaluation import metrics_table
    metrics_table(results).to_csv(paths.tables / "supervised_test_metrics.csv", index=False)
    return results


def run_ablation(df: pd.DataFrame, roles: Roles, drop: list[str], seed: int = config.RANDOM_STATE,
                 test_size: float = config.TEST_SIZE) -> pd.DataFrame:
    """Train every model with and without the chosen features, on the SAME split, and compare test scores."""
    drop = [d for d in drop if d in roles.feature_cols]
    if not drop:
        raise ValueError("Choose at least one feature that exists in the dataset.")
    keep = [c for c in roles.feature_cols if c not in drop]
    if not keep:
        raise ValueError("Cannot remove every feature.")
    rows = []
    for label, cols in (("All features", roles.feature_cols), ("Without " + ", ".join(drop), keep)):
        X_tr, X_te, y_tr, y_te, num, cat, _ = split_data(df, Roles(roles.id_col, roles.target_col, cols), test_size, seed)
        for name, (family, factory, hybrid) in model_specs(seed).items():
            if hybrid and len(num) < 2:
                continue
            pipe = Pipeline([("prep", build_preprocessor(num, cat, hybrid, config.HYBRID_K)), ("clf", factory())]).fit(X_tr, y_tr)
            m = classification_metrics(y_te, pipe.predict(X_te), pipe.predict_proba(X_te)[:, 1])
            rows.append({"Variant": label, "Model": name, "Type": family, "Accuracy": m["accuracy"], "Recall": m["recall"],
                         "Specificity": m["specificity"], "F1": m["f1"], "ROC-AUC": m["roc_auc"]})
    return pd.DataFrame(rows)


def load_results(paths: config.Paths) -> dict | None:
    res = read_json(results_path(paths))
    if not res or not all(model_path(paths, n).exists() for n in res.get("models", {})):
        return None
    return res


def is_stale(results: dict, dataset_sha: str | None) -> bool:
    return bool(dataset_sha and results.get("dataset_sha256") != dataset_sha)


def load_model(paths: config.Paths, name: str):
    return joblib.load(model_path(paths, name))


def validate_input(spec: dict, values: dict) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) for a single prediction request."""
    errors, warnings = [], []
    for col, s in spec.items():
        if col not in values or values[col] is None or values[col] == "":
            errors.append(f"{s.get('label', col)} is required.")
            continue
        if s["kind"] == "numeric":
            try:
                v = float(values[col])
            except (TypeError, ValueError):
                errors.append(f"{s['label']} must be a number.")
                continue
            if not np.isfinite(v):
                errors.append(f"{s['label']} must be finite.")
            elif s.get("plausible") and not (s["plausible"][0] <= v <= s["plausible"][1]):
                errors.append(f"{s['label']} = {v:g} is outside the plausible range {s['plausible'][0]} to {s['plausible'][1]}.")
            elif not (s["train_min"] <= v <= s["train_max"]):
                warnings.append(f"{s['label']} = {v:g} is outside the training range "
                                f"({s['train_min']:g} to {s['train_max']:g}); the model is extrapolating.")
        elif str(values[col]) not in s["options"]:
            errors.append(f"{s['label']} = {values[col]} was not seen in training (allowed: {', '.join(s['options'])}).")
    return errors, warnings


def predict_one(model, results: dict, values: dict) -> dict:
    """Uncalibrated model score for one record, using exactly the saved feature set."""
    feats = results["features"]
    row = pd.DataFrame([{c: values[c] for c in feats}])
    row = coerce_features(row, results["numeric_features"], results["categorical_features"])
    score = float(model.predict_proba(row)[0, 1])
    pred = int(model.predict(row)[0])
    return {"predicted_class": results["classes"][pred], "score": score}


if __name__ == "__main__":
    res = run_training(config.default_paths())
    for n, m in res["models"].items():
        t = m["test_metrics"]
        print(f"{n}: acc={t['accuracy']:.3f} recall={t['recall']:.3f} roc_auc={t['roc_auc']:.3f}")
