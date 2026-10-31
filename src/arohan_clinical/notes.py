"""Clinical notes processing module (Eka Care clinical notes & patient history).

Provides:
- Clinical text cleaning and tokenization.
- Transparent rule-based extraction of cardiology mentions (symptoms, risk factors, conditions, medications).
- Unsupervised topic modeling via TF-IDF, TruncatedSVD, and K-Means.
- Documented weak-label derivation with strict leakage prevention (source label field excluded from inputs).
- Stratified held-out train/test split with pipeline fitting on training rows only.
- Baseline and hybrid classifiers with separate training-only cross-validation and small-sample warnings.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from .config import Paths

# Transparent cardiology clinical vocabulary
CARDIAC_DICTIONARY: dict[str, list[str]] = {
    "symptoms": [
        "chest pain", "angina", "breathlessness", "dyspnea", "shortness of breath",
        "palpitation", "palpitations", "diaphoresis", "sweating", "syncope",
        "fatigue", "edema", "orthopnea", "dizziness"
    ],
    "risk_factors": [
        "hypertension", "high blood pressure", "htn", "diabetes", "diabetic",
        "dm", "t2dm", "smoking", "smoker", "tobacco", "dyslipidemia",
        "hyperlipidemia", "obesity", "family history"
    ],
    "cardiac_conditions": [
        "coronary artery disease", "cad", "myocardial infarction", "heart attack",
        "heart failure", "chf", "arrhythmia", "atrial fibrillation", "afib",
        "ischemic heart disease", "ihd", "cardiomyopathy", "valvular heart disease",
        "stent", "angioplasty", "cabg"
    ],
    "medications": [
        "aspirin", "ecosprin", "atorvastatin", "rosuvastatin", "clopidogrel",
        "metoprolol", "atenolol", "bisoprolol", "amlodipine", "ramipril",
        "telmisartan", "losartan", "nitroglycerin", "sorbitrate", "statin",
        "beta blocker", "ace inhibitor", "arb"
    ],
}


def clean_clinical_text(text: Any) -> str:
    """Standardize and clean free-form clinical text."""
    if text is None or pd.isna(text):
        return ""
    t = str(text).lower()
    # Replace non-alphanumeric chars except basic punctuation
    t = re.sub(r"[^a-z0-9\s\.,\-]", " ", t)
    # Collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t


def extract_cardiac_mentions(text: str) -> dict[str, Any]:
    """Transparent rule-based detection of cardiology symptoms, risk factors, conditions, and medications.

    Caveat: Keyword matching does not reliably handle negation (e.g. 'no chest pain')
    or hypothetical discussions. Mentions are descriptive, not diagnoses.
    """
    cleaned = clean_clinical_text(text)
    results: dict[str, list[str]] = {}
    total_mentions = 0

    for category, terms in CARDIAC_DICTIONARY.items():
        found = []
        for term in terms:
            # Word boundary match
            pattern = rf"\b{re.escape(term)}\b"
            if re.search(pattern, cleaned):
                found.append(term)
        results[category] = found
        total_mentions += len(found)

    return {
        "mentions_by_category": results,
        "total_cardiac_mentions": total_mentions,
        "has_cardiac_condition": len(results["cardiac_conditions"]) > 0,
        "has_cardiac_symptom": len(results["symptoms"]) > 0,
        "caveat": "Rule-based keyword extraction does not account for negation or temporal context.",
    }


def derive_weak_cardiac_label(
    row: pd.Series | dict[str, Any],
    source_field: str = "diagnosis",
    text_fallback_field: str = "text",
) -> tuple[int, str]:
    """Derive a weak binary label (1 = cardiac / 0 = non-cardiac) based on explicit condition mentions.

    Weak label rule:
    - If diagnosis/source_field contains known cardiac condition keywords -> 1
    - Else if no cardiac condition mentions -> 0
    Documented limitation: Weak labels have inherent label noise and are NOT gold-standard clinical adjudications.
    """
    raw_source = str(row.get(source_field, "") or "")
    if not raw_source.strip() and text_fallback_field in row:
        raw_source = str(row.get(text_fallback_field, "") or "")

    cleaned_source = clean_clinical_text(raw_source)
    matched_conditions = []
    for term in CARDIAC_DICTIONARY["cardiac_conditions"]:
        if re.search(rf"\b{re.escape(term)}\b", cleaned_source):
            matched_conditions.append(term)

    if matched_conditions:
        name = f"Cardiovascular disease ({matched_conditions[0].title()})"
        return 1, name
    return 0, "Healthy"


def run_notes_unsupervised(
    texts: list[str] | pd.Series,
    n_topics: int = 5,
    n_clusters: int = 3,
    max_features: int = 500,
    random_state: int = 42,
) -> dict[str, Any]:
    """Perform unsupervised topic extraction (TF-IDF + TruncatedSVD) and K-Means clustering."""
    cleaned_texts = [clean_clinical_text(t) for t in texts if clean_clinical_text(t)]
    if len(cleaned_texts) < 5:
        raise ValueError("At least 5 non-empty clinical text records required for topic modeling.")

    # Adaptive parameters for small datasets
    min_df = 1 if len(cleaned_texts) < 20 else 2
    n_topics = min(n_topics, len(cleaned_texts) - 1, max(1, max_features // 20))
    n_clusters = min(n_clusters, len(cleaned_texts))

    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=0.98,
        stop_words="english",
    )
    X_tfidf = tfidf.fit_transform(cleaned_texts)

    svd = TruncatedSVD(n_components=n_topics, random_state=random_state)
    X_topics = svd.fit_transform(X_tfidf)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X_topics)

    # Extract top terms per topic
    feature_names = tfidf.get_feature_names_out()
    topics_summary = []
    for topic_idx, comp in enumerate(svd.components_):
        top_idx = comp.argsort()[-6:][::-1]
        top_terms = [feature_names[i] for i in top_idx]
        topics_summary.append({
            "topic": f"Topic {topic_idx + 1}",
            "top_terms": ", ".join(top_terms),
            "explained_variance": round(float(svd.explained_variance_ratio_[topic_idx]), 4),
        })

    return {
        "n_samples": len(cleaned_texts),
        "n_topics": n_topics,
        "n_clusters": n_clusters,
        "topics": pd.DataFrame(topics_summary),
        "cluster_labels": cluster_labels,
        "models": {"tfidf": tfidf, "svd": svd, "kmeans": kmeans},
    }


def run_notes_supervised(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "weak_label",
    raw_source_field: str | None = "diagnosis",
    test_size: float = 0.2,
    cv_folds: int = 3,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train baseline and hybrid classifiers on clinical text using weak labels.

    Leakage Prevention:
    - The raw source field used to derive labels is strictly excluded from inputs.
    - TF-IDF and SVD components are fitted on training rows ONLY.
    - Stratified held-out test split (evaluated once).
    - Training-only cross-validation is reported separately.
    """
    valid = df[df[text_col].notna() & df[label_col].notna()].copy()
    if len(valid) < 10:
        raise ValueError("At least 10 labeled text records required for supervised training.")

    # Leakage guard: ensure source label field is not the text input
    if raw_source_field and raw_source_field == text_col:
        raise ValueError(
            f"Data leakage guard violation: input feature column '{text_col}' cannot be the label source field."
        )

    X_texts = valid[text_col].astype(str).tolist()
    y = valid[label_col].astype(int).values

    # Check class balance
    class_counts = pd.Series(y).value_counts().to_dict()
    if len(class_counts) < 2:
        raise ValueError("Supervised classification requires at least two classes (binary).")

    # Stratified held-out split
    X_train, X_test, y_train, y_test = train_test_split(
        X_texts, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Small sample warning flag
    small_sample_warning = len(valid) < 200

    # 1. Baseline Pipeline: TF-IDF -> Logistic Regression (fitted on train only)
    min_df = 1 if len(X_train) < 30 else 2
    baseline_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=300, min_df=min_df, stop_words="english")),
        ("clf", LogisticRegression(random_state=random_state, max_iter=200)),
    ])
    baseline_pipe.fit(X_train, y_train)

    y_pred_base = baseline_pipe.predict(X_test)
    y_prob_base = baseline_pipe.predict_proba(X_test)[:, 1] if hasattr(baseline_pipe, "predict_proba") else y_pred_base

    # 2. Hybrid Pipeline: TF-IDF -> TruncatedSVD (unsupervised features) -> Classifier (fitted on train only)
    n_comp = min(10, len(X_train) - 1, 5)
    hybrid_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=300, min_df=min_df, stop_words="english")),
        ("svd", TruncatedSVD(n_components=n_comp, random_state=random_state)),
        ("clf", LogisticRegression(random_state=random_state, max_iter=200)),
    ])
    hybrid_pipe.fit(X_train, y_train)

    y_pred_hyb = hybrid_pipe.predict(X_test)
    y_prob_hyb = hybrid_pipe.predict_proba(X_test)[:, 1] if hasattr(hybrid_pipe, "predict_proba") else y_pred_hyb

    def compute_metrics(y_true, y_pred, y_prob):
        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auc = 0.5
        cm = confusion_matrix(y_true, y_pred).tolist()
        return {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "roc_auc": round(auc, 4),
            "confusion_matrix": cm,
        }

    test_metrics_base = compute_metrics(y_test, y_pred_base, y_prob_base)
    test_metrics_hyb = compute_metrics(y_test, y_pred_hyb, y_prob_hyb)

    # Cross-validation on training rows only
    cv_folds_safe = min(cv_folds, min(pd.Series(y_train).value_counts()))
    cv_scores = []
    if cv_folds_safe >= 2:
        skf = StratifiedKFold(n_splits=cv_folds_safe, shuffle=True, random_state=random_state)
        for tr_idx, val_idx in skf.split(X_train, y_train):
            fold_X_tr = [X_train[i] for i in tr_idx]
            fold_y_tr = y_train[tr_idx]
            fold_X_va = [X_train[i] for i in val_idx]
            fold_y_va = y_train[val_idx]
            fold_pipe = Pipeline([
                ("tfidf", TfidfVectorizer(max_features=300, min_df=min_df, stop_words="english")),
                ("clf", LogisticRegression(random_state=random_state, max_iter=200)),
            ])
            fold_pipe.fit(fold_X_tr, fold_y_tr)
            preds = fold_pipe.predict(fold_X_va)
            cv_scores.append(float(accuracy_score(fold_y_va, preds)))

    return {
        "n_total": len(valid),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "class_distribution": class_counts,
        "small_sample_warning": small_sample_warning,
        "weak_label_disclaimer": "Metrics reflect performance against a derived weak label, NOT verified clinical ground truth.",
        "held_out_test_metrics": {
            "baseline_tfidf": test_metrics_base,
            "hybrid_svd": test_metrics_hyb,
        },
        "train_cross_validation_accuracy": {
            "folds": cv_folds_safe,
            "mean": round(float(np.mean(cv_scores)), 4) if cv_scores else 0.0,
            "std": round(float(np.std(cv_scores)), 4) if cv_scores else 0.0,
        },
        "models": {
            "baseline": baseline_pipe,
            "hybrid": hybrid_pipe,
        },
    }


def find_notes_files(paths: Paths) -> list[Path]:
    """Locate clinical notes parquet or csv files in data/raw/notes/."""
    candidates = []
    if paths.raw_notes.exists():
        candidates.extend(sorted(paths.raw_notes.glob("*.parquet")))
        candidates.extend(sorted(paths.raw_notes.glob("*.csv")))
    return candidates
