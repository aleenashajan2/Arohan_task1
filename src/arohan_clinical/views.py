"""The six AROHAN pages. Every number shown is computed from the loaded dataset or saved model outputs."""
from __future__ import annotations

import platform
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sklearn
import streamlit as st

from . import acquisition, config, labs, notes, reports, ui
from .clinical_coverage import coverage_table, implementation_status, scan_modality_status, scan_raw_folder
from .data_loader import (DatasetError, Roles, build_dataset_info, detect_roles, find_dataset_file, load_dataset,
                          write_dataset_info)
from .data_quality import build_quality_report, correlation_matrix
from .evaluation import confusion_counts, metrics_table, pr_points, roc_points
from .supervised import is_stale, load_model, load_results, predict_one, run_ablation, run_training, validate_input
from .unsupervised import run_analysis, save_outputs

CITE = "Experimental model output — not a clinical diagnosis."


@dataclass
class Context:
    paths: config.Paths
    csv: Path
    df: pd.DataFrame
    roles: Roles
    info: dict


@st.cache_data(show_spinner="Loading dataset…")
def _load_cached(root: str, csv_path: str, mtime: float, size: int):
    paths = config.Paths(Path(root))
    path = Path(csv_path)
    df = load_dataset(path)
    roles = detect_roles(df)
    info = write_dataset_info(paths, build_dataset_info(path, df, roles)) if paths.reports.exists() else build_dataset_info(path, df, roles)
    return df, roles, info


@st.cache_data(show_spinner="Running K-Means and PCA…")
def _unsup_cached(sha: str, k: int | None, df: pd.DataFrame, roles: Roles):
    return run_analysis(df, roles, k=k)


@st.cache_resource(show_spinner=False)
def _model_cached(path: str, mtime: float):
    import joblib
    return joblib.load(path)


def _locate():
    paths = config.default_paths()
    return paths, find_dataset_file(paths.raw)


def get_context() -> Context | None:
    paths, csv = _locate()
    if csv is None:
        return None
    try:
        st_ = csv.stat()
        df, roles, info = _load_cached(str(paths.root), str(csv), st_.st_mtime, st_.st_size)
    except DatasetError as exc:
        ui.notice(str(exc), "error", "Dataset could not be loaded")
        return None
    return Context(paths, csv, df, roles, info)


def require_context() -> Context | None:
    ctx = get_context()
    if ctx is None:
        ui.empty_state(
            "Dataset not found",
            "Demo data is not available in this deployment. See the project report for results."
        )
    return ctx


def _status_text() -> str:
    paths, csv = _locate()
    status = scan_modality_status(paths)
    parts = []
    if csv:
        parts.append("Structured: Ready")
    if status["labs"]["available"]:
        parts.append("Labs: Active")
    if status["notes"]["available"]:
        parts.append("Notes: Active")
    return " | ".join(parts) if parts else "Pending"


def _results(ctx: Context):
    res = load_results(ctx.paths)
    if res is None:
        return None, False
    return res, is_stale(res, ctx.info["sha256"])


def _fmt(df: pd.DataFrame, cols=None, digits=3):
    cols = cols or [c for c in df.columns if pd.api.types.is_float_dtype(df[c])]
    return df.style.format({c: f"{{:.{digits}f}}" for c in cols}, na_rep="n/a")


# ----------------------------------------------------------------------------------- Overview
def render_overview() -> None:
    ui.page_header("Overview", "Cardiovascular disease detection and analysis dashboard")
    ctx = require_context()
    if ctx is None:
        return
    df, roles = ctx.df, ctx.roles
    q = build_quality_report(df, roles)
    res, stale = _results(ctx)
    n_models = len(res["models"]) if res else 0
    n_hyb = sum(1 for m in res["models"].values() if m.get("family") == "hybrid") if res else 0

    ui.metric_row([
        {"label": "Records", "value": f"{len(df):,}", "hint": ctx.csv.name},
        {"label": "Columns", "value": df.shape[1], "hint": f"{len(roles.feature_cols)} feature columns"},
        {"label": "Models trained", "value": n_models, "hint": f"{n_models - n_hyb} baseline + {n_hyb} hybrid" if res and not stale else ("stale: retrain" if stale else "not yet trained"),
         "tone": "success" if n_models and not stale else "warning"},
        {"label": "Data quality status", "value": q["status"], "hint": f"{len(q['issues'])} issues, {len(q['warnings'])} warnings",
         "tone": ui.tone_for(q["status"])},
    ])
    st.write("")

    c1, c2 = st.columns(2)
    with c1:
        td = q["target_distribution"]
        if len(td):
            fig = px.bar(td, x="class", y="count", text="count", labels={"class": f"{roles.target_col} (class)"})
            fig.update_traces(marker_color=config.COLORS["primary"])
            ui.plot(fig, "ov_target", title="Target class distribution")
        else:
            ui.notice("No target column detected.", "error")
    with c2:
        feats = q["numeric"] + q["categorical"]
        pick = st.selectbox("Feature distribution", feats, key="ov_feat")
        if pick in q["numeric"]:
            fig = px.histogram(df, x=pick, nbins=30)
        else:
            vc = df[pick].astype(str).value_counts().sort_index().rename_axis(pick).reset_index(name="count")
            fig = px.bar(vc, x=pick, y="count")
        fig.update_traces(marker_color=config.COLORS["secondary"])
        ui.plot(fig, "ov_dist", title=f"Distribution: {pick}")

    st.markdown("### Model evaluation summary")
    if res and not stale:
        mt = metrics_table(res)
        long = mt.melt(id_vars="Model", value_vars=["Accuracy", "Recall (sensitivity)", "F1", "ROC-AUC"], var_name="Metric", value_name="Score")
        fig = px.bar(long, x="Metric", y="Score", color="Model", barmode="group")
        fig.update_yaxes(range=[0, 1.05])
        ui.plot(fig, "ov_models", title=f"Held-out test performance (n = {res['split']['n_test']})")
    elif stale:
        ui.notice("Saved results were produced from a different dataset file. Retrain on the Supervised Learning page.", "warning")
    else:
        ui.notice("No evaluation results yet. Train the baseline models on the Supervised Learning page.", "warning")


# ----------------------------------------------------------------------------------- Dataset
def render_dataset() -> None:
    ui.page_header("Dataset and Data Quality", "Inspect the downloaded file, validate it, and review data-quality findings")
    ctx = require_context()
    if ctx is None:
        return
    df, roles, info = ctx.df, ctx.roles, ctx.info
    q = build_quality_report(df, roles)
    ui.metric_row([
        {"label": "Records", "value": f"{len(df):,}"}, {"label": "Features", "value": len(roles.feature_cols),
         "hint": f"ID: {roles.id_col or 'none'} · target: {roles.target_col or 'not found'}"},
        {"label": "Missing cells", "value": int(q["missing"]["missing"].sum())},
        {"label": "Duplicate rows (excl. ID)", "value": q["duplicates"]["duplicates_excluding_id"]},
        {"label": "Quality status", "value": q["status"], "tone": ui.tone_for(q["status"])},
    ])
    if info.get("missing_expected_columns"):
        ui.notice("Expected columns not found in the file: " + ", ".join(info["missing_expected_columns"]) +
                  ". The app still works with the columns that exist; check the dictionary on the Schema tab.", "warning")
    tabs = st.tabs(["Preview", "Schema and dictionary", "Quality checks", "Distributions", "Correlation", "Preprocessing"])

    with tabs[0]:
        shown = df.drop(columns=[roles.id_col]) if roles.id_col else df
        if roles.id_col:
            st.caption(f"Identifier column '{roles.id_col}' is hidden to avoid exposing patient identifiers.")
        c1, c2, c3 = st.columns([2, 2, 2])
        text = c1.text_input("Search (any column)", key="ds_search")
        cols = c2.multiselect("Columns", list(shown.columns), default=list(shown.columns), key="ds_cols")
        tvals = sorted(shown[roles.target_col].astype(str).unique()) if roles.target_col else []
        tsel = c3.multiselect("Target class", tvals, default=tvals, key="ds_target") if tvals else []
        view = shown
        if text:
            mask = view.astype(str).apply(lambda s: s.str.contains(text, case=False, regex=False)).any(axis=1)
            view = view[mask]
        if tvals:
            view = view[view[roles.target_col].astype(str).isin(tsel)]
        if len(view) == 0:
            ui.empty_state("No rows match", "Clear the search box or widen the filters.")
        else:
            st.dataframe(view[cols] if cols else view, height=380, width="stretch")
        st.caption(f"{len(view):,} of {len(df):,} rows shown")

    with tabs[1]:
        rows = []
        for c in df.columns:
            spec = config.spec_for(c)
            role = "identifier" if c == roles.id_col else "target" if c == roles.target_col else \
                   ("numeric" if c in q["numeric"] else "categorical")
            rows.append({"Column": c, "Dtype": str(df[c].dtype), "Role": role, "Unique": int(df[c].nunique(dropna=True)),
                         "Meaning": spec["description"] if spec else "Unverified",
                         "Unit": spec["unit"] if spec else "", "Verification": "Verified" if spec else "Unknown"})
        st.dataframe(pd.DataFrame(rows), height=420, width="stretch", hide_index=True)
        st.markdown("**File record**")
        st.json({k: info[k] for k in ("filename", "size_bytes", "sha256", "source_url", "doi", "version", "licence",
                                      "file_modified_utc", "first_recorded_utc") if k in info})

    with tabs[2]:
        d = q["duplicates"]
        for m in q["issues"]:
            ui.notice(m, "error")
        for m in q["warnings"]:
            ui.notice(m, "warning")
        if not q["issues"] and not q["warnings"]:
            ui.notice("All automated checks passed.", "success")
        a, b = st.columns(2)
        a.markdown("**Missing values per column**")
        a.dataframe(q["missing"], width="stretch", hide_index=True)
        b.markdown("**Duplicate checks**")
        b.dataframe(pd.DataFrame({"check": ["Fully duplicated rows", "Duplicates ignoring ID", "Duplicate patient IDs"],
                                  "count": [d["full_row_duplicates"], d["duplicates_excluding_id"],
                                            d["duplicate_ids"] if d["duplicate_ids"] is not None else "n/a (no ID column)"]}),
                     width="stretch", hide_index=True)
        st.markdown("**Unexpected category codes**")
        if len(q["unexpected"]):
            st.dataframe(q["unexpected"], width="stretch", hide_index=True)
        else:
            st.caption("None found.")
        st.markdown("**Numeric range checks**")
        st.dataframe(_fmt(q["ranges"], ["observed_min", "observed_max"], 2), width="stretch", hide_index=True)
        c, e = st.columns(2)
        c.markdown("**Zero values in continuous columns**")
        c.dataframe(q["zeros"], width="stretch", hide_index=True)
        e.markdown("**Outliers (1.5 × IQR rule)**")
        e.dataframe(_fmt(q["outliers"], None, 2), width="stretch", hide_index=True)

    with tabs[3]:
        by_target = st.checkbox("Colour by target class", key="ds_bytarget", disabled=roles.target_col is None)
        cA, cB = st.columns(2)
        with cA:
            if q["numeric"]:
                col = st.selectbox("Numeric feature", q["numeric"], key="ds_num")
                fig = px.histogram(df, x=col, nbins=30, color=df[roles.target_col].astype(str) if by_target else None,
                                   barmode="overlay", opacity=0.75)
                if not by_target:
                    fig.update_traces(marker_color=config.COLORS["primary"])
                ui.plot(fig, "ds_hist", title=f"{col} distribution")
        with cB:
            if q["categorical"]:
                col = st.selectbox("Categorical feature", q["categorical"], key="ds_cat")
                tmp = df[[col] + ([roles.target_col] if by_target else [])].astype(str)
                fig = px.histogram(tmp, x=col, color=roles.target_col if by_target else None, barmode="group")
                if not by_target:
                    fig.update_traces(marker_color=config.COLORS["secondary"])
                ui.plot(fig, "ds_catbar", title=f"{col} distribution")
        st.markdown("**Target distribution**")
        st.dataframe(q["target_distribution"], width="stretch", hide_index=True)

    with tabs[4]:
        cols = [c for c in df.columns if c != roles.id_col]
        corr = correlation_matrix(df, cols)
        fig = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r", aspect="auto")
        ui.plot(fig, "ds_corr", height=560, title="Pearson correlation")

    with tabs[5]:
        st.dataframe(pd.DataFrame([
            {"Step": "Target column", "Configuration": str(roles.target_col)},
            {"Step": "Excluded identifier", "Configuration": str(roles.id_col or "None")},
            {"Step": "Numeric features", "Configuration": f"{len(q['numeric'])} cols (median impute, standard scale)"},
            {"Step": "Categorical features", "Configuration": f"{len(q['categorical'])} cols (mode impute, one-hot encode)"},
            {"Step": "Pipeline scope", "Configuration": "Fitted strictly on training split rows only"},
            {"Step": "Zero values", "Configuration": "Preserved as recorded numerical measurements"},
        ]), width="stretch", hide_index=True)


# ----------------------------------------------------------------------------------- Supervised
def _train_button(ctx: Context, label: str) -> None:
    if st.button(label, type="primary", key="train_btn"):
        with st.spinner("Training Logistic Regression, Decision Tree and Random Forest…"):
            try:
                run_training(ctx.paths, ctx.df, ctx.roles)
            except (DatasetError, ValueError) as exc:
                ui.notice(str(exc), "error", "Training failed")
                return
        st.rerun()


def render_supervised() -> None:
    ui.page_header("Supervised Learning", "Baseline classifiers evaluated on a held-out test split")
    ctx = require_context()
    if ctx is None:
        return
    if ctx.roles.target_col is None:
        ui.notice("No target column was detected. Columns: " + ", ".join(ctx.df.columns) +
                  ". Set the AROHAN_TARGET environment variable to the label column and restart.", "error", "Target label not found")
        return
    res, stale = _results(ctx)
    if res is None:
        ui.empty_state("Models have not been trained yet",
                       f"Training fits three baselines on a stratified {int((1 - config.TEST_SIZE) * 100)}/{int(config.TEST_SIZE * 100)} "
                       f"split (seed {config.RANDOM_STATE}) for target <code>{ctx.roles.target_col}</code>. Nothing is shown until real results exist.")
        _train_button(ctx, "Train models")
        return
    if stale:
        ui.notice("These results come from a different dataset file than the one currently in data/raw. Retrain before using them.",
                  "warning", "Stale results")
    sp = res["split"]
    ui.metric_row([
        {"label": "Train / test rows", "value": f"{sp['n_train']} / {sp['n_test']}", "hint": f"seed {sp['seed']}, stratified: {sp['stratified']}"},
        {"label": "Target", "value": res["target_column"], "hint": f"positive class = {res['positive_class']}"},
        {"label": "Features used", "value": len(res["features"]), "hint": f"ID excluded: {res['id_column_excluded']}"},
        {"label": "Test positives", "value": f"{sp['test_positive']} / {sp['n_test']}"},
    ])
    _train_button(ctx, "Retrain models")
    if not any(m.get("family") == "hybrid" for m in res["models"].values()):
        ui.notice("These saved results contain no hybrid models. Click 'Retrain models' to add them.", "warning")
    else:
        st.caption(res.get("hybrid_description", ""))
    tabs = st.tabs(["Test results", "Curves and confusion matrix", "Cross-validation", "Leakage screen", "Feature ablation",
                    "Experimental prediction"])
    mt = metrics_table(res)

    with tabs[0]:
        st.dataframe(_fmt(mt), width="stretch", hide_index=True)
        long = mt.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall (sensitivity)", "Specificity", "F1", "ROC-AUC"],
                       var_name="Metric", value_name="Score")
        fig = px.bar(long, x="Metric", y="Score", color="Model", barmode="group")
        fig.update_yaxes(range=[0, 1.05])
        ui.plot(fig, "sup_bar", height=380, title=f"Held-out test metrics (n = {sp['n_test']}, single split)")

    y_test = np.array(res["y_test"])
    with tabs[1]:
        a, b = st.columns(2)
        fig = go.Figure()
        fig2 = go.Figure()
        for name, m in res["models"].items():
            fpr, tpr = roc_points(y_test, m["y_score"])
            rec, prec = pr_points(y_test, m["y_score"])
            fig.add_scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC {m['test_metrics']['roc_auc']:.3f})")
            fig2.add_scatter(x=rec, y=prec, mode="lines", name=f"{name} (AP {m['test_metrics']['pr_auc']:.3f})")
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash", color=config.COLORS["muted"]))
        fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate")
        fig2.update_layout(xaxis_title="Recall", yaxis_title="Precision")
        with a:
            ui.plot(fig, "sup_roc", title="ROC curves (test set)")
        with b:
            ui.plot(fig2, "sup_pr", title="Precision-recall curves (test set)")
        pick = st.selectbox("Confusion matrix for", list(res["models"]), key="sup_cm")
        cm = confusion_counts(y_test, res["models"][pick]["y_pred"])
        labels = [f"{c}" for c in res["classes"]]
        fig = px.imshow(cm, text_auto=True, x=[f"Pred {l}" for l in labels], y=[f"True {l}" for l in labels],
                        color_continuous_scale=[[0, config.COLORS["card"]], [1, config.COLORS["primary"]]])
        ui.plot(fig, "sup_cmfig", height=340, title=f"Confusion matrix: {pick}")

    with tabs[2]:
        rows = []
        for name, m in res["models"].items():
            r = {"Model": name}
            for k, lab in (("accuracy", "Accuracy"), ("recall", "Recall"), ("f1", "F1"), ("roc_auc", "ROC-AUC")):
                r[lab] = f"{m['cv'][k]['mean']:.3f} ± {m['cv'][k]['std']:.3f}"
            rows.append(r)
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption(f"{res['cv_folds']}-fold stratified cross-validation on training rows (mean ± sd).")

    with tabs[3]:
        ls = pd.DataFrame(res["leakage_screen"])
        fig = px.bar(ls.sort_values("univariate_auc"), x="univariate_auc", y="feature", orientation="h", color="flagged",
                     color_discrete_map={True: config.COLORS["error"], False: config.COLORS["primary"]})
        fig.update_xaxes(range=[0.5, 1.0], title="Univariate AUC (training rows, in-sample)")
        ui.plot(fig, "sup_leak", height=420, title="Single-feature association with the target")

    with tabs[4]:
        flagged = [r["feature"] for r in res["leakage_screen"] if r["flagged"]]
        pick = st.multiselect("Features to remove", res["features"], default=flagged[:1], key="abl_pick")
        if st.button("Run comparison", key="abl_btn", disabled=not pick):
            with st.spinner("Training models with and without the selected features…"):
                try:
                    st.session_state["abl"] = (tuple(pick), run_ablation(ctx.df, ctx.roles, list(pick), sp["seed"], sp["test_size"]))
                except (ValueError, DatasetError) as exc:
                    ui.notice(str(exc), "error", "Comparison failed")
        saved = st.session_state.get("abl")
        if saved:
            dropped, abl = saved
            st.markdown(f"**Showing:** all features vs without {', '.join(dropped)}")
            full = abl[abl["Variant"] == "All features"].set_index("Model")
            wo = abl[abl["Variant"] != "All features"].set_index("Model")
            cmp_df = pd.DataFrame({"Type": full["Type"]})
            for met in ("ROC-AUC", "Recall", "Accuracy"):
                cmp_df[f"{met} (all)"] = full[met]
                cmp_df[f"{met} (without)"] = wo[met]
                cmp_df[f"{met} change"] = wo[met] - full[met]
            st.dataframe(_fmt(cmp_df.reset_index()), width="stretch", hide_index=True)
            fig = px.bar(abl, x="Model", y="ROC-AUC", color="Variant", barmode="group")
            fig.update_yaxes(range=[0, 1.05])
            ui.plot(fig, "abl_fig", title="ROC-AUC with and without the selected features (same test rows)")

    with tabs[5]:
        ui.notice(CITE, "warning")
        if stale:
            st.caption("Prediction is disabled while results are stale.")
        else:
            spec = res["input_spec"]
            name = st.selectbox("Model", list(res["models"]), key="pred_model")
            with st.form("pred_form"):
                cols = st.columns(3)
                values = {}
                for i, (c, s) in enumerate(spec.items()):
                    with cols[i % 3]:
                        if s["kind"] == "numeric":
                            unit = f" ({s['unit']})" if s["unit"] else ""
                            values[c] = st.number_input(f"{s['label']}{unit}", value=float(s["default"]), key=f"in_{c}")
                        else:
                            values[c] = st.selectbox(s["label"], s["options"], key=f"in_{c}")
                go_ = st.form_submit_button("Run model", type="primary")
            if go_:
                errors, warns = validate_input(spec, values)
                for e in errors:
                    ui.notice(e, "error")
                for w in warns:
                    ui.notice(w, "warning")
                if not errors:
                    mp = ctx.paths.models / res["models"][name]["model_file"]
                    out = predict_one(_model_cached(str(mp), mp.stat().st_mtime), res, values)
                    raw_cls = out["predicted_class"]
                    pred_label = "Healthy" if str(raw_cls) in ("0", "Healthy") else "Cardiovascular disease (type not specified in the source data)"
                    ui.metric_row([{"label": "Predicted class", "value": pred_label, "hint": f"source class code = {raw_cls}"},
                                   {"label": "Model score (uncalibrated)", "value": f"{out['score']:.3f}", "hint": "not a calibrated probability"}])
                    ui.notice(CITE, "warning")


# ----------------------------------------------------------------------------------- Unsupervised
def render_unsupervised() -> None:
    ui.page_header("Unsupervised Learning", "Exploratory K-Means clustering and PCA on continuous clinical measurements")
    ctx = require_context()
    if ctx is None:
        return
    try:
        base = _unsup_cached(ctx.info["sha256"], None, ctx.df, ctx.roles)
    except ValueError as exc:
        ui.notice(str(exc), "error", "Unsupervised analysis unavailable")
        return
    ks = base["scores"]["k"].tolist()
    k = st.selectbox("Number of clusters (k)", ks, index=ks.index(base["best_k"]), key="un_k",
                     help="Default = highest silhouette score. You can override it.")
    res = base if k == base["best_k"] else _unsup_cached(ctx.info["sha256"], int(k), ctx.df, ctx.roles)
    sil = float(base["scores"].loc[base["scores"]["k"] == k, "silhouette"].iloc[0])
    ui.metric_row([
        {"label": "Features clustered", "value": len(res["features"])}, {"label": "Clusters (k)", "value": k, "hint": f"best silhouette at k = {base['best_k']}"},
        {"label": "Silhouette", "value": f"{sil:.3f}", "hint": "cluster separation, −1 to 1"},
        {"label": "PCA PC1 + PC2", "value": f"{res['explained']['explained_variance'].head(2).sum() * 100:.1f}%", "hint": "variance explained"},
    ])
    st.caption(f"Features: {', '.join(res['features'])} · Scaler: {res['settings']['scaler']} · Seed: {res['settings']['seed']}")
    a, b = st.columns(2)
    with a:
        fig = px.line(base["scores"], x="k", y="silhouette", markers=True)
        ui.plot(fig, "un_sil", title="Silhouette score by k")
    with b:
        fig = px.bar(res["sizes"], x="cluster", y="rows", text="rows")
        fig.update_traces(marker_color=config.COLORS["secondary"])
        ui.plot(fig, "un_sizes", title="Cluster sizes")
    c, d = st.columns([3, 2])
    with c:
        colour = st.radio("Colour PCA points by", ["Cluster", "Target label (inspection only)"], horizontal=True, key="un_col",
                          disabled="target" not in res["pca"])
        key = "cluster" if colour == "Cluster" or "target" not in res["pca"] else "target"
        fig = px.scatter(res["pca"], x="PC1", y="PC2", color=key, opacity=0.8)
        ui.plot(fig, "un_pca", height=420, title="PCA projection (first two components)")
    with d:
        ev = res["explained"]
        fig = go.Figure()
        fig.add_bar(x=ev["component"], y=ev["explained_variance"], name="Explained", marker_color=config.COLORS["primary"])
        fig.add_scatter(x=ev["component"], y=ev["cumulative"], name="Cumulative", mode="lines+markers")
        fig.update_yaxes(tickformat=".0%")
        ui.plot(fig, "un_ev", height=420, title="PCA explained variance")
    st.markdown("**Cluster feature summary (mean in original units)**")
    st.dataframe(res["summary"], width="stretch", hide_index=True)
    if res["crosstab"] is not None:
        with st.expander("Cross-tabulation with the target label"):
            st.dataframe(res["crosstab"], width="stretch")
    if st.button("Save analysis tables to outputs/tables", key="un_save"):
        names = save_outputs(ctx.paths, res)
        st.success("Saved: " + ", ".join(names))


# ----------------------------------------------------------------------------------- Coverage
def render_coverage() -> None:
    ui.page_header("Clinical Data Coverage", "Which clinical data types are actually available and processed")
    paths, csv = _locate()
    ctx = get_context() if csv else None
    extra = scan_raw_folder(paths)
    cov = coverage_table(ctx is not None, ctx.info if ctx else None, extra, paths=paths)
    
    is_labs = cov.loc[cov["Clinical data type"] == "Laboratory reports", "Available"].iloc[0] == "Yes"
    if is_labs:
        ui.notice("Structured clinical measurements and NidaanKosha laboratory reports are active as independent single modalities. "
                  "Clinical notes, imaging and discharge summaries are pending or unavailable. Multimodal fusion is NOT performed.",
                  "warning", "Coverage summary")
    else:
        ui.notice("The supervisor's brief covers blood reports, laboratory reports, imaging reports, clinical notes and discharge summaries. "
                  "Currently only structured clinical measurements are available.", "warning", "Coverage summary")
    ui.status_table(cov, ["Available"])
    if extra:
        st.markdown("**Non-CSV files found in data/raw (listed only; not processed):** " + ", ".join(extra))

    st.markdown("### Multimodal Fusion Readiness Matrix")
    fusion_df = reports.get_fusion_readiness_table(paths)
    st.dataframe(fusion_df, width="stretch", hide_index=True)
    ui.notice("Late fusion not performed: no shared patient identifier across datasets. "
              "Do not merge independent clinical cohorts or imply shared patients without a verified, reliable patient ID.",
              "warning", "Fusion Feasibility")

    with st.expander("Team Export Schema (Standardized format for multimodal fusion integration)"):
        st.write("For downstream integration into the multimodal fusion pipeline, each module outputs records conforming to this documented schema:")
        schema_df = pd.DataFrame([{"Field": k, "Specification": v} for k, v in reports.EXPORT_SCHEMA.items()])
        st.dataframe(schema_df, width="stretch", hide_index=True)

    st.markdown("""
**Rules applied on this page**
- A cholesterol or blood-sugar column is **not** a laboratory report; an ECG category code is **not** an imaging report.
- Nothing is marked processed unless a real file was obtained and a working pipeline ran on it.
- Datasets from different sources must not be combined as if they described the same patients. Multimodal claims need a reliable patient-level link.
- When adding data later, record its source URL, licence, population, format and modalities before using it.""")


# ----------------------------------------------------------------------------------- Methodology
def render_methodology() -> None:
    ui.page_header("Methodology and Limitations", "How the prototype works, and what it does not do")
    paths, csv = _locate()
    ctx = get_context() if csv else None
    res, stale = _results(ctx) if ctx else (None, False)
    st.markdown("### Objective & Research Context")
    st.write("Early detection and prediction of cardiovascular disease using Indian-population clinical data (AIKosh, Mendeley, Eka Care). "
             "The supervisor brief requires modular handling of structured measurements, blood/lab reports, imaging reports, clinical notes, "
             "and discharge summaries, building towards a multimodal fusion system.")
    
    st.markdown("### Honest Data Situation & Modality Separation")
    st.write("No public Indian dataset exists that connects labs, imaging, clinical notes and discharge summaries for the same patients. "
             "Therefore, each clinical modality is built as a separate, clearly labelled module. Multimodal late fusion is explicitly "
             "NOT performed because records are not linked by a genuine patient identifier.")

    st.markdown("### Datasets Registered")
    st.write("- **Structured Measurements:** Mendeley Cardiovascular Disease Dataset (DOI 10.17632/dzz48mvjht.1, CC BY 4.0), 1,000 subjects from an Indian hospital.")
    st.write("- **Laboratory Reports:** NidaanKosha-100k-V1.0 (Eka Care / IndiaAI, CC-BY-SA-4.0), 6.84 million investigation readings from 100,000 lab reports across India extracted via vision LLM.")
    st.write("- **Clinical Notes (Gated):** Eka Clinical Note Generation Dataset (transcriptions in English, Hindi, Marathi with structured entities, MIT).")

    st.markdown("### Diagnostic Honesty & Labeling Policies")
    st.write("- **Binary Output Wording:** Class outputs are strictly formulated as 'Healthy' or 'Cardiovascular disease (type not specified in the source data)' unless a dataset genuinely provides a specific disease diagnosis.")
    st.write("- **Rejection of Circular Labels:** Creating pseudo-labels from lab reference thresholds (e.g. cholesterol > 200) and then training a model to predict them is circular. Reference ranges are used strictly for descriptive profiling.")
    st.write("- **Weak Labels & Leakage Prevention:** Where labels are derived from clinical text mentions, the source label field is strictly excluded from feature inputs, stratified splits are enforced, and small-sample warnings are flagged.")

    st.markdown("### Supervised and Hybrid Modeling")
    if res:
        sp = res["split"]
        st.write(f"Logistic Regression, Decision Tree (max depth 5), Random Forest (300 trees), fixed untuned hyper-parameters, each as a "
                 f"baseline and as a hybrid (the same classifier plus K-Means distance features fitted on training rows only; structured data only). Stratified split "
                 f"{int((1 - sp['test_size']) * 100)}/{int(sp['test_size'] * 100)}, seed {sp['seed']}, {sp['n_train']} train / {sp['n_test']} test rows. "
                 f"Test metrics come from held-out predictions; {res['cv_folds']}-fold CV on training rows is reported separately. Trained {res['trained_utc']}.")
    else:
        st.write("Logistic Regression, Decision Tree, Random Forest on a stratified train/test split. Not yet trained.")

    st.markdown("### Unsupervised Learning")
    st.write("Standard scaling, K-Means (k chosen by silhouette over 2-8 unless overridden) and PCA on continuous measurements and lab reports. "
             "Clusters represent exploratory mathematical groupings and are NOT clinical diagnoses.")

    st.markdown("### Reproducibility")
    st.code(f"python {sys.version.split()[0]} · scikit-learn {sklearn.__version__} · pandas {pd.__version__} · numpy {np.__version__} · {platform.system()}\n"
            f"random_state = {config.RANDOM_STATE}", language="text")

    st.markdown("### Limitations")
    for t in [
        "No shared patient identifier connects the structured data, lab reports, and clinical notes: cross-dataset fusion is impossible without record linkage.",
        "NidaanKosha lab values were extracted by vision LLMs from heterogeneous lab reports and carry inherent OCR/extraction noise.",
        "Clinical note datasets are small (N < 200) and require user authentication/terms acceptance.",
        "Echocardiography reports (InfoBayAI) origin is not confirmed as Indian and requires research verification.",
        "Parrotlet-V-lite-4B vision model was not executed locally due to hardware constraints; external hosted APIs were not used.",
        "High evaluation scores on cross-sectional datasets do not imply clinical validity or diagnostic safety.",
        "Model outputs are uncalibrated experimental scores. This is a research prototype; no diagnostic advice is provided.",
    ]:
        st.markdown(f"- {t}")


@st.cache_data(show_spinner="Loading NidaanKosha lab investigation sample…")
def _load_labs_sample_cached(parquet_path_str: str, max_rows: int = 40000):
    import pyarrow.parquet as pq
    table = pq.read_table(parquet_path_str)
    if table.num_rows > max_rows:
        table = table.slice(0, max_rows)
    df = table.to_pandas()
    cleaned, stats = labs.clean_and_normalize_lab_readings(df)
    matrix = labs.build_lab_feature_matrix(cleaned, min_tests_per_doc=2, max_docs=1500)
    discovery = labs.discover_lab_tests(Path(parquet_path_str), max_rows=max_rows)
    return cleaned, stats, matrix, discovery


@st.cache_data(show_spinner="Running lab unsupervised clustering…")
def _run_labs_unsup_cached(matrix_len: int, matrix_cols: str, chosen_k: int | None, matrix_df: pd.DataFrame):
    return labs.run_lab_unsupervised(matrix_df, n_clusters=chosen_k)


@st.cache_data(show_spinner="Loading clinical notes dataset…")
def _load_notes_data_cached(notes_file_str: str):
    p = Path(notes_file_str)
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    text_col = "text" if "text" in df.columns else ("transcript" if "transcript" in df.columns else df.columns[0])
    mentions_list = [notes.extract_cardiac_mentions(t) for t in df[text_col]]
    df["cardiac_mentions_count"] = [m["total_cardiac_mentions"] for m in mentions_list]
    df["has_cardiac_condition"] = [m["has_cardiac_condition"] for m in mentions_list]
    df["cardiac_terms"] = [
        ", ".join(
            m["mentions_by_category"]["symptoms"] +
            m["mentions_by_category"]["risk_factors"] +
            m["mentions_by_category"]["cardiac_conditions"] +
            m["mentions_by_category"]["medications"]
        ) or "None detected" for m in mentions_list
    ]
    unsup = notes.run_notes_unsupervised(df[text_col], n_topics=4, n_clusters=3)
    return df, text_col, unsup


@st.cache_data(show_spinner="Loading longitudinal patient history dataset…")
def _load_history_data_cached(history_file_str: str):
    p = Path(history_file_str)
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    def _extract_summary(val):
        if hasattr(val, "__iter__") and len(val) > 0:
            item = val[0]
            if isinstance(item, dict):
                return item.get("ideal_completion", "")
        return str(val) if val is not None else ""
    if "ideal_completions_data" in df.columns:
        df["trajectory_summary"] = df["ideal_completions_data"].apply(_extract_summary)
    else:
        df["trajectory_summary"] = ""
    return df


# ----------------------------------------------------------------------------------- Labs, Notes and Reports
def render_labs_notes_reports() -> None:
    ui.page_header("Labs, Notes and Reports", "Laboratory data analysis (NidaanKosha)")
    paths, _ = _locate()

    tabs = st.tabs([
        "Labs (NidaanKosha)",
        "Clinical Notes",
        "Patient History",
        "Report Images",
        "Echo Reports & Fusion",
    ])

    # TAB 1: Labs (NidaanKosha)
    with tabs[0]:
        st.markdown("### Laboratory Investigation Reports (NidaanKosha 100k)")
        lab_file, is_demo = labs.get_active_labs_source(paths)
        if not lab_file:
            ui.empty_state(
                "NidaanKosha dataset not found",
                "Demo data is not available in this deployment. See the project report for results."
            )
        else:
            cleaned, stats, matrix, discovery = _load_labs_sample_cached(str(lab_file), max_rows=40000)

            readings_val = f"Demo sample: {len(cleaned):,} of 6,844,304 readings" if is_demo else "6,844,304"
            readings_hint = "Sampled with fixed seed 42 (CC-BY-SA-4.0)" if is_demo else "Across 2 parquet files (V1.0)"

            ui.metric_row([
                {"label": "Total Readings", "value": readings_val, "hint": readings_hint, "tone": "warning" if is_demo else "primary"},
                {"label": "Lab reports", "value": f"{len(matrix):,}" if is_demo else "100,000", "hint": "Sampled reports" if is_demo else "Diverse diagnostic centers in India"},
                {"label": "Extraction Method", "value": "Vision LLM", "hint": "Parrotlet AI from reports & PDFs"},
                {"label": "Licence", "value": "CC-BY-SA-4.0", "hint": "Open research dataset", "tone": "success"},
                {"label": "Diagnostic Labels", "value": "None (Unlabeled)", "hint": "Unsupervised profiling only", "tone": "warning"},
            ])
            st.write("")

            col_a, col_b = st.columns([1, 1])
            with col_a:
                st.markdown("#### Data-Driven Test Discovery")
                st.caption("Top tests extracted across Indian diagnostic reports with common units and LOINC codes.")
                st.dataframe(discovery.head(12), width="stretch", hide_index=True)

            with col_b:
                st.markdown("#### Quality Screening & Safe Normalization")
                st.caption("Unit validation and physiological anomaly screening results on the profiled lab reports sample:")
                anomaly_items = [
                    {"label": "Cardiometabolic readings", "value": f"{stats['total_cardio_rows']:,}", "hint": "Lipids, glucose, creatinine, urea"},
                    {"label": "Non-numeric filtered", "value": stats['non_numeric_values'], "hint": "Vision LLM extraction noise", "tone": "warning"},
                    {"label": "Creatinine units converted", "value": stats['unit_conversions_applied'], "hint": "umol/L safely normalized to mg/dL", "tone": "success"},
                    {"label": "Anomalies filtered", "value": stats['anomalies_filtered'], "hint": "Values outside physiological limits", "tone": "warning"},
                ]
                ui.metric_row(anomaly_items)

            st.markdown("---")
            st.markdown("#### Unsupervised Cardiometabolic Phenotyping (K-Means & PCA)")
            st.caption("Per-report wide feature matrix built from cardiometabolic biomarkers.")

            if not matrix.empty:
                chosen_k = st.slider("Select cluster count k (exploratory grouping)", min_value=2, max_value=6, value=2, key="labs_k_slider")
                unsup_res = _run_labs_unsup_cached(len(matrix), ",".join(matrix.columns), chosen_k, matrix)

                p1, p2 = st.columns([1, 1])
                with p1:
                    pca_df = unsup_res["pca_df"]
                    fig = px.scatter(
                        pca_df, x="PC1", y="PC2", color="cluster",
                        opacity=0.8,
                        hover_data=["document_id"] if "document_id" in pca_df.columns else None
                    )
                    ui.plot(fig, "lab_pca", height=380, title="PCA Projection of Laboratory Reports (2 Components)")
                with p2:
                    st.markdown("**Biomarker Mean Profiles by Discovered Cluster (mg/dL)**")
                    st.dataframe(unsup_res["cluster_summaries"], width="stretch", hide_index=True)
                    sil_scores = unsup_res.get("silhouette_scores", {})
                    if sil_scores:
                        st.caption(f"Silhouette scores: {sil_scores}")

    # TAB 2: Clinical Notes
    with tabs[1]:
        st.markdown("### Clinical Notes & Medical Scribing (Eka Care)")
        note_files = notes.find_notes_files(paths)
        if not note_files:
            ui.empty_state(
                "Not available in this deployment",
                "Demo data is not available in this deployment. See the project report for results."
            )
        else:
            df_notes, text_col, unsup = _load_notes_data_cached(str(note_files[0]))
            total_cardio_count = int(df_notes["cardiac_mentions_count"].sum())
            cardio_docs = int((df_notes["cardiac_mentions_count"] > 0).sum())
            ui.metric_row([
                {"label": "Consultations", "value": f"{len(df_notes):,}", "hint": Path(note_files[0]).name},
                {"label": "Cardio Mentions Found", "value": total_cardio_count, "hint": f"In {cardio_docs} consultations", "tone": "success" if total_cardio_count else "warning"},
                {"label": "Topics Extracted", "value": unsup["n_topics"], "hint": "TF-IDF + TruncatedSVD"},
                {"label": "Source", "value": "Eka Care", "hint": "Clinical note generation dataset"},
                {"label": "Licence", "value": "MIT", "hint": "Open research dataset", "tone": "success"},
            ])
            st.write("")

            col_n1, col_n2 = st.columns([3, 2])
            with col_n1:
                st.markdown("#### Clinical Consultation Records & Extracted Cardiology Terms")
                st.caption("Structured extraction of cardiac symptoms, risk factors, conditions, and medications from notes:")
                display_cols = [c for c in ["session_id", text_col, "cardiac_terms", "cardiac_mentions_count"] if c in df_notes.columns]
                st.dataframe(df_notes[display_cols].head(30), width="stretch", hide_index=True)

            with col_n2:
                st.markdown("#### Unsupervised Clinical Topic Modeling")
                st.caption("Latent topics extracted via TF-IDF (1-2 ngrams) and Truncated SVD dimensionality reduction:")
                st.dataframe(unsup["topics"], width="stretch", hide_index=True)

        st.markdown("---")
        st.markdown("#### Interactive Clinical Note Analyzer (Rule-Based NLP & Weak Labeling)")
        st.caption("Demonstration of the transparent cardiology mention extractor and weak-label derivation algorithm on clinical consultations:")

        note_input = st.text_area(
            "Paste any note text to see the rule-based extraction. No data is stored or sent anywhere.",
            value="",
            height=120,
            key="note_text_in",
        )

        if note_input:
            mentions = notes.extract_cardiac_mentions(note_input)
            weak_lbl, weak_name = notes.derive_weak_cardiac_label({"text": note_input})

            m1, m2 = st.columns(2)
            with m1:
                st.markdown("**Identified Cardiology Mentions by Category**")
                cat_df = pd.DataFrame([
                    {"Category": cat.replace("_", " ").title(), "Extracted Terms": ", ".join(terms) if terms else "None detected"}
                    for cat, terms in mentions["mentions_by_category"].items()
                ])
                st.dataframe(cat_df, width="stretch", hide_index=True)
            with m2:
                st.markdown("**Derived Weak Classification & Caveats**")
                ui.metric_row([
                    {"label": "Weak Label", "value": weak_name, "hint": f"Binary code: {weak_lbl}", "tone": "warning" if weak_lbl else "success"},
                    {"label": "Total Mentions", "value": mentions["total_cardiac_mentions"], "hint": "Raw term frequency"},
                ])

    # TAB 3: Patient History
    with tabs[2]:
        st.markdown("### Longitudinal Patient History Summarisation")
        hist_files = list(paths.raw_history.glob("*")) if paths.raw_history.exists() else []
        hist_files = [f for f in hist_files if f.is_file() and not f.name.startswith(".")]
        if not hist_files:
            ui.empty_state(
                "Patient History dataset not found",
                "Demo data is not available in this deployment. See the project report for results."
            )
        else:
            df_hist = _load_history_data_cached(str(hist_files[0]))
            ui.metric_row([
                {"label": "Patient Journeys", "value": f"{len(df_hist):,}", "hint": Path(hist_files[0]).name},
                {"label": "Modality", "value": "Trajectories & Vitals", "hint": "Multi-visit medical history"},
                {"label": "Source", "value": "Eka Care", "hint": "Medical history summarisation dataset"},
                {"label": "Licence", "value": "MIT", "hint": "Open research dataset", "tone": "success"},
            ])
            st.write("")

            h_col1, h_col2 = st.columns([1, 2])
            with h_col1:
                st.markdown("#### Patient Journey Selector")
                journey_idx = st.selectbox(
                    "Select patient index",
                    options=list(range(len(df_hist))),
                    format_func=lambda i: f"Patient {i + 1} (Prompt: {str(df_hist.get('prompt_id', pd.Series(range(len(df_hist))))[i])[:12]}…)",
                    key="hist_journey_select"
                )
                st.caption("Longitudinal health history contains multi-timepoint visit notes, diagnoses, and lab results.")
            with h_col2:
                st.markdown("#### Longitudinal Medical Trajectory & Findings")
                summary_text = df_hist["trajectory_summary"].iloc[journey_idx]
                if summary_text:
                    st.markdown(summary_text, unsafe_allow_html=True)
                else:
                    st.caption("No trajectory summary text recorded for this entry.")

            st.write("")
            st.markdown("#### Complete Patient History Dataset Overview")
            show_cols = [c for c in ["prompt_id", "trajectory_summary"] if c in df_hist.columns]
            st.dataframe(df_hist[show_cols].head(25), width="stretch", hide_index=True)

    # TAB 4: Report Images
    with tabs[3]:
        st.markdown("### Medical Report Images & Vision AI")
        img_profile = reports.profile_report_images(paths)
        if not img_profile["available"]:
            ui.empty_state(
                "Report images not found",
                "Demo data is not available in this deployment. See the project report for results."
            )
        else:
            ui.metric_row([
                {"label": "Total Images", "value": img_profile["total_images"], "hint": "Lab reports & prescriptions"},
                {"label": "Total Storage", "value": f"{img_profile.get('total_size_mb', 0)} MB", "hint": "Raw storage"},
                {"label": "Average Size", "value": f"{img_profile.get('avg_size_kb', 0)} KB", "hint": "Per image"},
                {"label": "Licence", "value": "MIT", "hint": "Eka Care Medical Records Set", "tone": "success"},
            ])
            st.write("")

            col_img1, col_img2 = st.columns([1, 2])
            with col_img1:
                st.markdown("#### Document Categories & Distribution")
                st.dataframe(pd.DataFrame([{"Document Type": k, "Count": v} for k, v in img_profile["formats"].items()]), width="stretch", hide_index=True)
            with col_img2:
                st.markdown("#### Document Image Previews")
                samples = img_profile.get("sample_images", [])
                if samples:
                    scols = st.columns(min(3, len(samples)))
                    for i, s in enumerate(samples[:3]):
                        with scols[i]:
                            st.image(s["bytes"], caption=f"{s['type']} ({s['id']})", width=220)
                else:
                    st.caption("No binary previews available to render.")

        st.markdown("---")
        st.markdown("#### Vision Model Status")
        p_stat = reports.get_parrotlet_status()
        ui.metric_row([
            {"label": "Model", "value": p_stat["model_name"]},
            {"label": "Status", "value": p_stat["status"]},
            {"label": "Execution", "value": p_stat["compliance"]},
        ])

    # TAB 5: Echo Reports & Multimodal Fusion Readiness
    with tabs[4]:
        st.markdown("### Echocardiography Reports & Multimodal Fusion Readiness")
        echo_profile = reports.profile_echo_reports(paths)
        if not echo_profile["available"]:
            ui.empty_state(
                "Echocardiography reports not found",
                "Demo data is not available in this deployment. See the project report for results."
            )

        st.markdown("#### Validated Clinical Echocardiography Regex Extractor")
        st.caption("Demonstration of structured parameter extraction from echocardiography text reports:")
        sample_echo_txt = st.text_area(
            "Sample Echocardiography Report Text",
            value="Transthoracic echocardiogram reveals normal left ventricular size and systolic function. LVEF: 58%, LVEDD: 47 mm, and LA dimension of 36 mm. No regional wall motion abnormality.",
            height=100,
            key="echo_sample_in"
        )
        if sample_echo_txt:
            extracted = reports.extract_echo_measurements(sample_echo_txt)
            echo_rows = []
            for m_key, m_val in extracted.items():
                if m_val:
                    echo_rows.append({
                        "Parameter": m_val["label"],
                        "Extracted Value": f"{m_val['value']} {m_val['unit']}",
                        "Physiologically Plausible": "Yes (Within range)" if m_val["plausible"] else "No (Abnormal/Extreme)",
                    })
                else:
                    echo_rows.append({
                        "Parameter": reports.ECHO_MEASUREMENT_PATTERNS[m_key]["label"],
                        "Extracted Value": "Not detected in text",
                        "Physiologically Plausible": "-",
                    })
            st.dataframe(pd.DataFrame(echo_rows), width="stretch", hide_index=True)

        st.markdown("---")
        st.markdown("#### Multimodal Fusion Readiness Matrix")
        fusion_df = reports.get_fusion_readiness_table(paths)
        st.dataframe(fusion_df, width="stretch", hide_index=True)

        with st.expander("Team Export Schema (Standardized format for multimodal fusion integration)"):
            st.write("For downstream integration into the multimodal fusion pipeline, each module outputs records conforming to this documented schema:")
            schema_df = pd.DataFrame([{"Field": k, "Specification": v} for k, v in reports.EXPORT_SCHEMA.items()])
            st.dataframe(schema_df, width="stretch", hide_index=True)
