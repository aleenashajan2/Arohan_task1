"""Render each page with Streamlit's AppTest. Uses the test-only fixture project from conftest."""
import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

PAGES = ["render_overview", "render_dataset", "render_supervised", "render_unsupervised",
         "render_labs_notes_reports"]
ROOT = Path(__file__).resolve().parents[1]


def _script():
    import os
    import sys
    sys.path.insert(0, os.environ["AROHAN_SRC"])
    from arohan_clinical import ui, views
    ui.inject_css()
    getattr(views, os.environ["AROHAN_PAGE"])()


def _run(page, monkeypatch):
    monkeypatch.setenv("AROHAN_SRC", str(ROOT / "src"))
    monkeypatch.setenv("AROHAN_PAGE", page)
    at = AppTest.from_function(_script, default_timeout=240)
    at.run()
    return at


@pytest.mark.parametrize("page", PAGES)
def test_pages_render_without_data(page, tmp_path, monkeypatch):
    monkeypatch.setenv("AROHAN_ROOT", str(tmp_path))
    at = _run(page, monkeypatch)
    assert not at.exception, at.exception
    assert page in ("render_labs_notes_reports",) or "Dataset not found" in at.markdown[-1].value or \
        any("Dataset not found" in m.value for m in at.markdown)


@pytest.mark.parametrize("page", PAGES)
def test_pages_render_with_data_untrained(page, project, monkeypatch):
    at = _run(page, monkeypatch)
    assert not at.exception, at.exception


@pytest.mark.parametrize("page", PAGES)
def test_pages_render_after_training(page, project, monkeypatch):
    from arohan_clinical.supervised import run_training
    run_training(project)
    at = _run(page, monkeypatch)
    assert not at.exception, at.exception


def test_untrained_shows_no_metrics(project, monkeypatch):
    at = _run("render_supervised", monkeypatch)
    text = " ".join(m.value for m in at.markdown)
    assert "have not been trained" in text
    assert not any(re.search(r"ROC-AUC", t.value) for t in at.markdown if "ar-card" in t.value and "0." in t.value)


def test_training_button_works_and_warning_shown(project, monkeypatch):
    at = _run("render_supervised", monkeypatch)
    at.button(key="train_btn").click().run()
    assert not at.exception, at.exception
    assert (project.reports / "supervised_results.json").exists()
    at.tabs  # tabs rendered after training
    assert any("not a clinical diagnosis" in m.value for m in at.markdown)


def test_dataset_page_does_not_modify_raw(project, monkeypatch):
    from arohan_clinical.utils import sha256_file
    f = next(project.raw.glob("*.csv"))
    before = sha256_file(f)
    _run("render_dataset", monkeypatch)
    assert sha256_file(f) == before


def test_app_entrypoint_starts(project, monkeypatch):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception, at.exception


def test_ablation_button_runs(project, monkeypatch):
    from arohan_clinical.supervised import run_training
    run_training(project)
    at = _run("render_supervised", monkeypatch)
    at.multiselect(key="abl_pick").set_value(["age"]).run()
    at.button(key="abl_btn").click().run()
    assert not at.exception, at.exception
    assert any("without age" in m.value for m in at.markdown)


def test_clinical_notes_tab_free_text_and_empty_state(project, monkeypatch):
    at = _run("render_labs_notes_reports", monkeypatch)
    assert not at.exception, at.exception
    markdown_texts = [m.value for m in at.markdown]
    assert any("Not available in this deployment" in t for t in markdown_texts)
    # Verify free-text area exists with requested label
    assert at.text_area(key="note_text_in") is not None
    assert at.text_area(key="note_text_in").label == "Paste any note text to see the rule-based extraction. No data is stored or sent anywhere."
    # Verify no preset scenario selector exists
    assert not any(sb.key == "note_preset" for sb in at.selectbox)
    # Type custom text and verify extraction works
    at.text_area(key="note_text_in").input("Patient with chest pain, palpitations and hypertension.").run()
    assert not at.exception, at.exception
    updated_markdown = " ".join(m.value for m in at.markdown)
    assert "Identified Cardiology Mentions" in updated_markdown
