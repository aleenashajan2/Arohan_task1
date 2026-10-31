"""Smoke test for report_builder."""
from pathlib import Path
from arohan_clinical import config, report_builder

def test_report_builder_runs():
    paths = config.default_paths()
    res = report_builder.build_reports()
    assert "markdown" in res
    assert "docx" in res
    assert res["markdown"].exists()
    assert res["docx"].exists()
    assert res["markdown"].stat().st_size > 5000
    assert res["docx"].stat().st_size > 10000
