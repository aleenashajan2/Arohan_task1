from arohan_clinical.clinical_coverage import coverage_table, implementation_status, scan_raw_folder


def test_with_dataset():
    t = coverage_table(True, {"filename": "x.csv"}).set_index("Clinical data type")
    assert len(t) == 7
    assert t.loc["Structured clinical measurements", "Available"] == "Yes"
    for k in ("Blood reports", "Laboratory reports", "Imaging reports", "Clinical notes", "Discharge summaries"):
        assert t.loc[k, "Available"] == "No"
    assert t.loc["Multimodal integration", "Integration status"] == "Not implemented"


def test_without_dataset():
    t = coverage_table(False, None).set_index("Clinical data type")
    assert t.loc["Structured clinical measurements", "Available"] == "Pending"


def test_extra_files_listed_not_processed(project):
    (project.raw / "report1.pdf").write_bytes(b"%PDF-1.4")
    assert scan_raw_folder(project) == ["report1.pdf"]
    t = coverage_table(True, {"filename": "x.csv"}, scan_raw_folder(project))
    assert t.loc[t["Clinical data type"] == "Blood reports", "Available"].iloc[0] == "No"


def test_implementation_status_honest():
    s = implementation_status(True, 0, True).set_index("Component")
    assert s.loc["Supervised learning", "Status"] == "Pending"
    assert s.loc["Hybrid supervised + unsupervised", "Status"] == "Pending"
    assert implementation_status(True, 6, True, 3).set_index("Component").loc["Hybrid supervised + unsupervised", "Status"] == "Implemented"
    assert s.loc["Imaging reports", "Status"] == "Unavailable"
    assert s.loc["Multimodal integration", "Status"] == "Not implemented"
