import numpy as np

from arohan_clinical.data_loader import detect_roles
from arohan_clinical.data_quality import (build_quality_report, duplicate_checks, missing_values, outlier_summary,
                                          range_checks, unexpected_categories, zero_value_report)


def test_missing_values_reported(frame):
    frame.loc[0:4, "age"] = np.nan
    m = missing_values(frame).set_index("column")
    assert m.loc["age", "missing"] == 5 and m.loc["gender", "missing"] == 0


def test_duplicates(frame):
    frame = frame.copy()
    frame.loc[1] = frame.loc[0]                 # same id and same values
    d = duplicate_checks(frame, "patientid")
    assert d["full_row_duplicates"] == 1 and d["duplicate_ids"] == 1 and d["duplicates_excluding_id"] >= 1
    frame.loc[2, "patientid"] = 99999
    assert duplicate_checks(frame, "patientid")["duplicate_ids"] == 1


def test_unexpected_categories(frame):
    assert unexpected_categories(frame).empty
    frame.loc[0, "gender"] = 7
    u = unexpected_categories(frame)
    assert list(u["column"]) == ["gender"] and u.loc[0, "rows_affected"] == 1


def test_zero_values_counted_not_imputed(frame):
    frame.loc[0:2, "serumcholestrol"] = 0
    roles = detect_roles(frame)
    before = frame.copy()
    rep = build_quality_report(frame, roles)
    z = rep["zeros"].set_index("column")
    assert z.loc["serumcholestrol", "zero_values"] == 3
    assert any("serumcholestrol" in w for w in rep["warnings"])
    assert frame.equals(before)                 # nothing modified


def test_range_and_outliers(frame):
    frame.loc[0, "age"] = 300
    r = range_checks(frame).set_index("column")
    assert r.loc["age", "rows_outside"] == 1
    o = outlier_summary(frame, ["age"]).set_index("column")
    assert o.loc["age", "outliers"] >= 1


def test_status_levels(frame):
    roles = detect_roles(frame)
    assert build_quality_report(frame, roles)["status"] == "Warnings"      # fixture has oldpeak zeros
    frame["oldpeak"] = frame["oldpeak"] + 0.1                              # remove zeros -> clean frame
    assert build_quality_report(frame, roles)["status"] == "Passed"
    frame.loc[0, "age"] = np.nan
    assert build_quality_report(frame, roles)["status"] == "Issues found"
