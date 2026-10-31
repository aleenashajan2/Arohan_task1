import pandas as pd
import pytest

from arohan_clinical import config
from arohan_clinical.data_loader import (DatasetError, build_dataset_info, detect_roles, find_dataset_file,
                                         load_dataset, validate_required_columns, write_dataset_info)
from arohan_clinical.utils import sha256_file


def test_find_and_load(project):
    f = find_dataset_file(project.raw)
    assert f is not None and f.suffix == ".csv"
    df = load_dataset(f)
    assert df.shape == (300, 14)


def test_missing_dataset_returns_none(tmp_path):
    assert find_dataset_file(tmp_path / "nope") is None


def test_empty_csv_raises(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("")
    with pytest.raises(DatasetError):
        load_dataset(p)


def test_roles_detected_and_id_not_a_feature(frame):
    roles = detect_roles(frame)
    assert roles.id_col == "patientid" and roles.target_col == "target"
    assert "patientid" not in roles.feature_cols and "target" not in roles.feature_cols
    assert len(roles.feature_cols) == 12


def test_required_column_validation(frame):
    assert validate_required_columns(frame) == []
    assert "age" in validate_required_columns(frame.drop(columns=["age"]))


def test_alias_columns_recognised(frame):
    renamed = frame.rename(columns={"serumcholestrol": "cholesterol", "restingrelectro": "restingECG"})
    assert validate_required_columns(renamed) == []


def test_info_has_checksum_and_raw_is_unchanged(project):
    f = find_dataset_file(project.raw)
    before = sha256_file(f)
    df = load_dataset(f)
    info = write_dataset_info(project, build_dataset_info(f, df, detect_roles(df)))
    assert info["sha256"] == before and info["source_url"] == config.SOURCE_URL and info["licence"] == "CC BY 4.0"
    assert (project.reports / "dataset_info.json").exists()
    assert sha256_file(f) == before
    again = write_dataset_info(project, build_dataset_info(f, df, detect_roles(df)))
    assert again["first_recorded_utc"] == info["first_recorded_utc"]


def test_dtypes_numeric(frame):
    assert all(pd.api.types.is_numeric_dtype(frame[c]) for c in frame.columns)
