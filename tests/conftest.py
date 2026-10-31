"""Shared fixtures.

IMPORTANT: `make_test_frame` builds a small TEST-ONLY table with the same column layout as the expected
dataset so the code paths can be exercised without the real file. It is only ever written inside pytest's
tmp_path, never to data/raw, and none of its numbers are reported anywhere as results.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arohan_clinical.config import Paths  # noqa: E402

COLUMNS = ["patientid", "age", "gender", "chestpain", "restingBP", "serumcholestrol", "fastingbloodsugar",
           "restingrelectro", "maxheartrate", "exerciseangia", "oldpeak", "slope", "noofmajorvessels", "target"]


def make_test_frame(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(30, 80, n)
    cp = rng.integers(0, 4, n)
    oldpeak = np.round(rng.gamma(1.5, 1.0, n), 1)
    logit = 0.04 * (age - 55) + 0.6 * (cp - 1.5) + 0.5 * (oldpeak - 1.5) + rng.normal(0, 1.2, n)
    return pd.DataFrame({
        "patientid": np.arange(1000, 1000 + n), "age": age, "gender": rng.integers(0, 2, n), "chestpain": cp,
        "restingBP": rng.integers(100, 180, n), "serumcholestrol": rng.integers(150, 350, n),
        "fastingbloodsugar": rng.integers(0, 2, n), "restingrelectro": rng.integers(0, 3, n),
        "maxheartrate": rng.integers(90, 190, n), "exerciseangia": rng.integers(0, 2, n), "oldpeak": oldpeak,
        "slope": rng.integers(0, 4, n), "noofmajorvessels": rng.integers(0, 4, n), "target": (logit > 0).astype(int),
    })[COLUMNS]


@pytest.fixture
def frame():
    return make_test_frame()


@pytest.fixture
def project(tmp_path, monkeypatch, frame):
    """A temporary project root containing a test-only CSV in data/raw."""
    paths = Paths(tmp_path)
    paths.ensure()
    frame.to_csv(paths.raw / "test_fixture_cardio.csv", index=False)
    monkeypatch.setenv("AROHAN_ROOT", str(tmp_path))
    return paths
