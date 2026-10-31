"""Central configuration: paths, dataset facts, colours and the feature dictionary.

Everything in FEATURE_DICTIONARY is an *expectation* based on column names and the
publisher's short description. None of it has been verified against a publisher data
dictionary, so the app labels every entry "unverified".
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .utils import normalize_name

APP_TITLE = "AROHAN Clinical AI"
SOURCE_URL = "https://data.mendeley.com/datasets/dzz48mvjht/1"
DOI = "10.17632/dzz48mvjht.1"
DATASET_NAME = "Cardiovascular Disease Dataset"
DATASET_VERSION = "1"
LICENCE = "CC BY 4.0"
PUBLISHED = "16 April 2021"
PUBLISHER_DESCRIPTION = (
    "Heart-disease dataset acquired from a multispecialty hospital in India, intended for "
    "early-stage heart-disease detection and predictive-model research. The landing page text is "
    "internally inconsistent: it mentions both '14 common features' and '1000 subjects with 12 features'. "
    "The real column count must be taken from the downloaded file."
)

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
HYBRID_K = None             # None = choose k by silhouette on the TRAINING rows inside each fit
HYBRID_K_RANGE = (2, 8)
LEAKAGE_AUC_FLAG = 0.90      # univariate AUC at/above this is flagged for review
HIGH_SCORE_FLAG = 0.95       # test metric at/above this triggers a caution banner
TARGET_OVERRIDE = os.environ.get("AROHAN_TARGET")  # optional: force a target column name

COLORS = {
    "bg": "#0B1120", "sidebar": "#0F172A", "card": "#151F32", "elevated": "#1B2940",
    "primary": "#38BDF8", "secondary": "#2DD4BF", "text": "#F1F5F9", "muted": "#A7B4C8",
    "border": "#2B3A50", "success": "#34D399", "warning": "#FBBF24", "error": "#F87171",
}
CHART_SEQUENCE = ["#38BDF8", "#2DD4BF", "#FBBF24", "#F87171", "#A78BFA", "#34D399"]


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def raw(self) -> Path: return self.root / "data" / "raw"
    @property
    def processed(self) -> Path: return self.root / "data" / "processed"
    @property
    def models(self) -> Path: return self.root / "models"
    @property
    def tables(self) -> Path: return self.root / "outputs" / "tables"
    @property
    def figures(self) -> Path: return self.root / "outputs" / "figures"
    @property
    def reports(self) -> Path: return self.root / "outputs" / "reports"
    @property
    def assets(self) -> Path: return self.root / "assets"
    @property
    def raw_labs(self) -> Path: return self.raw / "labs"
    @property
    def raw_notes(self) -> Path: return self.raw / "notes"
    @property
    def raw_history(self) -> Path: return self.raw / "history"
    @property
    def raw_report_images(self) -> Path: return self.raw / "report_images"
    @property
    def raw_echo_reports(self) -> Path: return self.raw / "echo_reports"

    def ensure(self) -> None:
        for p in (self.raw, self.processed, self.models, self.tables, self.figures, self.reports,
                  self.raw_labs, self.raw_notes, self.raw_history, self.raw_report_images, self.raw_echo_reports):
            p.mkdir(parents=True, exist_ok=True)


def default_paths() -> Paths:
    """Project root = folder containing app.py (or $AROHAN_ROOT, used by tests)."""
    env = os.environ.get("AROHAN_ROOT")
    root = Path(env) if env else Path(__file__).resolve().parents[2]
    return Paths(root.resolve())


UNVERIFIED = "Unverified - based on column name / UCI-style convention"

FEATURE_DICTIONARY = {
    "patientid": dict(label="Patient ID", kind="identifier", unit="", expected=None, plausible=None,
                      description="Record identifier. Never used as a model feature; hidden in previews."),
    "age": dict(label="Age", kind="numeric", unit="years", expected=None, plausible=(1, 120),
                description="Patient age (unit assumed to be years)."),
    "gender": dict(label="Gender", kind="binary", unit="", expected=(0, 1), plausible=None,
                   description="Patient sex, numerically coded. Which code is which is not documented here."),
    "chestpain": dict(label="Chest pain type", kind="categorical", unit="", expected=(0, 1, 2, 3), plausible=None,
                      description="Chest-pain category coded 0-3 (UCI-style convention assumed; mapping unverified)."),
    "restingbp": dict(label="Resting blood pressure", kind="numeric", unit="mm Hg", expected=None, plausible=(50, 260),
                      description="Resting blood pressure on admission (unit assumed mm Hg)."),
    "serumcholestrol": dict(label="Serum cholesterol", kind="numeric", unit="mg/dl", expected=None, plausible=(80, 700),
                            description="Serum cholesterol (unit assumed mg/dl). Zeros are reported separately and are NOT "
                                        "automatically treated as missing."),
    "fastingbloodsugar": dict(label="Fasting blood sugar", kind="binary", unit="", expected=(0, 1), plausible=None,
                              description="Fasting blood sugar flag (UCI convention: 1 = above 120 mg/dl). Unverified."),
    "restingrelectro": dict(label="Resting ECG result", kind="categorical", unit="", expected=(0, 1, 2), plausible=None,
                            description="Resting ECG category coded 0-2. A coded value, not an ECG report or image."),
    "maxheartrate": dict(label="Maximum heart rate", kind="numeric", unit="bpm", expected=None, plausible=(40, 250),
                         description="Maximum heart rate achieved (unit assumed bpm)."),
    "exerciseangia": dict(label="Exercise-induced angina", kind="binary", unit="", expected=(0, 1), plausible=None,
                          description="Exercise-induced angina flag (1 = yes assumed)."),
    "oldpeak": dict(label="ST depression (oldpeak)", kind="numeric", unit="mm", expected=None, plausible=(-3, 10),
                    description="ST depression induced by exercise relative to rest (unit assumed)."),
    "slope": dict(label="Peak-exercise ST slope", kind="categorical", unit="", expected=(0, 1, 2, 3), plausible=None,
                  description="Slope category of the peak-exercise ST segment (coding unverified)."),
    "noofmajorvessels": dict(label="Number of major vessels", kind="categorical", unit="", expected=(0, 1, 2, 3),
                             plausible=None, description="Number of major vessels (0-3 expected; unverified)."),
    "target": dict(label="Target label", kind="target", unit="", expected=(0, 1), plausible=None,
                   description="Presence/absence of heart disease (1 vs 0). Exact label definition is NOT verified; "
                               "it is a cross-sectional label, not a future-risk outcome."),
}

ALIASES = {
    "id": "patientid", "serumcholesterol": "serumcholestrol", "cholesterol": "serumcholestrol",
    "exerciseangina": "exerciseangia", "restingecg": "restingrelectro", "restingelectro": "restingrelectro",
    "numberofmajorvessels": "noofmajorvessels", "heartdisease": "target", "trestbps": "restingbp",
    "thalach": "maxheartrate", "sex": "gender", "cp": "chestpain", "fbs": "fastingbloodsugar", "exang": "exerciseangia",
}

REQUIRED_KEYS = [k for k in FEATURE_DICTIONARY if k != "patientid"]


def canonical_key(column: str) -> str:
    key = normalize_name(column)
    return ALIASES.get(key, key)


def spec_for(column: str) -> dict | None:
    return FEATURE_DICTIONARY.get(canonical_key(column))
