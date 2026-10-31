import numpy as np

from arohan_clinical.supervised import (is_stale, load_model, load_results, model_path, predict_one, run_training,
                                        validate_input)
from arohan_clinical.utils import sha256_file
from arohan_clinical.data_loader import find_dataset_file


def test_training_saves_models_and_results(project):
    before = sha256_file(find_dataset_file(project.raw))
    res = run_training(project)
    base = {"Logistic Regression", "Decision Tree", "Random Forest"}
    assert set(res["models"]) == base | {f"Hybrid {n}" for n in base}
    assert {m["family"] for m in res["models"].values()} == {"baseline", "hybrid"}
    assert all(res["models"][f"Hybrid {n}"]["n_clusters"] >= 2 for n in base)
    for name in res["models"]:
        assert model_path(project, name).exists()
    assert (project.processed / "train.csv").exists() and (project.tables / "supervised_test_metrics.csv").exists()
    assert sha256_file(find_dataset_file(project.raw)) == before      # raw untouched
    assert "patientid" not in res["features"] and res["target_column"] == "target"
    assert load_results(project) is not None


def test_all_models_scored_on_same_test_rows(project):
    res = run_training(project)
    n = len(res["y_test"])
    assert all(len(m["y_pred"]) == n and len(m["y_score"]) == n for m in res["models"].values())
    assert n == res["split"]["n_test"]


def test_metrics_match_saved_predictions(project):
    from arohan_clinical.evaluation import classification_metrics
    res = run_training(project)
    for m in res["models"].values():
        again = classification_metrics(res["y_test"], m["y_pred"], m["y_score"])
        assert again["accuracy"] == m["test_metrics"]["accuracy"]


def test_reproducible(project):
    a = run_training(project)
    b = run_training(project)
    for n in a["models"]:
        assert a["models"][n]["y_score"] == b["models"][n]["y_score"]


def test_save_load_predict(project):
    res = run_training(project)
    model = load_model(project, "Random Forest")
    spec = res["input_spec"]
    values = {c: (s["default"] if s["kind"] == "numeric" else s["options"][0]) for c, s in spec.items()}
    errors, _ = validate_input(spec, values)
    assert errors == []
    out = predict_one(model, res, values)
    assert 0.0 <= out["score"] <= 1.0 and out["predicted_class"] in res["classes"]


def test_input_validation(project):
    spec = run_training(project)["input_spec"]
    good = {c: (s["default"] if s["kind"] == "numeric" else s["options"][0]) for c, s in spec.items()}
    bad = dict(good, age=500, gender="9")
    errors, _ = validate_input(spec, bad)
    assert len(errors) == 2
    missing = {k: v for k, v in good.items() if k != "age"}
    assert validate_input(spec, missing)[0]
    assert validate_input(spec, dict(good, age="abc"))[0]


def test_stale_detection(project):
    res = run_training(project)
    assert not is_stale(res, res["dataset_sha256"])
    assert is_stale(res, "different")


def test_leakage_screen_flags_perfect_feature(project, frame):
    frame = frame.copy()
    frame["chestpain"] = frame["target"]          # deliberately leaky feature
    frame.to_csv(project.raw / "test_fixture_cardio.csv", index=False)
    res = run_training(project)
    flagged = [r["feature"] for r in res["leakage_screen"] if r["flagged"]]
    assert "chestpain" in flagged


def test_ablation_same_split_and_consistent_with_saved_results(project, frame):
    from arohan_clinical.data_loader import detect_roles
    from arohan_clinical.supervised import run_ablation
    res = run_training(project)
    abl = run_ablation(frame, detect_roles(frame), ["age"], res["split"]["seed"], res["split"]["test_size"])
    full = abl[abl["Variant"] == "All features"].set_index("Model")
    for name, m in res["models"].items():       # same rows + same code path => identical test scores
        assert full.loc[name, "ROC-AUC"] == m["test_metrics"]["roc_auc"]
    assert set(abl["Variant"]) == {"All features", "Without age"}


def test_ablation_of_leaky_feature_lowers_score(project, frame):
    from arohan_clinical.data_loader import detect_roles
    from arohan_clinical.supervised import run_ablation
    leaky = frame.copy()
    leaky["chestpain"] = leaky["target"]
    abl = run_ablation(leaky, detect_roles(leaky), ["chestpain"])
    lr = abl[abl["Model"] == "Logistic Regression"].set_index("Variant")["ROC-AUC"]
    assert lr["All features"] > 0.99 and lr["Without chestpain"] < lr["All features"] - 0.05


def test_ablation_validates_input(frame):
    import pytest
    from arohan_clinical.data_loader import detect_roles
    from arohan_clinical.supervised import run_ablation
    with pytest.raises(ValueError):
        run_ablation(frame, detect_roles(frame), ["not_a_column"])
