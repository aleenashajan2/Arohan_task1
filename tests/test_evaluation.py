import pytest

from arohan_clinical.evaluation import classification_metrics, confusion_counts


def test_known_metrics():
    y_true = [1, 1, 1, 0, 0, 0, 0, 0]
    y_pred = [1, 1, 0, 0, 0, 0, 0, 1]          # tp=2 fn=1 tn=4 fp=1
    m = classification_metrics(y_true, y_pred)
    assert (m["tp"], m["fn"], m["tn"], m["fp"]) == (2, 1, 4, 1)
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["specificity"] == pytest.approx(0.8)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["roc_auc"] is None


def test_auc_perfect_and_inverse():
    y = [0, 0, 1, 1]
    assert classification_metrics(y, y, [0.1, 0.2, 0.8, 0.9])["roc_auc"] == pytest.approx(1.0)
    assert classification_metrics(y, y, [0.9, 0.8, 0.2, 0.1])["roc_auc"] == pytest.approx(0.0)


def test_single_class_has_no_auc():
    m = classification_metrics([1, 1, 1], [1, 1, 0], [0.9, 0.8, 0.1])
    assert m["roc_auc"] is None and m["specificity"] is None


def test_confusion_layout():
    cm = confusion_counts([0, 1, 1], [0, 1, 0])
    assert cm.tolist() == [[1, 0], [1, 1]]
