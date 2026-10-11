import pytest

from evaluate import calculate_metrics


def test_metrics_calculation():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]

    metrics = calculate_metrics(y_true, y_pred)

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(0.5)