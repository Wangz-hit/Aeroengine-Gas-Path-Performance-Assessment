import numpy as np
import pytest
from aero_gas_path.evaluation import regression_metrics, maintenance_metrics


def test_metrics_match_direct_calculation_and_constant_handling():
    truth = np.array([[1., 2], [2, 4], [4, 6]])
    prediction = np.array([[2., 2], [1, 2], [5, 2]])
    result = regression_metrics(truth, prediction, ["a", "b"])
    for i, name in enumerate(("a", "b")):
        diff = prediction[:, i]-truth[:, i]
        assert result[name]["mae"] == pytest.approx(np.mean(abs(diff)))
        assert result[name]["rmse"] == pytest.approx(np.sqrt(np.mean(diff**2)))
        assert result[name]["r2"] == pytest.approx(1-np.sum(diff**2)/np.sum((truth[:, i]-truth[:, i].mean())**2))
    assert result["a"]["pearson"] == pytest.approx(np.corrcoef(truth[:, 0], prediction[:, 0])[0, 1])
    assert result["b"]["pearson"] is None


def test_maintenance_uses_fixed_windows_and_direction():
    truth = np.full((20, 5), 0.5)
    truth[10:] -= 0.2
    prediction = np.full_like(truth, 0.5)
    prediction[10:] -= 0.1
    result = maintenance_metrics(truth, prediction, [10], np.full(20, "test"), 3)
    for row in result:
        assert row["true_delta"] == pytest.approx(-0.2)
        assert row["predicted_recovery"] == pytest.approx(0.1)
        assert row["recovery_absolute_error"] == pytest.approx(0.1)
        assert row["direction_correct"] and row["split"] == "test"
    with pytest.raises(ValueError):
        maintenance_metrics(truth, prediction, [1], np.full(20, "test"), 3)

