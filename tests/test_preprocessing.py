import numpy as np
from aero_gas_path.preprocessing import romanovsky_filter, eemd_nlm, preprocess_monitoring
from aero_gas_path.simulation import simulate_engine


def test_outlier_filter():
    x = 0.02 * np.sin(np.arange(90) / 10)
    x[45] = 3
    filtered = romanovsky_filter(x)
    assert abs(filtered[45]) < 0.1


def test_preprocess_and_shapes():
    observations, truth = simulate_engine(n_samples=96)
    assert observations.shape[0] == truth.shape[0] == 96
    processed = preprocess_monitoring(observations, ensembles=2)
    assert processed.shape == (96, 4)
    assert np.isfinite(processed.iloc[:, 1:].to_numpy()).all()
    signal = eemd_nlm(np.sin(np.arange(96) / 8), ensembles=2)
    assert len(signal) == 96


def test_offline_path_is_finite_and_is_explicitly_bidirectional():
    observations, _ = simulate_engine(n_samples=96)
    full = preprocess_monitoring(observations, mode="offline", ensembles=2)
    changed = observations.copy()
    changed.loc[60:, "DEGT"] += 0.5
    altered = preprocess_monitoring(changed, mode="offline", ensembles=2)
    assert np.isfinite(full.iloc[:, 1:].to_numpy()).all()
    # The retained batch algorithm must not be misrepresented as causal.
    assert np.max(abs(full.DEGT.iloc[:60]-altered.DEGT.iloc[:60])) > 1e-6
