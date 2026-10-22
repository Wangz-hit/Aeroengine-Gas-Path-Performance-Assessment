import numpy as np
import torch
from aero_gas_path.dataset import rolling_windows, chronological_split
from aero_gas_path.demo import make_datasets
from aero_gas_path.losses import physics_objective
from aero_gas_path.model import DTBAutoEncoder
from aero_gas_path.physics import fingerprint_matrix
from aero_gas_path.training import TrainConfig, fit_model, load_model, predict, seed_everything


def test_objective_matches_paper_terms():
    model = DTBAutoEncoder(dropout=0).eval()
    windows = torch.randn(3, 8, 3)
    target = torch.randn(3, 3)
    loss, parts = physics_objective(model, windows, target)
    z, y = model(windows)
    expected_l1 = ((y[:, -1] - target)**2).sum(-1).mean()
    expected_l2 = (z[:, -1].abs()/torch.linalg.vector_norm(model.decoder.fingerprint, dim=1)).sum(-1).mean()
    torch.testing.assert_close(loss, expected_l1+expected_l2)
    assert parts["reconstruction"] == expected_l1.item()
    assert parts["sensitivity"] == expected_l2.item()
    assert all("label" not in x for x in [parts])


def test_train_reload_and_deterministic_inference(tmp_path):
    torch.set_num_threads(2)
    states = np.random.default_rng(1).uniform(0.05, 0.4, (80, 5)).astype(np.float32)
    windows = rolling_windows(states @ fingerprint_matrix(), 6)
    partition = chronological_split(80, 0.6, 0.15)
    train, valid = make_datasets([windows], [partition])
    seed_everything(17)
    model = DTBAutoEncoder(conv_channels=(4,), lstm_hidden=6, mlp_hidden=8, dropout=0)
    cfg = TrainConfig(epochs=2, patience=2, batch_size=24)
    before = model.tcn.network[0].conv1.weight.detach().clone()
    fit_model(model, train, valid, cfg, tmp_path/"model.pt")
    assert not torch.equal(before, model.tcn.network[0].conv1.weight)
    restored = load_model(tmp_path/"model.pt")
    assert not restored.training
    a, ya = predict(model, windows[partition.test])
    b, yb = predict(restored, windows[partition.test])
    np.testing.assert_allclose(a, b, atol=1e-7)
    np.testing.assert_allclose(ya, yb, atol=1e-7)
