import numpy as np
import pytest
import torch
from aero_gas_path.model import DTBAutoEncoder
from aero_gas_path.physics import fingerprint_matrix, identifiability_report


def test_temporal_network_and_affine_decoder():
    torch.manual_seed(9)
    model = DTBAutoEncoder(conv_channels=(6, 6), lstm_hidden=8, mlp_hidden=10).eval()
    x = torch.randn(2, 13, 3)
    z, y = model(x)
    changed = x.clone()
    changed[:, 7:] += 20
    z_changed, _ = model(changed)
    torch.testing.assert_close(z[:, :7], z_changed[:, :7])
    torch.testing.assert_close(y, z @ model.decoder.fingerprint + model.decoder.bias)
    assert list(model.decoder.parameters()) == [model.decoder.bias]
    y[:, -1].square().sum().backward()
    gradients = [p.grad for p in model.tcn.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in gradients)
    assert all(g.abs().sum() > 0 for g in gradients)
    assert model.decoder.bias.grad is not None


def test_fingerprint_rank_and_ambiguity():
    report = identifiability_report()
    assert (report["rank"], report["nullity"]) == (3, 2)
    a, b = np.array(report["ambiguous_nonnegative_states"])
    assert np.min(b) > 0 and np.linalg.norm(a-b) > 0.1
    np.testing.assert_allclose(a @ fingerprint_matrix(), b @ fingerprint_matrix(), atol=1e-10)


def test_invalid_network_dimensions():
    with pytest.raises(ValueError):
        DTBAutoEncoder(conv_channels=())
    with pytest.raises(ValueError):
        DTBAutoEncoder()(torch.zeros(2, 10, 4))
