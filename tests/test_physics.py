import numpy as np
import torch
from aero_gas_path.physics import RAW_FINGERPRINT, FingerprintDecoder, fingerprint_matrix
from aero_gas_path.model import DTBAutoEncoder


def test_original_script_fingerprint_values_and_normalization():
    original = np.array([[6.0, 1.3, 0.7], [4.0, 0.4, 0.1], [6.0, 0.6, -0.8],
                         [8.0, 0.8, -1.0], [8.0, 1.7, 0.8]])
    np.testing.assert_array_equal(RAW_FINGERPRINT, original)
    np.testing.assert_allclose(fingerprint_matrix(), original/np.linalg.norm(original, axis=0), atol=1e-7)


def test_linear_decoder_bias():
    x = torch.randn(2, 12, 5)
    decoder = FingerprintDecoder()
    with torch.no_grad():
        decoder.bias.copy_(torch.tensor([0.2, -0.1, 0.03]))
    y = decoder(x)
    torch.testing.assert_close(y, x @ decoder.fingerprint + decoder.bias)
    assert y.shape == (2, 12, 3)


def test_three_five_three_and_gradients():
    model = DTBAutoEncoder(conv_channels=(8, 8), lstm_hidden=12, mlp_hidden=16)
    latent, signal = model(torch.randn(4, 9, 3))
    assert latent.shape == (4, 9, 5) and signal.shape == (4, 9, 3)
    signal.square().mean().backward()
    assert model.tcn.network[0].conv1.weight.grad.abs().sum().item() > 0
