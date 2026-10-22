"""Gas-path sensitivity relations and control-setting corrections."""

import numpy as np
import torch
from torch import nn

COMPONENTS = ("FAN", "LPC", "HPC", "HPT", "LPT")
PARAMETERS = ("DEGT", "DFF", "DN2")

# Rows: FAN, LPC, HPC, HPT, LPT; columns: DEGT, DFF, DN2.
# Original entries from fun_filter() in the supplied research source script.
# Their numerical provenance is the source file; no new calibration is implied.
RAW_FINGERPRINT = np.array(
    [[6.0, 1.3, 0.7],
     [4.0, 0.4, 0.1],
     [6.0, 0.6, -0.8],
     [8.0, 0.8, -1.0],
     [8.0, 1.7, 0.8]], dtype=np.float64
)

# Control setting response coefficients, after normalization of VSV and VBV.
VSV_RESPONSE = np.array([1.0, 0.0, 1.2], dtype=np.float32)
VBV_RESPONSE = np.array([-11.0, -1.0, -0.1], dtype=np.float32)


def fingerprint_matrix() -> np.ndarray:
    """Return the 5-by-3 fingerprint normalized by parameter column norms."""
    scale = np.linalg.norm(RAW_FINGERPRINT, axis=0, keepdims=True)
    return RAW_FINGERPRINT / scale


def identifiability_report() -> dict:
    """Diagnose the observable subspace; temporal learning does not change rank."""
    matrix = fingerprint_matrix().astype(np.float64)
    left, singular, _ = np.linalg.svd(matrix, full_matrices=True)
    rank = int(np.linalg.matrix_rank(matrix))
    norms = np.linalg.norm(matrix, axis=1)
    cosine = matrix @ matrix.T / np.outer(norms, norms)
    null = left[:, rank:]
    # Two nonnegative, different states with numerically identical observations.
    state_a = np.full(5, 0.3)
    state_b = state_a + 0.15 * null[:, 0]
    return {"rank": rank, "nullity": 5-rank,
            "singular_values": singular.tolist(),
            "condition_number": float(singular[0]/singular[-1]),
            "row_cosine_similarity": cosine.tolist(),
            "nullspace_basis": null.tolist(),
            "ambiguous_nonnegative_states": [state_a.tolist(), state_b.tolist()],
            "ambiguous_response_max_difference": float(np.max(abs((state_a-state_b) @ matrix)))}


def control_response(vsv: np.ndarray, vbv: np.ndarray) -> np.ndarray:
    """Gas-path shifts associated with VSV and VBV control settings."""
    return ((np.asarray(vsv) / 2.0)[..., None] * VSV_RESPONSE
            + (np.asarray(vbv) / 10.0)[..., None] * VBV_RESPONSE)


class FingerprintDecoder(nn.Module):
    """Fingerprint-constrained affine decoder (paper, Eq. 38)."""

    def __init__(self, fingerprint: np.ndarray | None = None):
        super().__init__()
        values = fingerprint_matrix() if fingerprint is None else np.asarray(fingerprint)
        if values.shape != (5, 3) or not np.isfinite(values).all():
            raise ValueError("Fingerprint sensitivity matrix must be 5 x 3.")
        self.register_buffer("fingerprint", torch.as_tensor(values, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(3, dtype=torch.float32))

    def forward(self, degradation: torch.Tensor) -> torch.Tensor:
        if degradation.shape[-1] != 5:
            raise ValueError("Expected 5 component states.")
        return degradation @ self.fingerprint + self.bias
