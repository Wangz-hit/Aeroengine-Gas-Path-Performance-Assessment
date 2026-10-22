"""Paper-aligned reconstruction and sensitivity-normalized degradation losses."""

import torch
from .model import DTBAutoEncoder


def physics_objective(model: DTBAutoEncoder, windows: torch.Tensor,
                      target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    """L1 + L2 as in the paper's Eqs. (39)--(41).

    L1 = mean_t sum_k (reconstruction_k - observation_k)^2.
    L2 = mean_t sum_j |component_j| / ||fingerprint_j||_2.
    Targets are original corrected deviation units. No supervision is used.
    """
    states, reconstructed = model(windows)
    z, y = states[:, -1, :], reconstructed[:, -1, :]
    if target.shape != y.shape:
        raise ValueError("Targets must contain three gas-path deviations per sample.")
    l1 = ((y - target) ** 2).sum(dim=-1).mean()
    norms = torch.linalg.vector_norm(model.decoder.fingerprint, dim=1).clamp_min(1e-12)
    l2 = (z.abs() / norms).sum(dim=-1).mean()
    total = l1 + l2
    return total, {"loss": total.detach().item(),
                   "reconstruction": l1.detach().item(),
                   "sensitivity": l2.detach().item()}
