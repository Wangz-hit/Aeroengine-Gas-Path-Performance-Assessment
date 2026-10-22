"""Reproducible synthetic in-service gas-path monitoring sequences."""

from pathlib import Path
import numpy as np
import pandas as pd
from .physics import COMPONENTS, PARAMETERS, control_response, fingerprint_matrix


def simulate_engine(n_samples: int = 360, seed: int = 17,
                    noise_std: float = 0.012, outlier_fraction: float = 0.018,
                    maintenance: tuple[float, ...] = (0.53, 0.81),
                    individual_variation: float = 0.25) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate 3 measured channels from 5 latent degradation trajectories.

    Synthetic component states and maintenance events are illustrative; they are
    not measurements or reconstructions of any published aeroengine case.
    """
    if n_samples < 80:
        raise ValueError("n_samples must be at least 80.")
    if noise_std < 0 or not 0 <= outlier_fraction <= 1 or not 0 <= individual_variation < 1:
        raise ValueError("Invalid noise, outlier fraction, or individual variation.")
    if any(not 0 < f < 1 for f in maintenance) or sorted(set(maintenance)) != list(maintenance):
        raise ValueError("Maintenance fractions must be strictly increasing in (0,1).")
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples, dtype=np.float32)
    dates = pd.date_range("2012-01-01", periods=n_samples, freq="7D")
    # Different degradation rates and baseline offsets for five components.
    base = np.array([0.085, 0.045, 0.11, 0.16, 0.09], dtype=np.float32)
    slopes = np.array([0.39, 0.27, 0.49, 0.68, 0.47], dtype=np.float32)
    cycles = np.array([0.014, 0.010, 0.019, 0.028, 0.016], dtype=np.float32)
    # Independent component variability creates different coupled degradation cases.
    base *= rng.uniform(1-individual_variation, 1+individual_variation, 5)
    slopes *= rng.uniform(1-individual_variation, 1+individual_variation, 5)
    latent = base[None, :] + t[:, None] / n_samples * slopes[None, :]
    latent += np.sin(t[:, None] / 17 + np.arange(5)[None, :] * 0.7) * cycles[None, :]
    latent += rng.normal(0.0, 0.004, size=latent.shape).astype(np.float32)
    for fraction in maintenance:
        event = round(fraction * n_samples)
        if 0 < event < n_samples:
            # Abrupt improvement and gradual post-repair deterioration.
            recovery = np.array([0.07, 0.04, 0.11, 0.23, 0.14], dtype=np.float32)
            recovery *= rng.uniform(1-individual_variation, 1+individual_variation, 5)
            latent[event:] -= recovery
    latent = np.maximum(latent, 0.002)
    clean = latent @ fingerprint_matrix()
    clean += rng.normal(0.0, noise_std, clean.shape)
    vsv = 0.020 * np.sin(t / 27) + rng.normal(0, 0.004, n_samples)
    vbv = 0.015 * np.cos(t / 19) + rng.normal(0, 0.003, n_samples)
    observed = clean - control_response(vsv, vbv)
    count = round(outlier_fraction * n_samples)
    idx = rng.choice(n_samples, size=count, replace=False)
    channel = rng.integers(0, 3, size=count)
    observed[idx, channel] += rng.choice([-1, 1], size=count) * rng.uniform(0.14, 0.26, size=count)
    frame = pd.DataFrame(observed, columns=PARAMETERS)
    frame.insert(0, "date", dates.strftime("%Y-%m-%d"))
    frame["VSV"] = vsv
    frame["VBV"] = vbv
    truth = pd.DataFrame(latent, columns=COMPONENTS)
    truth.insert(0, "date", dates.strftime("%Y-%m-%d"))
    return frame, truth


def save_simulation(output_dir: Path, **kwargs) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    measurements, truth = simulate_engine(**kwargs)
    measurements_path = output_dir / "monitoring.csv"
    truth_path = output_dir / "synthetic_truth.csv"
    measurements.to_csv(measurements_path, index=False)
    truth.to_csv(truth_path, index=False)
    return measurements_path, truth_path
