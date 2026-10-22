"""Fingerprint correction, sliding-window error removal, and EEMD-NLM filtering."""

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.signal import find_peaks
from scipy.stats import norm, ttest_1samp
from .physics import PARAMETERS, control_response


def romanovsky_filter(signal: np.ndarray, window: int = 15, alpha: float = 0.05) -> np.ndarray:
    """Romanovsky median/MAD screening with interpolation (paper Eqs. 7--9).

    A local window supplies the reference median and MAD. For a rejected point,
    neighboring accepted observations interpolate the missing value. The full
    batch is used, as in the paper's offline signal-processing procedure.
    """
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1 or window < 5 or window % 2 == 0 or not 0 < alpha < 1:
        raise ValueError("Expected a 1D signal, odd window >=5 and alpha in (0,1).")
    if not np.isfinite(values).all():
        raise ValueError("Signal must be finite.")
    radius = window // 2
    bad = np.zeros(len(values), dtype=bool)
    for i in range(len(values)):
        local = values[max(0, i-radius):min(len(values), i+radius+1)]
        if len(local) < 5:
            continue
        center = np.median(local)
        mad = np.median(np.abs(local - center))
        if mad <= 1e-12:
            continue
        k = norm.ppf(1-alpha/(2*len(local)))
        bad[i] = abs(values[i] - center) > k * mad
    if not bad.any():
        return values.copy()
    valid = np.flatnonzero(~bad)
    if not len(valid):
        return values.copy()
    cleaned = values.copy()
    missing = np.flatnonzero(bad)
    cleaned[missing] = np.interp(missing, valid, values[valid])
    return cleaned


def _envelope(signal: np.ndarray, mode: str) -> np.ndarray:
    peaks = find_peaks(signal if mode == "upper" else -signal)[0]
    positions = np.unique(np.r_[0, peaks, len(signal)-1])
    return PchipInterpolator(positions, signal[positions])(np.arange(len(signal)))


def _emd(signal: np.ndarray, max_imfs: int = 3, sifts: int = 5) -> tuple[list[np.ndarray], np.ndarray]:
    """Empirical mode decomposition with shape-preserving cubic envelopes."""
    residual = np.asarray(signal, dtype=np.float64).copy()
    components = []
    for _ in range(max_imfs):
        if min(len(find_peaks(residual)[0]), len(find_peaks(-residual)[0])) < 2:
            break
        h = residual.copy()
        for _ in range(sifts):
            if min(len(find_peaks(h)[0]), len(find_peaks(-h)[0])) < 2:
                break
            h -= 0.5 * (_envelope(h, "upper") + _envelope(h, "lower"))
        residual -= h
        components.append(h)
    return components, residual


def eemd(signal: np.ndarray, ensembles: int = 8, noise_width: float = 0.14,
         max_imfs: int = 3, seed: int = 17) -> tuple[np.ndarray, np.ndarray]:
    """Ensemble EMD with seeded complementary-noise realizations."""
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1 or len(values) < 15:
        raise ValueError("Expected a one-dimensional signal with >=15 values.")
    if ensembles < 2:
        raise ValueError("ensembles must be >= 2")
    rng = np.random.default_rng(seed)
    acc = np.zeros((max_imfs, len(values)), dtype=np.float64)
    sigma = max(1e-8, np.std(values)) * noise_width
    # Pair positive and negative noise realizations to reduce residual noise.
    for iteration in range(ensembles):
        noise = rng.normal(0.0, sigma, len(values))
        if iteration % 2:
            noise = -stored_noise
        else:
            stored_noise = noise
        imfs, _ = _emd(values + noise, max_imfs=max_imfs)
        for j, imf in enumerate(imfs):
            acc[j] += imf
    mean_imfs = acc / ensembles
    trend = values - mean_imfs.sum(axis=0)
    return mean_imfs, trend


def nonlocal_means_1d(signal: np.ndarray, patch_radius: int = 2,
                      search_radius: int = 12, bandwidth: float | None = None) -> np.ndarray:
    """One-dimensional nonlocal means using local patch similarity."""
    x = np.asarray(signal, dtype=np.float64)
    if len(x) < 2:
        return x.copy()
    pad = np.pad(x, patch_radius, mode="reflect")
    patches = np.stack([pad[j:j+len(x)] for j in range(2*patch_radius+1)], axis=1)
    h = max(1e-8, float(bandwidth) if bandwidth is not None else 0.65*np.std(x))
    output = np.empty_like(x)
    for i in range(len(x)):
        lo, hi = max(0, i-search_radius), min(len(x), i+search_radius+1)
        dist2 = np.mean((patches[lo:hi] - patches[i])**2, axis=1)
        weights = np.exp(-dist2/(h*h))
        output[i] = np.dot(weights, x[lo:hi]) / weights.sum()
    return output


def eemd_nlm(signal: np.ndarray, ensembles: int = 8, seed: int = 17,
             alpha: float = 0.05) -> np.ndarray:
    """Ensemble EMD, cumulative-IMF significance screening, then NLM.

    The paper uses a one-sample T test to discriminate high-frequency IMF
    components. Here non-rejection of zero IMF mean (p > alpha)
    selects a high-frequency IMF. The EMD envelopes and 1D NLM are compact
    numerical approximations, not a bit-exact reproduction of the paper's
    two-dimensional NLM grid.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1).")
    components, trend = eemd(signal, ensembles=ensembles, seed=seed)
    if not len(components):
        return trend
    reconstructed = components.copy()
    for index, imf in enumerate(components):
        if np.std(imf, ddof=1) < 1e-12:
            p_value = 1.0
        else:
            p_value = float(ttest_1samp(imf, popmean=0).pvalue)
        if p_value > alpha:
            reconstructed[index] = nonlocal_means_1d(imf)
        else:
            break  # lower-frequency IMFs are preserved
    return trend + reconstructed.sum(axis=0)


def preprocess_monitoring(frame: pd.DataFrame, *, window: int = 15,
                          ensembles: int = 8, seed: int = 17,
                          denoise: bool = True, mode: str = "offline") -> pd.DataFrame:
    """Return corrected and filtered 3-channel gas-path measurements."""
    required = {"date", "VSV", "VBV", *PARAMETERS}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns: {sorted(required-set(frame.columns))}")
    if mode != "offline" or window < 5 or window % 2 == 0:
        raise ValueError("The paper-aligned preprocessing mode is offline, with odd window >=5.")
    dates = pd.to_datetime(frame["date"], errors="raise")
    if not dates.is_monotonic_increasing or dates.duplicated().any():
        raise ValueError("Dates must be unique and increasing; align truth before preprocessing.")
    frame = frame.reset_index(drop=True)
    corrected = frame[list(PARAMETERS)].to_numpy(dtype=np.float64)
    corrected += control_response(frame["VSV"].to_numpy(), frame["VBV"].to_numpy())
    if not np.isfinite(corrected).all():
        raise ValueError("Monitoring and control values must be finite.")
    values = []
    for j in range(3):
        x = romanovsky_filter(corrected[:, j], window=window)
        if denoise:
            x = eemd_nlm(x, ensembles=ensembles, seed=seed+j)
        values.append(x)
    output = pd.DataFrame(np.stack(values, axis=1), columns=PARAMETERS)
    output.insert(0, "date", frame["date"].to_numpy())
    return output
