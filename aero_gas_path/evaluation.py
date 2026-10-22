"""Unit-preserving metrics and prespecified maintenance-window diagnostics."""

import numpy as np
from .physics import COMPONENTS, PARAMETERS, fingerprint_matrix


def regression_metrics(truth: np.ndarray, prediction: np.ndarray, names) -> dict:
    truth, prediction = np.asarray(truth, float), np.asarray(prediction, float)
    if truth.shape != prediction.shape or truth.ndim != 2 or truth.shape[1] != len(names):
        raise ValueError("Expected matching [N,channels] arrays.")
    if len(truth) < 2 or not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("At least two finite observations are required.")
    result = {}
    for j, name in enumerate(names):
        y, p = truth[:, j], prediction[:, j]
        error = p-y
        variance = np.sum((y-y.mean())**2)
        nonconstant = np.std(y) > 1e-12 and np.std(p) > 1e-12
        result[name] = {"mae": float(np.mean(abs(error))),
                        "rmse": float(np.sqrt(np.mean(error**2))),
                        "r2": float(1-np.sum(error**2)/variance) if variance > 1e-20 else None,
                        "pearson": float(np.corrcoef(y, p)[0, 1]) if nonconstant else None,
                        "std_ratio": float(np.std(p)/np.std(y)) if np.std(y) > 1e-12 else None}
    return result


def evaluate_predictions(truth: np.ndarray, target: np.ndarray,
                         latent: np.ndarray, reconstructed: np.ndarray,
                         split: np.ndarray, engine_ids: np.ndarray) -> dict:
    matrix = fingerprint_matrix().astype(float)
    observable = matrix @ np.linalg.pinv(matrix)
    result = {}
    for section in ("train", "valid", "test"):
        mask = split == section
        error = latent[mask]-truth[mask]
        components = regression_metrics(truth[mask], latent[mask], COMPONENTS)
        # Difference metrics never cross engine or partition boundaries.
        adjacent = mask[1:] & mask[:-1] & (engine_ids[1:] == engine_ids[:-1])
        dtrue, dpred = np.diff(truth, axis=0)[adjacent], np.diff(latent, axis=0)[adjacent]
        for j, name in enumerate(COMPONENTS):
            components[name]["delta_mae"] = float(np.mean(abs(dpred[:, j]-dtrue[:, j])))
        result[section] = {
            "n": int(mask.sum()), "components": components,
            "reconstruction": regression_metrics(target[mask], reconstructed[mask], PARAMETERS),
            "clean_response": regression_metrics(truth[mask] @ matrix, reconstructed[mask], PARAMETERS),
            "observable_state_rmse": float(np.sqrt(np.mean((error @ observable)**2))),
            "nullspace_state_rmse": float(np.sqrt(np.mean((error @ (np.eye(5)-observable))**2)))}
    return result


def maintenance_metrics(truth: np.ndarray, predicted: np.ndarray,
                        events: list[int], split: np.ndarray, width: int = 3) -> list[dict]:
    """Delta = mean[e:e+w] - mean[e-w:e]; negative means performance recovery.

    Event uses future observations only for retrospective evaluation of a saved
    causal prediction series, never for individual predictions or training.
    """
    if width < 1 or truth.shape != predicted.shape or truth.shape[1] != 5:
        raise ValueError("Invalid maintenance inputs.")
    records = []
    for event in events:
        if event-width < 0 or event+width > len(truth):
            raise ValueError("Maintenance window extends beyond the series.")
        before, after = slice(event-width, event), slice(event, event+width)
        delta = truth[after].mean(0)-truth[before].mean(0)
        estimated = predicted[after].mean(0)-predicted[before].mean(0)
        labels = set(split[event-width:event+width])
        section = labels.pop() if len(labels) == 1 else "boundary"
        for j, name in enumerate(COMPONENTS):
            records.append({"event_index": int(event), "window_width": width,
                            "before_start": event-width, "after_stop_exclusive": event+width,
                            "split": section, "component": name,
                            "true_delta": float(delta[j]), "predicted_delta": float(estimated[j]),
                            "true_recovery": float(-delta[j]), "predicted_recovery": float(-estimated[j]),
                            "recovery_absolute_error": float(abs(delta[j]-estimated[j])),
                            "direction_correct": bool(np.sign(delta[j]) == np.sign(estimated[j]))})
    return records
