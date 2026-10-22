"""Export computed metrics and publication-style Synthetic Data diagnostics."""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .physics import COMPONENTS, PARAMETERS

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.2, "savefig.dpi": 160})


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def prediction_frame(frame, processed, truth, latent, reconstructed, split, engine_id):
    result = pd.DataFrame({"engine_id": engine_id, "date": frame["date"], "split": split})
    for j, name in enumerate(COMPONENTS):
        result[f"true_{name}"] = truth[name].to_numpy()
        result[f"estimated_{name}"] = latent[:, j]
    for j, name in enumerate(PARAMETERS):
        result[f"observed_{name}"] = processed[name].to_numpy()
        result[f"reconstructed_{name}"] = reconstructed[:, j]
    return result


def plot_results(output: Path, predictions: pd.DataFrame, history: list[dict],
                 metrics: dict, events: list[int], maintenance: pd.DataFrame,
                 mode: str) -> None:
    directory = output / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(predictions))

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(directory / name)
        plt.close(fig)

    def boundaries(ax):
        for event in events:
            ax.axvline(event, color="#555555", linestyle=":", linewidth=0.9)
        labels = predictions["split"].to_numpy()
        for section, color in (("valid", "#e9d8a6"), ("test", "#cce3de")):
            idx = np.flatnonzero(labels == section)
            if len(idx):
                ax.axvspan(idx[0], idx[-1], color=color, alpha=0.25)

    fig, axes = plt.subplots(5, 1, figsize=(10, 10), sharex=True)
    for name, ax in zip(COMPONENTS, axes):
        ax.plot(x, predictions[f"true_{name}"], label="Synthetic truth", color="#202020")
        ax.plot(x, predictions[f"estimated_{name}"], label="Estimated", color="#0077b6")
        boundaries(ax)
        ax.set_ylabel(name+"\nstate [s.u.]")
    axes[0].set_title(f"Synthetic Data | {mode} | engine 0 | larger state = worse degradation")
    axes[0].legend(ncol=2)
    axes[-1].set_xlabel("Record index | dotted: maintenance; amber: validation; green: test")
    save(fig, "component_degradation.png")

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for name, ax in zip(PARAMETERS, axes):
        ax.plot(x, predictions[f"observed_{name}"], label="Fingerprint-corrected, EEMD-NLM observation", alpha=0.8)
        ax.plot(x, predictions[f"reconstructed_{name}"], label="Fingerprint reconstruction")
        boundaries(ax)
        ax.set_ylabel(name+" [s.u.]")
    axes[0].set_title(f"Synthetic Data | {mode} | original synthetic signal units, no plot rescaling")
    axes[0].legend()
    axes[-1].set_xlabel("Record index")
    save(fig, "signal_reconstruction.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    for section in ("train", "val"):
        ax.plot([h["epoch"] for h in history], [h[f"{section}_loss"] for h in history], label=section)
    ax.set(title=f"Synthetic Data | {mode}", xlabel="Epoch", ylabel="Dimensionless weighted objective")
    ax.legend()
    save(fig, "training_history.png")

    fig, ax = plt.subplots(figsize=(9, 4))
    positions = np.arange(5)
    ax.bar(positions, [metrics["test"]["components"][c]["mae"] for c in COMPONENTS])
    ax.set(xticks=positions, xticklabels=COMPONENTS,
           ylabel="Test MAE [synthetic state units]",
           title="Synthetic Data | model evaluation")
    save(fig, "component_mae.png")

    if len(maintenance):
        primary = maintenance[maintenance.engine_id == 0]
        fig, axes = plt.subplots(1, len(events), figsize=(5*len(events), 4), squeeze=False)
        for event, ax in zip(events, axes[0]):
            subset = primary[primary.event_index == event].set_index("component").loc[list(COMPONENTS)]
            ax.bar(positions-0.18, subset.true_recovery, 0.36, label="True recovery")
            ax.bar(positions+0.18, subset.predicted_recovery, 0.36, label="Predicted recovery")
            ax.set(xticks=positions, xticklabels=COMPONENTS, ylabel="Recovery [s.u.]; positive = improvement",
                   title=f"Synthetic Data | event {event} ({subset['split'].iloc[0]})")
            ax.legend()
        save(fig, "maintenance_recovery.png")

