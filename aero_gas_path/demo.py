"""Single paper-inspired physics-constrained training pipeline."""

import argparse
import hashlib
import importlib.metadata
from pathlib import Path
import platform
import time
import numpy as np
import pandas as pd
import torch
import yaml
from .simulation import simulate_engine
from .preprocessing import preprocess_monitoring
from .dataset import chronological_split, rolling_windows, WindowDataset
from .model import DTBAutoEncoder
from .training import TrainConfig, fit_model, predict, seed_everything, load_model
from .evaluation import evaluate_predictions, maintenance_metrics
from .reporting import save_json, prediction_frame, plot_results
from .physics import COMPONENTS, PARAMETERS, identifiability_report


def make_datasets(windows, partitions):
    """Build unlabelled train and validation windows."""
    datasets = []
    for section in ("train", "valid"):
        features = np.concatenate([x[getattr(p, section)] for x, p in zip(windows, partitions)])
        datasets.append(WindowDataset(features, np.arange(len(features))))
    return tuple(datasets)


def run_demo(config_path: Path, output_dir: Path, *, epochs: int | None = None,
             samples: int | None = None, device: str = "cpu",
             seed: int | None = None, engines: int | None = None) -> dict:
    started = time.perf_counter()
    config_path, output_dir = Path(config_path), Path(output_dir)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if seed is not None:
        cfg["seed"] = seed
    if epochs is not None:
        cfg["training"]["epochs"] = epochs
    if samples is not None:
        cfg["simulation"]["n_samples"] = samples
    if engines is not None:
        cfg["engines"] = engines
    if int(cfg["engines"]) < 1:
        raise ValueError("At least one synthetic engine is required.")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable.")
    train_cfg = TrainConfig(**cfg["training"], seed=int(cfg["seed"]))
    seed_everything(train_cfg.seed)
    torch.set_num_threads(int(cfg.get("cpu_threads", 2)))
    output_dir.mkdir(parents=True, exist_ok=True)
    # Save the resolved protocol before any training or test evaluation.
    (output_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    frames, truths, processed_frames, windows, labels, partitions, splits = [], [], [], [], [], [], []
    split_records = []
    data_cfg = dict(cfg["simulation"])
    for engine in range(int(cfg["engines"])):
        data_seed = int(cfg["data_seed"])+engine*1009
        frame, truth = simulate_engine(**data_cfg, seed=data_seed)
        partition = chronological_split(len(frame), cfg["split"]["train"], cfg["split"]["valid"])
        processing_cfg = dict(cfg["preprocessing"])
        if processing_cfg["mode"] == "offline":
            # Never allow bidirectional filtering across a held-out boundary.
            segments = [preprocess_monitoring(frame.iloc[idx], seed=data_seed, **processing_cfg)
                        for idx in (partition.train, partition.valid, partition.test)]
            processed = pd.concat(segments, ignore_index=True)
        else:
            processed = preprocess_monitoring(frame, seed=data_seed, **processing_cfg)
        features = processed[list(PARAMETERS)].to_numpy(np.float32)
        values = truth[list(COMPONENTS)].to_numpy(np.float32)
        split = np.full(len(frame), "test", dtype="<U5")
        split[partition.train], split[partition.valid] = "train", "valid"
        directory = output_dir / "data" if engine == 0 else output_dir / "data" / f"engine_{engine:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        for name, table in (("monitoring", frame), ("synthetic_truth", truth), ("processed", processed)):
            table.to_csv(directory / f"{name}.csv", index=False)
        for section in ("train", "valid", "test"):
            idx = getattr(partition, section)
            split_records.append({"engine_id": engine, "split": section, "start": int(idx[0]),
                                  "stop_exclusive": int(idx[-1]+1), "count": len(idx),
                                  "start_date": frame.date.iloc[idx[0]], "end_date": frame.date.iloc[idx[-1]],
                                  "data_seed": data_seed})
        frames.append(frame)
        truths.append(truth)
        processed_frames.append(processed)
        windows.append(rolling_windows(features, int(cfg["model"]["sequence_length"])))
        labels.append(values)
        partitions.append(partition)
        splits.append(split)
    pd.DataFrame(split_records).to_csv(output_dir / "splits.csv", index=False)
    train_set, valid_set = make_datasets(windows, partitions)
    network_cfg = {key: value for key, value in cfg["model"].items() if key != "sequence_length"}
    model = DTBAutoEncoder(**network_cfg)
    checkpoint = output_dir / "checkpoints" / "model.pt"
    history = fit_model(model, train_set, valid_set, train_cfg, checkpoint, device=device)
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    all_windows = np.concatenate(windows)
    latent, reconstructed = predict(model, all_windows, device=device)
    reloaded = load_model(checkpoint, device)
    check_z, check_y = predict(reloaded, all_windows[:16], device=device)
    np.testing.assert_allclose(check_z, latent[:16], atol=1e-6)
    np.testing.assert_allclose(check_y, reconstructed[:16], atol=1e-6)
    ids = np.repeat(np.arange(len(frames)), [len(frame) for frame in frames])
    reference, split = np.concatenate(labels), np.concatenate(splits)
    target = np.concatenate([table[list(PARAMETERS)].to_numpy(float) for table in processed_frames])
    model_metrics = evaluate_predictions(reference, target, latent, reconstructed, split, ids)
    prediction_tables, maintenance_tables = [], []
    events = [round(f*len(frames[0])) for f in cfg["simulation"]["maintenance"]]
    offset = 0
    for engine, (frame, processed, truth, section) in enumerate(zip(frames, processed_frames, truths, splits)):
        stop = offset+len(frame)
        table = prediction_frame(frame, processed, truth, latent[offset:stop], reconstructed[offset:stop], section, engine)
        prediction_tables.append(table)
        event_records = maintenance_metrics(labels[engine], latent[offset:stop], events,
                                            section, cfg["evaluation"]["event_width"])
        for record in event_records:
            record["engine_id"], record["model"] = engine, "physics-only"
        maintenance_tables.extend(event_records)
        offset = stop
    prediction_tables[0].to_csv(output_dir / "predictions.csv", index=False)
    pd.concat(prediction_tables, ignore_index=True).to_csv(output_dir / "fleet_predictions.csv", index=False)
    maintenance = pd.DataFrame(maintenance_tables)
    maintenance.to_csv(output_dir / "maintenance.csv", index=False)
    metrics = {"benchmark": "synthetic", "mode": "physics-only", "seed": train_cfg.seed,
               "preprocessing_mode": cfg["preprocessing"]["mode"],
               "note": "Independent synthetic implementation, not reproduction of published in-service results.",
               "label_policy": "Component ground truth is reserved for evaluation and never enters training or checkpoint selection.",
               "models": {"physics-only": model_metrics},
               "identifiability": identifiability_report(),
               "reconstruction_rmse": {s: {c: model_metrics[s]["reconstruction"][c]["rmse"] for c in PARAMETERS}
                                       for s in ("train", "valid", "test")},
               "latent_mae": {s: {c: model_metrics[s]["components"][c]["mae"] for c in COMPONENTS}
                              for s in ("train", "valid", "test")}}
    save_json(output_dir / "metrics.json", metrics)
    save_json(output_dir / "identifiability.json", metrics["identifiability"])
    plot_results(output_dir, prediction_tables[0], history, model_metrics, events,
                 maintenance, "physics-only")
    checkpoint_data = torch.load(checkpoint, weights_only=True, map_location="cpu")
    source_root = Path(__file__).resolve().parent
    manifest = {"python": platform.python_version(), "device": device,
                "versions": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scipy", "torch", "matplotlib", "PyYAML")},
                "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source_root.glob("*.py"))},
                "data_sha256": {str(p.relative_to(output_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted((output_dir / "data").rglob("*.csv"))},
                "config_sha256": hashlib.sha256((output_dir / "config_resolved.yaml").read_bytes()).hexdigest(),
                "best_epoch": checkpoint_data["best_epoch"], "reload_predictions_verified": True,
                "elapsed_seconds": time.perf_counter()-started}
    save_json(output_dir / "run_manifest.json", manifest)
    print(f"Outputs: {output_dir.resolve()} | physics-only | seed={train_cfg.seed}")
    print("Test reconstruction RMSE:", metrics["reconstruction_rmse"]["test"])
    print("Test component MAE:", metrics["latent_mae"]["test"])
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--engines", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()
    run_demo(args.config, args.output_dir, epochs=args.epochs, samples=args.samples,
             device=args.device, seed=args.seed, engines=args.engines)


if __name__ == "__main__":
    main()
