from pathlib import Path
import numpy as np
import torch
import yaml
from aero_gas_path.dataset import rolling_windows, chronological_split
from aero_gas_path.demo import make_datasets
from aero_gas_path.model import DTBAutoEncoder
from aero_gas_path.preprocessing import preprocess_monitoring
from aero_gas_path.simulation import simulate_engine


def test_offline_processing_is_not_mistaken_for_causal():
    frame, _ = simulate_engine(n_samples=96)
    changed = frame.copy()
    changed.loc[60:, "DEGT"] += 0.5
    full = preprocess_monitoring(frame, ensembles=2)
    altered = preprocess_monitoring(changed, ensembles=2)
    assert np.max(abs(full.DEGT.iloc[:60]-altered.DEGT.iloc[:60])) > 1e-6
    assert np.isfinite(full.iloc[:, 1:].to_numpy()).all()


def test_training_scaler_and_engine_windows_are_isolated():
    windows = [rolling_windows(np.full((100, 3), value), 12) for value in (1, 50)]
    split = chronological_split(100, 0.6, 0.15)
    train, valid = make_datasets(windows, [split, split])
    assert (train.windows[:60] == 1).all() and (train.windows[60:] == 50).all()
    assert (valid.windows[:15] == 1).all() and (valid.windows[15:] == 50).all()
    scaler = DTBAutoEncoder()
    scaler.fit_input_scaler(train.targets)
    np.testing.assert_allclose(scaler.input_mean.numpy(), train.targets.mean(dim=0).numpy(), atol=1e-6)


def test_test_observations_and_states_do_not_affect_fitting(tmp_path, monkeypatch):
    import importlib
    demo = importlib.import_module("aero_gas_path.demo")
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root/"configs/default.yaml").read_text())
    config["preprocessing"]["ensembles"] = 2
    config["engines"] = 1
    config["simulation"]["n_samples"] = 120
    config["training"]["epochs"] = 2
    path = tmp_path/"config.yaml"
    path.write_text(yaml.safe_dump(config))
    demo.run_demo(path, tmp_path/"original")
    original_simulate = demo.simulate_engine

    def changed_test(**kwargs):
        measured, truth = original_simulate(**kwargs)
        measured.loc[90:, ["DEGT", "DFF", "DN2"]] += 5
        truth.loc[90:, ["FAN", "LPC", "HPC", "HPT", "LPT"]] += 3
        return measured, truth

    monkeypatch.setattr(demo, "simulate_engine", changed_test)
    demo.run_demo(path, tmp_path/"altered")
    a = torch.load(tmp_path/"original/checkpoints/model.pt", weights_only=True)
    b = torch.load(tmp_path/"altered/checkpoints/model.pt", weights_only=True)
    assert a["best_epoch"] == b["best_epoch"]
    assert a["best_validation_loss"] == b["best_validation_loss"]
    for key in a["model_state"]:
        torch.testing.assert_close(a["model_state"][key], b["model_state"][key], rtol=0, atol=0)
