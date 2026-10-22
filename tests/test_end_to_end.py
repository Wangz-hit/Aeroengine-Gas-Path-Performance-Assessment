from pathlib import Path
import json
import pandas as pd
from aero_gas_path.demo import run_demo


def test_demo_runs_and_saves_artifacts(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    metrics = run_demo(root/"configs/default.yaml", tmp_path, epochs=2, samples=96, engines=2)
    assert metrics["benchmark"] == "synthetic" and metrics["mode"] == "physics-only"
    for path in ["data/monitoring.csv", "data/synthetic_truth.csv", "data/processed.csv",
                 "checkpoints/model.pt", "predictions.csv", "metrics.json",
                 "training_history.csv", "figures/component_degradation.png",
                 "figures/signal_reconstruction.png", "figures/training_history.png",
                 "figures/maintenance_recovery.png", "figures/component_mae.png",
                 "fleet_predictions.csv", "maintenance.csv", "splits.csv", "run_manifest.json"]:
        assert (tmp_path/path).is_file(), path
    frame = pd.read_csv(tmp_path/"predictions.csv")
    assert len(frame) == 96
    assert len([c for c in frame if c.startswith("estimated_")]) == 5
    assert len([c for c in frame if c.startswith("reconstructed_")]) == 3
    fleet = pd.read_csv(tmp_path/"fleet_predictions.csv")
    assert len(fleet) == 192 and fleet.engine_id.nunique() == 2
    assert metrics["models"]["physics-only"]["test"]["n"] == 48
    assert json.loads((tmp_path/"metrics.json").read_text())["benchmark"] == "synthetic"
