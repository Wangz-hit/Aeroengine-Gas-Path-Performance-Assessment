"""Train/validation optimization and self-contained, inference-only checkpoints."""

from dataclasses import asdict, dataclass
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from .model import DTBAutoEncoder
from .losses import physics_objective


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 0.002
    patience: int = 20
    seed: int = 17
    log_every: int = 10

    def __post_init__(self):
        if min(self.epochs, self.batch_size, self.patience, self.log_every) < 1:
            raise ValueError("Epochs, batch size, patience and log interval must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive.")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def fit_model(model: DTBAutoEncoder, train_set, valid_set,
              cfg: TrainConfig, checkpoint: Path, device: str = "cpu") -> list[dict]:
    """Optimize paper reconstruction and degradation terms without state labels."""
    seed_everything(cfg.seed)
    model.fit_input_scaler(train_set.targets)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    generator = torch.Generator().manual_seed(cfg.seed)
    loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(valid_set, batch_size=cfg.batch_size, shuffle=False)
    history, best, since_best = [], float("inf"), 0

    def pass_data(data_loader, training):
        model.train(training)
        totals = dict.fromkeys(("loss", "reconstruction", "sensitivity"), 0.0)
        count = 0
        with torch.set_grad_enabled(training):
            for batch in data_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                if training:
                    optimizer.zero_grad(set_to_none=True)
                loss, parts = physics_objective(model, batch["windows"], batch["target"])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training/validation loss.")
                if training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0, error_if_nonfinite=True)
                    optimizer.step()
                size = len(batch["windows"])
                count += size
                for key, value in parts.items():
                    totals[key] += value*size
        if not count:
            raise ValueError("Training/validation dataset is empty.")
        return {key: value/count for key, value in totals.items()}

    for epoch in range(1, cfg.epochs+1):
        train = pass_data(loader, True)
        valid = pass_data(val_loader, False)
        history.append({"epoch": epoch, **{f"train_{k}": v for k, v in train.items()},
                        **{f"val_{k}": v for k, v in valid.items()}})
        if epoch == 1 or epoch % cfg.log_every == 0:
            print(f"Epoch {epoch:03d} | train={train['loss']:.5f} | valid={valid['loss']:.5f}")
        if valid["loss"] < best-1e-6:
            best, since_best = valid["loss"], 0
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state": model.state_dict(), "model_config": model.model_config,
                        "training_config": asdict(cfg), "best_epoch": epoch,
                        "best_validation_loss": best}, checkpoint)
        else:
            since_best += 1
            if since_best >= cfg.patience:
                print(f"Early stopping at epoch {epoch}; best validation={best:.5f}.")
                break
    data = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(data["model_state"])
    model.eval()
    return history


def load_model(checkpoint: Path, device: str = "cpu") -> DTBAutoEncoder:
    data = torch.load(checkpoint, map_location=device, weights_only=True)
    model = DTBAutoEncoder(**data["model_config"]).to(device)
    model.load_state_dict(data["model_state"])
    return model.eval()


@torch.no_grad()
def predict(model: DTBAutoEncoder, windows: np.ndarray,
            batch_size: int = 128, device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    if len(windows) == 0 or batch_size < 1:
        raise ValueError("Nonempty windows and positive batch size required.")
    model.eval().to(device)
    latent_list, recon_list = [], []
    for start in range(0, len(windows), batch_size):
        batch = torch.as_tensor(windows[start:start+batch_size], dtype=torch.float32, device=device)
        degradation, reconstruction = model(batch)
        latent_list.append(degradation[:, -1].cpu().numpy())
        recon_list.append(reconstruction[:, -1].cpu().numpy())
    latent, recon = np.concatenate(latent_list), np.concatenate(recon_list)
    if not np.isfinite(latent).all() or not np.isfinite(recon).all():
        raise FloatingPointError("Non-finite predictions.")
    return latent, recon
