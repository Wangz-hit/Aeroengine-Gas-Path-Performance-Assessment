"""TCN-LSTM-MLP encoder and Fingerprint-constrained affine decoder."""

import torch
from torch import nn
from .physics import FingerprintDecoder


class CausalConvBlock(nn.Module):
    """Causal two-layer residual convolution with exponentially dilated filters."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int, dropout: float):
        super().__init__()
        self.pad = (3 - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, dilation=dilation)
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(nn.functional.pad(x, (self.pad, 0)))
        hidden = self.dropout(self.activation(hidden))
        hidden = self.conv2(nn.functional.pad(hidden, (self.pad, 0)))
        return self.activation(self.skip(x) + self.dropout(hidden))


class TemporalConvNet(nn.Module):
    def __init__(self, input_channels: int = 3, channels: tuple[int, ...] = (16, 16),
                 dropout: float = 0.05):
        super().__init__()
        blocks = []
        c = input_channels
        for index, next_channels in enumerate(channels):
            blocks.append(CausalConvBlock(c, next_channels, 2**index, dropout))
            c = next_channels
        self.network = nn.Sequential(*blocks)
        self.out_channels = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DTBAutoEncoder(nn.Module):
    """3 gas-path inputs -> 5 component states -> 3 reconstructed inputs."""

    def __init__(self, conv_channels: tuple[int, ...] = (16, 16), lstm_hidden: int = 24,
                 mlp_hidden: int = 32, dropout: float = 0.05):
        super().__init__()
        if not conv_channels or min(*conv_channels, lstm_hidden, mlp_hidden) < 1 or not 0 <= dropout < 1:
            raise ValueError("Positive layer sizes and dropout in [0,1) are required.")
        self.model_config = dict(conv_channels=list(conv_channels), lstm_hidden=lstm_hidden,
                                 mlp_hidden=mlp_hidden, dropout=dropout)
        self.register_buffer("input_mean", torch.zeros(3))
        self.register_buffer("input_scale", torch.ones(3))
        self.tcn = TemporalConvNet(3, conv_channels, dropout)
        self.lstm = nn.LSTM(input_size=self.tcn.out_channels,
                            hidden_size=lstm_hidden, num_layers=1, batch_first=True)
        self.encoder = nn.Sequential(
            nn.Linear(lstm_hidden, mlp_hidden), nn.ELU(),
            nn.Dropout(dropout), nn.Linear(mlp_hidden, 5)
        )
        self.decoder = FingerprintDecoder()
        nn.init.zeros_(self.encoder[-1].bias)

    def fit_input_scaler(self, training_targets: torch.Tensor) -> None:
        """Fit only on training observations; decoder remains in original units."""
        if training_targets.ndim != 2 or training_targets.shape[1] != 3:
            raise ValueError("Scaler requires [N,3] training observations.")
        if not torch.isfinite(training_targets).all() or len(training_targets) < 2:
            raise ValueError("Scaler requires at least two finite observations.")
        self.input_mean.copy_(training_targets.mean(0))
        self.input_scale.copy_(training_targets.std(0, unbiased=False).clamp(min=0.02))

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 3 or inputs.shape[-1] != 3:
            raise ValueError("Expected input shape [batch, sequence_length, 3].")
        features = (inputs-self.input_mean)/self.input_scale
        features = self.tcn(features.transpose(1, 2)).transpose(1, 2)
        features, _ = self.lstm(features)
        degradation = self.encoder(features)
        reconstruction = self.decoder(degradation)
        return degradation, reconstruction
