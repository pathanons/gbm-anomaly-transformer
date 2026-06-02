from __future__ import annotations

import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        enc_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            n_features,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=enc_dropout,
        )
        self.decoder = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=enc_dropout,
        )
        self.output = nn.Linear(hidden_dim, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        _, (hidden, cell) = self.encoder(x)
        decoder_input = hidden[-1].unsqueeze(1).expand(batch_size, seq_len, self.hidden_dim)
        decoded, _ = self.decoder(decoder_input, (hidden, cell))
        return self.output(decoded)


class MLPAutoencoder(nn.Module):
    def __init__(
        self,
        window_size: int,
        n_features: int,
        latent_dim: int = 32,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.window_size = window_size
        self.n_features = n_features
        flat_dim = window_size * n_features
        self.encoder = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, flat_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        flat = x.reshape(batch_size, -1)
        latent = self.encoder(flat)
        recon = self.decoder(latent)
        return recon.reshape(batch_size, self.window_size, self.n_features)


class CNN1DLSTMAutoencoder(nn.Module):
    """1D CNN feature extractor + LSTM sequence autoencoder."""

    def __init__(
        self,
        n_features: int,
        cnn_channels: tuple[int, ...] = (32, 64),
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        padding = kernel_size // 2

        encoder_cnn_layers: list[nn.Module] = []
        in_channels = n_features
        for out_channels in cnn_channels:
            encoder_cnn_layers.extend(
                [
                    nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
                    nn.ReLU(),
                    nn.BatchNorm1d(out_channels),
                ]
            )
            in_channels = out_channels
        self.encoder_cnn = nn.Sequential(*encoder_cnn_layers)
        self.cnn_out_channels = in_channels

        enc_dropout = dropout if num_layers > 1 else 0.0
        self.encoder_lstm = nn.LSTM(
            self.cnn_out_channels,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=enc_dropout,
        )
        self.decoder_lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=enc_dropout,
        )
        self.to_cnn = nn.Linear(hidden_dim, self.cnn_out_channels)

        decoder_cnn_layers: list[nn.Module] = []
        rev_channels = list(cnn_channels)[::-1]
        in_channels = self.cnn_out_channels
        for out_channels in rev_channels[1:] + [n_features]:
            decoder_cnn_layers.extend(
                [
                    nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
                    nn.ReLU(),
                ]
            )
            if out_channels != n_features:
                decoder_cnn_layers.append(nn.BatchNorm1d(out_channels))
            in_channels = out_channels
        self.decoder_cnn = nn.Sequential(*decoder_cnn_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        encoded = self.encoder_cnn(x.permute(0, 2, 1)).permute(0, 2, 1)
        _, (hidden, cell) = self.encoder_lstm(encoded)
        decoder_input = hidden[-1].unsqueeze(1).expand(batch_size, seq_len, self.hidden_dim)
        decoded, _ = self.decoder_lstm(decoder_input, (hidden, cell))
        decoded = self.to_cnn(decoded).permute(0, 2, 1)
        return self.decoder_cnn(decoded).permute(0, 2, 1)
