"""Model architectures compared in this study. See README.md for the full comparison table"""

from __future__ import annotations

import torch
from torch import nn
from attention_mts.attention import PositionalEncoding, TransformerEncoderBlock


class PooledLinearBaseline(nn.Module):
    """Mean-pool over time, then a single linear layer. No temporal modelling at all."""

    def __init__(self, n_channels: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(n_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, time) -> pooled: (batch, channels) -> logits: (batch, 1)
        pooled = x.mean(dim=2)
        return self.classifier(pooled)

class CNNBaseline(nn.Module):
    def __init__(self, n_channels: int, n_mid: int, n_hidden: int, kernel_size: int = 7) -> None:
        super().__init__()
        self.project = nn.Conv1d(n_channels, n_mid, kernel_size=1)
        self.block1 = nn.Sequential(
            nn.Conv1d(n_mid, n_hidden, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(n_hidden), nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(n_hidden, n_hidden, kernel_size, padding = kernel_size // 2),
            nn.BatchNorm1d(n_hidden), nn.ReLU(),
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(n_hidden, n_hidden, kernel_size, padding = kernel_size // 2),
            nn.BatchNorm1d(n_hidden), nn.ReLU(),
        )
        self.classifier = nn.Linear(n_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        pooled = x.mean(dim=2)
        return self.classifier(pooled)


class TransformerClassifier(nn.Module):
    def __init__(
        self, n_channels: int, d_model: int, n_heads: int, d_ff: int, n_layers: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.embedding = nn.Linear(n_channels, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.blocks = nn.ModuleList(
            [TransformerEncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, time) -> (batch, time, channels), so time steps are tokens
        x = x.transpose(1,2)
        x = self.embedding(x)        # (batch, time, d_model)
        x = self.pos_encoding(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        pooled = x.mean(dim=1)        # mean over time (not dim=2 anymore - time is now dim 1)
        return self.classifier(pooled)

class BiLSTMBaseline(nn.Module):
    def __init__(self, n_channels: int, hidden_size: int) -> None:
        super().__init__()
        self.lstm=nn.LSTM(
            input_size=n_channels, hidden_size=hidden_size,
            num_layers=1, batch_first=True, bidirectional=True
        )
        self.classifier = nn.Linear(hidden_size * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, time) -> (batch, time, channels)
        x = x.transpose(1,2)              
        outputs, _ = self.lstm(x)         # outputs: (batch, time, hidden_size*2)
        pooled = outputs.mean(dim=1)
        return self.classifier(pooled)

class CNNTransformerHybrid(nn.Module):
    def __init__(
            self, n_channels: int, n_mid: int, d_model: int, n_heads: int, d_ff: int, n_layers: int,
            kernel_size: int = 7, downsample_factor: int = 2, dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.project = nn.Conv1d(n_channels, n_mid, kernel_size=1)
        self.downsample = nn.Sequential(
            nn.Conv1d(n_mid, d_model, kernel_size, stride = downsample_factor, padding=kernel_size // 2),
            nn.BatchNorm1d(d_model), nn.ReLU(),
        )
        self.pos_encoding = PositionalEncoding(d_model)
        self.blocks = nn.ModuleList(
            [TransformerEncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project(x)          # (batch, n_mid, time)
        x = self.downsample(x)       # (batchm d_model, time // downsample_factor)
        x = x.transpose(1, 2)        # (batch, time', d_model) - tokens are now downsampled chunks
        x = self.pos_encoding(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        pooled = x.mean(dim=1)
        return self.classifier(pooled)
    
