"""LSTM and GRU sequence classifiers over per-frame joint-angle channels.

Both share the same shape: recurrent layer(s) -> take the final hidden state -> linear
classifier head. Kept as two separate small classes (rather than one class with an
`rnn_type` flag) because the constructor signatures genuinely differ enough
(nn.LSTM vs nn.GRU) that a flag would just be an if/else hiding in __init__ for no
real benefit.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]  # (batch, hidden_size) from the final layer
        return self.classifier(last_hidden)


class GRUClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(x)
        last_hidden = h_n[-1]
        return self.classifier(last_hidden)
