from __future__ import annotations

import torch.nn as nn


class MetadataOnlyRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...] = (64, 32), dropout: float = 0.1):
        super().__init__()
        layers = []
        previous_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.net(inputs)
