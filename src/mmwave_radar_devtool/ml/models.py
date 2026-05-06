"""Neural network models for radar classification and regression."""

from __future__ import annotations

import torch
from torch import nn


class RadarMLP(nn.Module):
    """Simple MLP backbone for flattened radar feature vectors."""

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if output_dim <= 0:
            raise ValueError("output_dim must be > 0")
        if not hidden_dims:
            raise ValueError("hidden_dims must not be empty")

        layers: list[nn.Module] = []
        prev = int(input_dim)
        for width in hidden_dims:
            layers.append(nn.Linear(prev, int(width)))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=float(dropout)))
            prev = int(width)
        layers.append(nn.Linear(prev, int(output_dim)))
        self.network = nn.Sequential(*layers)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return penultimate-layer embeddings for one batch."""
        modules = list(self.network.children())
        if len(modules) <= 1:
            return x
        features = x
        for layer in modules[:-1]:
            features = layer(features)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return model predictions for one batch."""
        features = self.forward_features(x)
        return self.network[-1](features)


__all__ = ["RadarMLP"]
