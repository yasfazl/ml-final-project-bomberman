"""Neural Q-network and exact warm-start support for the DQN agent."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class DQN(nn.Module):
    """A small multilayer perceptron mapping features to action values."""

    def __init__(
        self,
        feature_dim: int,
        action_count: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.action_count = int(action_count)
        self.hidden_dim = int(hidden_dim)

        self.network = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_count),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def initialize_from_linear_q(
    model: DQN,
    linear_weights: np.ndarray | torch.Tensor,
) -> None:
    """Make ``model(x)`` exactly equal the supplied linear Q-function.

    A ReLU network can preserve a signed feature ``x`` by representing it as
    ``relu(x) - relu(-x)``.  The first two hidden layers carry those positive
    and negative parts, and the output layer reconstructs the original
    linear action values.  Unused neurons start at zero and remain available
    for later nonlinear learning.
    """
    weights = torch.as_tensor(linear_weights, dtype=torch.float32)
    expected_shape = (model.action_count, model.feature_dim)
    if tuple(weights.shape) != expected_shape:
        raise ValueError(
            f"Expected linear weights with shape {expected_shape}, "
            f"found {tuple(weights.shape)}."
        )

    required_hidden = 2 * model.feature_dim
    if model.hidden_dim < required_hidden:
        raise ValueError(
            f"Exact warm start requires at least {required_hidden} hidden "
            f"units, found {model.hidden_dim}."
        )

    first = model.network[0]
    second = model.network[2]
    output = model.network[4]

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

        feature_indices = torch.arange(model.feature_dim)
        first.weight[feature_indices, feature_indices] = 1.0
        first.weight[
            model.feature_dim + feature_indices,
            feature_indices,
        ] = -1.0

        preserved_indices = torch.arange(required_hidden)
        second.weight[preserved_indices, preserved_indices] = 1.0

        output.weight[:, :model.feature_dim] = weights
        output.weight[
            :,
            model.feature_dim:required_hidden,
        ] = -weights

