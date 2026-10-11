from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(channels),
        )
        self.activation = nn.ReLU(inplace=True)
        # A new block starts as an identity mapping and can be appended safely.
        nn.init.zeros_(self.layers[-1].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.layers(x))


class PolicyValueNetwork(nn.Module):
    """A compact policy-value network for a 5x5x5 board.

    Inputs are canonical boards from the side-to-move perspective, with shape
    ``(batch, 1, 5, 5, 5)`` and values in ``{-1, 0, 1}``.
    """

    def __init__(
        self,
        board_size: int = 5,
        num_actions: int = 25,
        channels: int = 32,
        num_residual_blocks: int = 2,
        value_head_channels: int = 1,
        value_hidden_size: int | None = None,
    ) -> None:
        super().__init__()
        self.board_size = board_size
        self.num_actions = num_actions
        self.value_head_channels = value_head_channels
        self.value_hidden_size = value_hidden_size or channels

        self.stem = nn.Sequential(
            nn.Conv3d(1, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
        )
        self.residual_tower = nn.Sequential(
            *(ResidualBlock(channels) for _ in range(num_residual_blocks))
        )

        cells = board_size**3
        self.policy_head = nn.Sequential(
            nn.Conv3d(channels, 2, kernel_size=1, bias=False),
            nn.BatchNorm3d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * cells, num_actions),
        )
        self.value_features = nn.Sequential(
            nn.Conv3d(
                channels, value_head_channels, kernel_size=1, bias=False
            ),
            nn.BatchNorm3d(value_head_channels),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(value_head_channels * cells, self.value_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.value_hidden_size, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim == 4:
            x = x.unsqueeze(1)
        features = self.residual_tower(self.stem(x.float()))
        policy_logits = self.policy_head(features)
        value = self.value_head(self.value_features(features)).squeeze(-1)
        return policy_logits, value
