"""
KidokuPolicyNet

A ResNet-style convolutional neural network for behavioral cloning of chess moves.

Architecture:
    - Input:  14 x 8 x 8 board tensor
    - Backbone: 6 Residual Blocks (deeper than before for better pattern recognition)
    - Policy Head: Predicts logits over 4672 possible moves (fixed encoding)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Standard ResNet residual block with two 3x3 convolutions."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class KidokuPolicyNet(nn.Module):
    """
    Policy network that predicts the probability distribution over moves
    given a chess position.

    Output size = 4672 (matches our fixed move encoding in process_pgn.py)
    """
    def __init__(self, num_moves: int = 4672):
        super().__init__()

        # Initial convolution
        self.stem = nn.Sequential(
            nn.Conv2d(14, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Residual tower (6 blocks = reasonable depth for this task)
        self.res_blocks = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),   # Reduce channels
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_moves)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Board tensor of shape (B, 14, 8, 8)
        Returns:
            Logits of shape (B, 4672)
        """
        x = self.stem(x)
        x = self.res_blocks(x)
        logits = self.policy_head(x)
        return logits


def count_parameters(model: nn.Module) -> int:
    """Helper to count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = KidokuPolicyNet()
    print(f"Model created with {count_parameters(model):,} trainable parameters")
    print(model)
