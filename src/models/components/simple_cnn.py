"""Simple CNN architecture for MNIST."""

import torch.nn as nn


class SimpleCNN(nn.Module):
    """Simple CNN with 2 conv layers and 2 fc layers."""

    def __init__(self, input_channels: int = 1, num_classes: int = 10):
        """Initialize SimpleCNN.

        Args:
            input_channels: Number of input channels (1 for grayscale, 3 for RGB)
            num_classes: Number of output classes
        """
        super().__init__()

        self.features = nn.Sequential(
            # Conv layer 1: 28x28x1 -> 28x28x32
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 28x28 -> 14x14
            # Conv layer 2: 14x14x32 -> 14x14x64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 14x14 -> 7x7
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        """Forward pass."""
        x = self.features(x)
        x = self.classifier(x)
        return x
