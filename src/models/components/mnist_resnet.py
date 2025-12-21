"""ResNet adapted for MNIST (grayscale 28x28 images)."""

import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class MNISTResNet18(nn.Module):
    """ResNet18 adapted for MNIST.

    Modifications:
    - First conv layer adapted for 1-channel input (grayscale)
    - Smaller input size (28x28 instead of 224x224)
    """

    def __init__(self, num_classes: int = 10, pretrained: bool = False):
        """Initialize MNIST ResNet18.

        Args:
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights (not recommended for MNIST)
        """
        super().__init__()

        # Load ResNet18
        if pretrained:
            weights = ResNet18_Weights.DEFAULT
            self.resnet = resnet18(weights=weights)
        else:
            self.resnet = resnet18(weights=None)

        # Modify first conv layer for grayscale input (1 channel instead of 3)
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Modify final fully connected layer for num_classes
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        """Forward pass."""
        return self.resnet(x)


class MNISTResNet50(nn.Module):
    """ResNet50 adapted for MNIST."""

    def __init__(self, num_classes: int = 10, pretrained: bool = False):
        """Initialize MNIST ResNet50."""
        super().__init__()

        from torchvision.models import resnet50, ResNet50_Weights

        if pretrained:
            weights = ResNet50_Weights.DEFAULT
            self.resnet = resnet50(weights=weights)
        else:
            self.resnet = resnet50(weights=None)

        # Modify first conv layer for grayscale
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Modify final fc layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.resnet(x)
