import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNet50Encoder(nn.Module):
    def __init__(
        self, embed_size: int = 512, pretrained: bool = True, freeze: bool = True
    ):
        super().__init__()
        if pretrained:
            weights = ResNet50_Weights.DEFAULT
            self.resnet = resnet50(weights=weights)
        else:
            self.resnet = resnet50(weights=None)

        # Freeze encoder if requested
        if freeze:
            for param in self.resnet.parameters():
                param.requires_grad = False

        # Modify final fc layer (always trainable)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, out_features=embed_size)

    def forward(self, x):
        return self.resnet(x)  # [batch, embed_size]
