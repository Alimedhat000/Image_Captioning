import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNet50Encoder(nn.Module):
    def __init__(self, embed_size: int = 512, pretrained: bool = True):
        super().__init__()

        if pretrained:
            weights = ResNet50_Weights.DEFAULT
            self.resnet = resnet50(weights=weights)
        else:
            self.resnet = resnet50(weights=None)

        # Modify final fc layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, out_features=embed_size)

    def forward(self, x):
        return self.resnet(x)
