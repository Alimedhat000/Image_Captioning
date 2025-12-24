import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class SpatialResNet50Encoder(nn.Module):
    def __init__(
        self, embed_size: int = 512, pretrained: bool = True, freeze: bool = True
    ):
        super().__init__()
        if pretrained:
            weights = ResNet50_Weights.DEFAULT
            full_resnet = resnet50(weights=weights)
        else:
            full_resnet = resnet50(weights=None)

        # 1. Remove the last two layers: AvgPool and FC
        self.resnet = nn.Sequential(*list(full_resnet.children())[:-2])

        # 2. Freeze encoder if requested
        if freeze:
            for param in self.resnet.parameters():
                param.requires_grad = False

        # 3. Create a projection layer to reach your desired embed_size
        self.embed_projection = nn.Linear(2048, embed_size)

    def forward(self, x):
        # x shape: [batch, 3, 224, 224]
        features = self.resnet(x)
        # features shape: [batch, 2048, 7, 7]

        # 4. Reshape to get the "49 feature vectors" (7x7=49)
        # We permute so that the 2048 (features) is the last dimension
        features = features.permute(0, 2, 3, 1)
        # New shape: [batch, 7, 7, 2048]

        features = features.view(features.size(0), -1, features.size(3))
        # Final shape: [batch, 49, 2048]

        # 5. Project each of the 49 vectors to your embed_size
        embeddings = self.embed_projection(features)
        # Final shape: [batch, 49, embed_size]

        return embeddings
