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

        # Remove the last two layers: AvgPool and FC
        self.resnet = nn.Sequential(*list(full_resnet.children())[:-2])

        # Freeze encoder if requested
        if freeze:
            for param in self.resnet.parameters():
                param.requires_grad = False

        # Create a projection layer to reach your desired embed_size
        self.embed_projection = nn.Linear(2048, embed_size)

    def forward(self, x):
        # x shape: [batch, 3(channels), 224(w), 224(h)]
        features = self.resnet(x)
        # features shape: [batch, 2048(channels), 7(w), 7(h)]

        # We permute so that the 2048 (features) is the last dimension
        features = features.permute(0, 2, 3, 1)
        # New shape: [batch, 7, 7, 2048]

        BATCH, WIDTH, HIEGHT, CHANNELS = features.size()

        # Reshape to get 49 feature vectors
        # Note -1 automatically multiplies WIDTH * HEIGHT to 49
        features = features.view(BATCH, -1, CHANNELS)
        # Final shape: [batch, 49, 2048]

        # Project to embed_size
        embeddings = self.embed_projection(features)
        # Final shape: [batch, 49, embed_size]

        return embeddings
