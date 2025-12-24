import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


class ViTEncoder(nn.Module):
    def __init__(
        self, embed_size: int = 512, pretrained: bool = True, freeze: bool = True
    ):
        super().__init__()
        if pretrained:
            weights = ViT_B_16_Weights.DEFAULT
            self.vit = vit_b_16(weights=weights)
        else:
            self.vit = vit_b_16(weights=None)

        # Freeze encoder if requested
        if freeze:
            for param in self.vit.parameters():
                param.requires_grad = False

        # Modify final head (always trainable)
        # ViT uses 'heads.head' instead of 'fc'
        num_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(num_features, out_features=embed_size)

    def forward(self, x):
        return self.vit(x)  # [batch, embed_size]
