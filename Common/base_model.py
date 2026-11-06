
# Shared ResNet-18 face embedding network.

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18


class FaceEmbeddingNet(nn.Module):

    def __init__(self, embedding_dim=512, dropout=0.4):
        super().__init__()

        backbone = resnet18(weights=None)

        # A 3x3, stride-1 stem retains resolution for 112x112 inputs.
        backbone.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        backbone.maxpool = nn.Identity()

        feature_dim = backbone.fc.in_features

        backbone.fc = nn.Identity()

        self.backbone = backbone

        # BN-Dropout-FC-BN embedding head shared by all five methods.
        self.embedding_head = nn.Sequential(
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

    def forward(self, x, normalize=False):

        features = self.backbone(x)

        embeddings = self.embedding_head(features)

        if normalize:
            embeddings = F.normalize(
                embeddings,
                p=2,
                dim=1
            )

        return embeddings


if __name__ == "__main__":

    # Check output shape and unit norms on a small synthetic batch.
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    model = FaceEmbeddingNet().to(device)
    model.eval()

    dummy_images = torch.randn(
        2, 3, 112, 112,
        device=device
    )

    with torch.no_grad():
        embeddings = model(
            dummy_images,
            normalize=True
        )

    print("Device:", device)
    print("Input shape:", dummy_images.shape)
    print("Embedding shape:", embeddings.shape)
    print("Embedding norms:", embeddings.norm(dim=1))
    print(
        "Trainable parameters:",
        sum(p.numel() for p in model.parameters())
    )
