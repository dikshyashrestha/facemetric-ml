
"""
Shared ResNet-18 Face Embedding Backbone

Applied Deep Learning - Project 1

Input:
    RGB face image: [batch_size, 3, 112, 112]

Output:
    Face embedding: [batch_size, 512]

Architecture:
    Modified ResNet-18
    -> BatchNorm1d
    -> Dropout(0.4)
    -> Fully Connected (512)
    -> BatchNorm1d

Shared across all five deep metric learning methods.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18


class FaceEmbeddingNet(nn.Module):

    def __init__(self, embedding_dim=512, dropout=0.4):
        super().__init__()

        # Initialize ResNet-18 without pretrained weights.
        backbone = resnet18(weights=None)

        # Adapt the first convolution for 112x112 input.
        backbone.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        # Remove the initial max-pooling layer.
        backbone.maxpool = nn.Identity()

        # Get the backbone's feature dimension.
        feature_dim = backbone.fc.in_features

        # Remove the original classification layer.
        backbone.fc = nn.Identity()

        self.backbone = backbone

        # Required embedding head: BN -> Dropout -> FC -> BN
        self.embedding_head = nn.Sequential(
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

    def forward(self, x, normalize=False):

        # Extract features using ResNet-18.
        features = self.backbone(x)

        # Generate 512-dimensional embeddings.
        embeddings = self.embedding_head(features)

        # Optional L2 normalization.
        if normalize:
            embeddings = F.normalize(
                embeddings,
                p=2,
                dim=1
            )

        return embeddings


if __name__ == "__main__":

    # Test the architecture with dummy images.
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    model = FaceEmbeddingNet().to(device)
    model.eval()

    # Two RGB images with resolution 112x112.
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
