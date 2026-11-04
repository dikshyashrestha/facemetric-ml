
## Shared ResNet-18 backbone for face embedding models.

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18


class FaceEmbeddingNet(nn.Module):

    def __init__(self, embedding_dim=512, dropout=0.4):
        super().__init__()

        # Use a plain ResNet-18 backbone.
        backbone = resnet18(weights=None)

        # Match the 112x112 input size.
        backbone.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        # Skip the initial max-pooling layer.
        backbone.maxpool = nn.Identity()

        # Get the feature dimension before the classifier.
        feature_dim = backbone.fc.in_features

        # Replace the classifier with a feature extractor.
        backbone.fc = nn.Identity()

        self.backbone = backbone

        # Add the embedding projection head.
        self.embedding_head = nn.Sequential(
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

    def forward(self, x, normalize=False):

        # Run the backbone and produce embeddings.
        features = self.backbone(x)

        # Project to the final embedding size.
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

    # Quick sanity check with dummy images.
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    model = FaceEmbeddingNet().to(device)
    model.eval()

    # Use a small batch of 112x112 RGB images.
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
