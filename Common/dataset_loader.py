
import csv
from collections import defaultdict
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from .dataset_transforms import get_train_transform, get_eval_transform


# Resolve paths relative to the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = PROJECT_ROOT / "Dataset"
IMAGE_ROOT = DATASET_ROOT / "Processed" / "deepfunneled_112"
MANIFEST_ROOT = DATASET_ROOT / "Manifests"


def read_manifest(filename):
    """Read a generated CSV manifest."""

    path = MANIFEST_ROOT / filename

    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")

    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def load_image(relative_path):
    """Load an RGB image from the processed image cache."""

    path = IMAGE_ROOT / relative_path

    with Image.open(path) as image:
        return image.convert("RGB")


class LFWIdentityDataset(Dataset):
    """
    Identity-based LFW dataset.

    Used for:
        - Triplet Loss
        - Center Loss
        - SphereFace
        - ArcFace
        - Softmax baseline

    Each sample returns:
        image_tensor, identity_label
    """

    def __init__(
        self,
        split="view1_train",
        held_out_folds=None,
        train=True
    ):

        self.transform = (
            get_train_transform()
            if train else get_eval_transform()
        )

        if split == "view1_train":
            if held_out_folds is not None:
                raise ValueError(
                    "held_out_folds is only used for View 2."
                )

            self.samples = read_manifest(
                "view1_train_images.csv"
            )

        elif split == "view2":
            if not held_out_folds:
                raise ValueError(
                    "View 2 training requires held_out_folds."
                )

            held_out = set(held_out_folds)

            if not held_out.issubset(set(range(1, 11))):
                raise ValueError("Fold numbers must be 1-10.")

            all_samples = read_manifest(
                "view2_images.csv"
            )

            # Never train on held-out identities.
            self.samples = [
                row for row in all_samples
                if int(row["fold"]) not in held_out
            ]

        else:
            raise ValueError(f"Unknown split: {split}")

        if not self.samples:
            raise ValueError("No training samples found.")

        # Assign a numeric label to each identity.
        identities = sorted({
            row["identity"] for row in self.samples
        })

        self.identity_to_label = {
            identity: index
            for index, identity in enumerate(identities)
        }

        self.labels = [
            self.identity_to_label[row["identity"]]
            for row in self.samples
        ]

        # Useful for Triplet Loss P x K sampling.
        self.indices_by_identity = defaultdict(list)

        for index, label in enumerate(self.labels):
            self.indices_by_identity[label].append(index)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        row = self.samples[index]

        image = load_image(row["image_path"])
        image = self.transform(image)

        label = self.labels[index]

        return image, label


class LFWPairDataset(Dataset):
    """
    Pair-based LFW dataset.

    Used for:
        - Siamese training
        - View 1 validation
        - View 2 verification

    Returns:
        image1, image2, label, fold

    label = 1: same person
    label = 0: different people
    """

    MANIFESTS = {
        "view1_train": "view1_train_pairs.csv",
        "view1_validation": "view1_validation_pairs.csv",
        "view2": "view2_pairs.csv",
    }

    def __init__(
        self,
        split="view1_validation",
        fold=None,
        train=False
    ):

        if split not in self.MANIFESTS:
            raise ValueError(f"Unknown split: {split}")

        self.samples = read_manifest(
            self.MANIFESTS[split]
        )

        if split == "view2":
            if fold is not None:
                if fold not in range(1, 11):
                    raise ValueError("Fold must be 1-10.")

                self.samples = [
                    row for row in self.samples
                    if int(row["fold"]) == fold
                ]

        elif fold is not None:
            raise ValueError(
                "Fold filtering applies only to View 2."
            )

        self.transform = (
            get_train_transform()
            if train else get_eval_transform()
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        row = self.samples[index]

        image1 = self.transform(
            load_image(row["img1"])
        )

        image2 = self.transform(
            load_image(row["img2"])
        )

        label = int(row["label"])
        fold = int(row["fold"])

        return image1, image2, label, fold
