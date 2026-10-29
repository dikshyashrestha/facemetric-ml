
import torch
from torchvision import transforms


def normalize_lfw(image: torch.Tensor) -> torch.Tensor:
    """
    Normalize RGB pixel values using the project formula:

        (pixel - 127.5) / 128.0

    Input:
        Tensor [3, H, W], values in [0, 1].

    Output:
        Normalized tensor [3, H, W].
    """
    return (image * 255.0 - 127.5) / 128.0


def get_train_transform():
    """
    Transform used during model training.

    Random horizontal flipping is the only augmentation.
    The input is already aligned and resized to 112x112.
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Lambda(normalize_lfw),
    ])


def get_eval_transform():
    """
    Deterministic preprocessing for validation and testing.
    No random augmentation is applied.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(normalize_lfw),
    ])


def get_test_time_tensors(image):
    """
    Return the original and horizontally flipped image tensors.

    Both are used for test-time feature extraction.
    """
    tensor = get_eval_transform()(image.convert("RGB"))

    flipped_tensor = torch.flip(tensor, dims=[2])

    return tensor, flipped_tensor
