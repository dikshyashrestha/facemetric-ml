
import torch
from torchvision import transforms


def normalize_lfw(image: torch.Tensor) -> torch.Tensor:
    ## Normalize LFW images with the project formula.
    return (image * 255.0 - 127.5) / 128.0


def get_train_transform():
    ## Train-time preprocessing with random horizontal flips.
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Lambda(normalize_lfw),
    ])


def get_eval_transform():
    ## Validation and test preprocessing with no augmentation.
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(normalize_lfw),
    ])


def get_test_time_tensors(image):
    ## Return the original and flipped tensors for test-time features.
    tensor = get_eval_transform()(image.convert("RGB"))

    flipped_tensor = torch.flip(tensor, dims=[2])

    return tensor, flipped_tensor
