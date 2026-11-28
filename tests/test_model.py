import torch

from Common.base_model import FaceEmbeddingNet
from Common.dataset_transforms import normalize_lfw
from Common import model_evaluation as evaluation


def make_model():
    torch.manual_seed(0)
    return FaceEmbeddingNet().eval()


def test_embedding_shape_and_unit_norm():
    model = make_model()
    with torch.no_grad():
        out = model(torch.randn(3, 3, 112, 112), normalize=True)
    assert out.shape == (3, 512)
    assert torch.allclose(out.norm(dim=1), torch.ones(3), atol=1e-5)


def test_parameter_count_matches_saved_configs():
    # Every saved config.json records 11,433,536 trainable parameters.
    model = make_model()
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 11_433_536


def test_normalization_formula():
    pixels = torch.tensor([0.0, 127.5 / 255.0, 1.0])
    assert torch.allclose(normalize_lfw(pixels), torch.tensor([-127.5 / 128, 0.0, 127.5 / 128]))


def test_test_time_embedding_is_flip_invariant_and_normalized():
    model = make_model()
    images = torch.randn(2, 3, 112, 112)
    a = evaluation.test_time_embeddings(model, images)
    b = evaluation.test_time_embeddings(model, torch.flip(images, dims=[-1]))
    assert torch.allclose(a, b, atol=1e-5)
    assert torch.allclose(a.norm(dim=1), torch.ones(2), atol=1e-5)
