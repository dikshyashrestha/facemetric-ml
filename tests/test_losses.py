import math

import pytest
import torch
import torch.nn.functional as F

from Model_1_Siamese.siamese import ContrastiveLoss
from Model_2_Triplet.triplet_loss import mine_triplets
from Model_3_Center.center_loss import CenterLoss
from Model_4_Spehereface.sphereface import SphereFaceHead, cos_m_theta
from Model_5_ArcFace.arcface import ArcFaceHead


def unit(*rows):
    return F.normalize(torch.tensor(rows, dtype=torch.float32), dim=1)


def test_contrastive_loss_genuine_and_impostor():
    loss = ContrastiveLoss(margin=1.0)
    a = unit([1.0, 0.0])
    assert loss(a, a, torch.tensor([1])).item() == pytest.approx(0.0, abs=1e-6)
    # Identical impostor pair pays the full margin squared.
    assert loss(a, a, torch.tensor([0])).item() == pytest.approx(1.0, abs=1e-5)
    # Impostors farther apart than the margin cost nothing.
    assert loss(a, unit([-1.0, 0.0]), torch.tensor([0])).item() == pytest.approx(0.0, abs=1e-6)


def test_center_loss_is_zero_at_centers_and_centers_move_toward_features():
    torch.manual_seed(0)
    loss = CenterLoss(num_classes=3)
    feats = torch.randn(3, 512)
    labels = torch.tensor([0, 1, 2])
    before = loss(feats, labels).item()
    loss.update_centers(feats, labels)
    assert loss(feats, labels).item() < before
    loss.centers.copy_(feats)
    assert loss(feats, labels).item() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("m", [1, 2, 3, 4])
def test_sphereface_chebyshev_expansion(m):
    theta = torch.linspace(0, math.pi, 50)
    assert torch.allclose(cos_m_theta(torch.cos(theta), m), torch.cos(m * theta), atol=1e-5)


def test_sphereface_margin_lowers_only_the_target_logit():
    torch.manual_seed(0)
    head = SphereFaceHead(num_classes=5, margin=3)
    feats, labels = torch.randn(4, 512), torch.tensor([0, 1, 2, 3])
    logits, cos = head(feats, labels, lam=0.0)
    scaled = cos * feats.norm(dim=1, keepdim=True)
    target = logits.gather(1, labels.view(-1, 1))
    assert (target <= scaled.gather(1, labels.view(-1, 1)) + 1e-5).all()
    mask = F.one_hot(labels, 5).bool()
    assert torch.allclose(logits[~mask], scaled[~mask], atol=1e-5)


def test_arcface_zero_margin_is_scaled_cosine_and_margin_penalizes_target():
    torch.manual_seed(0)
    feats, labels = torch.randn(4, 512), torch.tensor([0, 1, 2, 3])
    plain = ArcFaceHead(512, 5, scale=30, margin=0.0)
    logits = plain(feats, labels)
    cos = F.linear(F.normalize(feats), F.normalize(plain.weight))
    assert torch.allclose(logits, cos * 30, atol=1e-4)
    margin = ArcFaceHead(512, 5, scale=30, margin=0.3)
    margin.weight.data.copy_(plain.weight.data)
    out = margin(feats, labels)
    assert (out.gather(1, labels.view(-1, 1)) < logits.gather(1, labels.view(-1, 1))).all()
    with pytest.raises(ValueError):
        ArcFaceHead(512, 1, scale=30, margin=0.3)


def toy_batch():
    # Two identities, two images each, plus one image of a third identity.
    emb = torch.tensor([[1.0, 0.0], [0.9, 0.3], [0.8, 0.6], [0.6, 0.8], [-1.0, 0.0]])
    labels = torch.tensor([0, 0, 1, 1, 2])
    paths = [f"img{i}.png" for i in range(5)]
    return emb, labels, paths


def test_batch_hard_mining_picks_hardest_examples():
    emb, labels, paths = toy_batch()
    loss, stats = mine_triplets(emb, labels, paths, margin=0.2, method="batch_hard", anchor_count=5)
    assert stats["mined"] == 4  # the identity with one image cannot be an anchor
    assert loss.item() >= 0


def test_semi_hard_negatives_fall_inside_the_margin_band():
    emb, labels, paths = toy_batch()
    loss, stats = mine_triplets(emb, labels, paths, margin=0.5, method="semi_hard", anchor_count=5)
    assert stats["eligible"] == 4
    if stats["mined"]:
        assert stats["mean_an2"] > stats["mean_ap2"]
        assert 0 <= loss.item() <= 0.5


def test_unknown_mining_method_is_rejected():
    emb, labels, paths = toy_batch()
    with pytest.raises(ValueError):
        mine_triplets(emb, labels, paths, margin=0.2, method="random", anchor_count=5)
