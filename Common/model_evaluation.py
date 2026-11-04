## Shared LFW evaluation helpers.

import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader


@torch.inference_mode()
def test_time_embeddings(model, images):
    ## Average original and flipped embeddings before L2-normalizing.
    original = model(images, normalize=False)
    flipped = model(torch.flip(images, dims=[-1]), normalize=False)
    return F.normalize(original + flipped, p=2, dim=1)


@torch.inference_mode()
def score_pair_dataset(model, dataset, device, batch_size=32):
    ## Score a pair dataset in dataset order.
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    was_training = model.training
    model.eval()
    results = []

    try:
        for image1, image2, labels, folds in loader:
            a = test_time_embeddings(model, image1.to(device))
            b = test_time_embeddings(model, image2.to(device))

            similarities = (a * b).sum(dim=1).clamp(-1, 1).cpu().tolist()
            distances = ((a - b) ** 2).sum(dim=1).cpu().tolist()

            for label, fold, similarity, distance in zip(
                labels.tolist(), folds.tolist(), similarities, distances
            ):
                source = dataset.samples[len(results)]
                if (int(source['label']) != label or
                        int(source.get('fold') or 0) != fold):
                    raise ValueError('Pair manifest and dataset are out of order')
                results.append({
                    'fold': int(fold),
                    'img1': source['img1'],
                    'img2': source['img2'],
                    'label': int(label),
                    'score': float(similarity),
                    'squared_l2': float(distance),
                })
    finally:
        model.train(was_training)

    return results


def select_accuracy_threshold(scores, labels):
    ## Choose the score threshold that maximizes pair accuracy.
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if len(scores) == 0 or len(scores) != len(labels):
        raise ValueError('Scores and labels must have the same nonzero length')
    if not np.isfinite(scores).all() or set(np.unique(labels)) != {0, 1}:
        raise ValueError('Scores must be finite, with both pair labels present')

    far, tar, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    positives = np.sum(labels == 1)
    negatives = np.sum(labels == 0)
    accuracies = (tar * positives + (1 - far) * negatives) / len(labels)
    threshold = float(thresholds[np.argmax(accuracies)])

    # Fall back to a finite threshold if the ROC fit produced an infinite value.
    if np.isinf(threshold):
        threshold = float(np.nextafter(scores.max(), np.inf))
    return threshold


def evaluate_view2(rows, score_key='score', higher_is_same=True):
    ## Evaluate View 2 scores using the standard 10-fold protocol.
    if len(rows) != 6000:
        raise ValueError('Expected exactly 6,000 held-out View 2 pair scores')

    folds = np.array([int(r['fold']) for r in rows])
    labels = np.array([int(r['label']) for r in rows])
    raw_scores = np.array([float(r[score_key]) for r in rows])
    scores = raw_scores if higher_is_same else -raw_scores

    if not np.isfinite(scores).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError('Invalid pair scores or labels')

    per_fold = []
    for fold in range(1, 11):
        test = folds == fold
        training_scores = ~test  # Other folds' held-out scores, NOT model training examples

        if test.sum() != 600 or labels[test].sum() != 300:
            raise ValueError(f'Fold {fold} must contain 300 positive and 300 negative pairs')

        threshold = select_accuracy_threshold(
            scores[training_scores], labels[training_scores]
        )
        accuracy = np.mean((scores[test] >= threshold) == labels[test])
        per_fold.append({
            'fold': fold,
            'threshold': threshold if higher_is_same else -threshold,
            'accuracy': float(accuracy),
        })

    far, tar, _ = roc_curve(labels, scores, drop_intermediate=False)
    frr = 1 - tar
    accuracies = [f['accuracy'] for f in per_fold]

    return {
        'accuracy_mean': float(np.mean(accuracies)),
        'accuracy_std': float(np.std(accuracies, ddof=1)),
        'auc': float(roc_auc_score(labels, scores)),
        'eer': float(np.interp(0, far - frr, far)),
        'tar_at_far_0.01': float(np.max(tar[far <= 0.01])),
        'folds': per_fold,
    }


def save_pair_scores(rows, filename, score_key='score'):
    ## Export the standard pair score columns.
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['fold', 'img1', 'img2', 'label', 'score'])
        for r in rows:
            writer.writerow([r['fold'], r['img1'], r['img2'],
                             r['label'], r[score_key]])


def save_evaluation_results(results, folder):
    ## Write per-fold metrics and aggregate summaries.
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'per_fold_accuracy.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['fold', 'threshold', 'accuracy'])
        writer.writeheader()
        writer.writerows(results['folds'])
    summary = {k: v for k, v in results.items() if k != 'folds'}
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


def save_roc_plot(method_scores, filename):
    ## Plot ROC curves for all methods in one figure.
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    for name, (scores, labels) in method_scores.items():
        far, tar, _ = roc_curve(labels, scores)
        ax.plot(far, tar, label=name)
    ax.set(xscale='log', xlim=(1 / 3000, 1), ylim=(0, 1),
           xlabel='False Accept Rate', ylabel='True Accept Rate')
    ax.legend()
    ax.grid(alpha=0.2)
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
