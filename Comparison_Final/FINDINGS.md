# LFW face verification: comparison of the five losses

We compared contrastive loss, triplet loss, Center Loss, SphereFace and ArcFace on LFW, with plain softmax as a baseline. Triplet loss with semi-hard mining achieved the highest verification accuracy. Center Loss improved on softmax at the accuracy-maximizing threshold, while softmax remained better at a strict false-accept rate.

The tables in this folder are calculated from the saved experiment results by `compare_methods.py`. The detailed consistency checks are in [verification_report.md](verification_report.md).

## Experimental setup

All methods use a randomly initialized, modified ResNet-18 and a 512-dimensional BN–Dropout–FC–BN embedding head. They share the same 112×112 deep-funneled images, normalization, horizontal-flip augmentation and test-time features. Evaluation sums the original and flipped embeddings and L2-normalizes the result.

We used the reduced five-run scheme from the assignment. Each run trains on eight folds and scores two held-out folds: (1, 2), (3, 4), (5, 6), (7, 8), or (9, 10). Each fold's threshold is selected using the other nine folds' out-of-fold scores. Every method is evaluated on the same 6,000 official pairs.

Final settings correspond to the highest-ranked View 1 configuration for each main variant. All final runs use seed 42. Standard deviations below describe variation across the ten folds; variability between random seeds was not measured.

## Main results

| Method | Protocol | Accuracy % (mean ± SD) | AUC | EER | TAR at FAR=1% |
|---|---|---|---|---|---|
| Triplet, semi-hard (α=0.2) | Unrestricted | 86.47 ± 1.62 | 0.936 | 0.134 | 0.428 |
| Triplet, batch-hard (α=0.3) | Unrestricted | 85.40 ± 1.90 | 0.932 | 0.145 | 0.428 |
| Softmax + Center Loss (λ=0.01) | Unrestricted | 80.68 ± 2.50 | 0.888 | 0.191 | 0.254 |
| Softmax baseline | Unrestricted | 78.27 ± 2.35 | 0.868 | 0.221 | 0.312 |
| SphereFace (m=3, annealed) | Unrestricted | 78.08 ± 3.02 | 0.869 | 0.215 | 0.303 |
| ArcFace (s=30, m=0.3) | Unrestricted | 71.77 ± 1.81 | 0.788 | 0.284 | 0.181 |
| Siamese (margin 1.25) | Unrestricted | 66.48 ± 3.97 | 0.713 | 0.335 | 0.092 |
| Siamese (margin 1.25) | Image-restricted | 63.23 ± 2.19 | 0.676 | 0.365 | 0.052 |

The full [comparison table](comparison_table.md) includes both cosine and squared-L2 scoring. They give the same reported metrics for Siamese and Triplet, consistent with `d² = 2 − 2 × cosine` for unit vectors. The saved values agree with this relation within approximately 5 × 10⁻⁷.

## What the results show

Semi-hard triplet reached 86.47% accuracy, 5.78 percentage points above Center Loss and 1.07 points above batch-hard triplet. It outperformed batch-hard on eight of ten folds. Both miners gave nearly the same TAR at FAR=1%, so the small accuracy difference does not translate into an advantage at that operating point. Neither selected triplet configuration reached its View 1 plateau criterion within the 30-epoch budget.

Unrestricted Siamese improved on restricted Siamese by 3.25 percentage points. Both variants used the same loss, margin, batch size and number of pairs per epoch. The unrestricted variant could generate fresh pairs from training identity labels, while the restricted variant used only the fixed official pairs. The improvement is consistent with the additional label information and pair variety, although both variants remained below the other methods.

Center Loss improved accuracy over softmax by 2.42 percentage points and increased AUC from 0.868 to 0.888. In the selected View 1 runs, mean feature-to-center distance fell from 19.5 to 10.6, while mean inter-center distance changed from 26.5 to 23.4. These training statistics support greater compactness. Softmax nevertheless had the higher TAR at FAR=1%: 0.312 versus 0.254.

SphereFace and softmax had similar final accuracy, 78.08% and 78.27%. ArcFace reached 71.77%. Its zero-margin View 1 control also scored below the other softmax-style baselines, so the lower result cannot be attributed to the additive margin alone. Sampling, feature scaling, stopping policy and software environment differed. The comparison therefore describes these training pipelines rather than isolating the loss function completely.

## View 1 experiments

The [development table](view1_ablations.md) contains all 49 configurations. Validation accuracy uses the best threshold on the same 1,000 development pairs, so it is a tuning measure rather than a held-out performance estimate.

- Siamese accuracy increased across the tested margins from 0.5 to 1.25. Both variants selected the largest tested margin.
- For both triplet miners, learning rate 0.0001 outperformed 0.0003 at every tested margin. At 0.0001, semi-hard accuracy ranged from 0.822 to 0.841 and batch-hard from 0.810 to 0.820.
- Center Loss selected λ=0.01, with accuracy 0.824. Increasing λ to 0.03 reduced accuracy to 0.725, below the softmax baseline of 0.778.
- SphereFace with annealing reached 0.798, 0.803 and 0.790 for margins 2, 3 and 4. Margin 1 reached 0.792. Margin 4 without annealing reached only 0.585.
- ArcFace selected s=30 and m=0.3. Three of the six s=64 configurations stopped after 10–13 epochs with accuracy between 0.559 and 0.569. At s=30 and learning rate 0.0001, the margin ablation gave 0.686, 0.646, 0.684 and 0.679 for m=0, 0.2, 0.3 and 0.5.

## Statistical comparison

The [paired tests](statistical_tests.md) use the ten cosine-accuracy folds, with Holm correction applied within each comparison group. The fold-level tests favor semi-hard triplet over the other methods, but the small miner difference needs more caution than the larger gaps.

Two folds share each trained model, and training sets overlap across runs. The fold differences are therefore not independent, which can make the p-values optimistic. One seed also leaves seed-to-seed variation unknown. The 0.18-point difference between SphereFace and softmax is small relative to fold variation and does not support a clear ranking between them.

## Compute and limitations

The [training-cost table](training_cost.md) records 6.6 hours for each softmax/Center variant, 6.5 for SphereFace, 9.5–10.0 for each triplet variant, 24.6–25.3 for each Siamese variant, and 4.2 for ArcFace, summed over five runs. These experiments used different machines and PyTorch versions, so those times do not measure the relative cost of the losses on identical hardware. MPS sampled driver memory and CUDA peak tensor memory also measure different quantities.

Batch composition and training budgets differed. Siamese used 16 pairs per batch; Triplet used 8×4 identity batches plus four singleton negatives; Softmax, Center Loss and SphereFace used shuffled batches of 64; ArcFace used 8×4 identity batches. ArcFace trained for the selected 28 epochs, while the other final configurations used 30.

Learning rate was varied only for Triplet and ArcFace. The other methods used 0.0001, so their learning-rate sensitivity was not measured. These results also cover only one seed and the reduced five-run scheme. TAR at FAR=1% is based on 3,000 impostor pairs, with 1% corresponding to 30 false accepts, so fine differences at that operating point remain uncertain.

## Files in this folder

| File | Contents |
|---|---|
| `compare_methods.py` | Calculates comparison outputs from the saved results |
| `comparison_table.csv` / `.md` | Main metrics for cosine and squared-L2 scoring |
| `fold_accuracies.csv` | Ten fold accuracies for each final variant |
| `statistical_tests.csv` / `.md` | Paired t-tests, Wilcoxon tests and Holm corrections |
| `training_cost.csv` / `.md` | Recorded time, epochs, memory and hardware |
| `view1_ablations.csv` / `.md` | Completed development experiments |
| `roc_combined.png` / `.pdf` | Log-FAR ROC, with restricted Siamese shown separately in the second panel |
| `verification_report.md` | Detailed results of the 204 consistency checks |
