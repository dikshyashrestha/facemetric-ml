# LFW View 2 verification results

Accuracy = mean ± sample SD (ddof=1) over the 10 View 2 folds, in %. The SD measures fold-to-fold variation of one seed (42); it is not seed variance. AUC, EER and TAR@FAR=1% are computed from all 6,000 out-of-fold pair scores. Every method uses the reduced five-run scheme (PDF §4.2).

| method | protocol | score | selected_setting | accuracy | auc | eer | tar_at_far_1pct |
|---|---|---|---|---|---|---|---|
| Softmax baseline | Unrestricted | cos | λ=0 (cross-entropy only) | 78.27 ± 2.35 | 0.8676 | 0.2207 | 0.3117 |
| Softmax + Center Loss | Unrestricted | cos | λ=0.01, α=0.5 | 80.68 ± 2.50 | 0.8881 | 0.1913 | 0.2537 |
| Siamese (restricted) | Image-restricted | cos | margin=1.25 | 63.23 ± 2.19 | 0.6761 | 0.365 | 0.052 |
| Siamese (restricted) | Image-restricted | L2² | margin=1.25 | 63.23 ± 2.19 | 0.6761 | 0.365 | 0.052 |
| Siamese (unrestricted) | Unrestricted | cos | margin=1.25 | 66.48 ± 3.97 | 0.7134 | 0.3347 | 0.0917 |
| Siamese (unrestricted) | Unrestricted | L2² | margin=1.25 | 66.48 ± 3.97 | 0.7134 | 0.3347 | 0.0917 |
| Triplet (semi-hard) | Unrestricted | cos | α=0.2, P×K=8×4 | 86.47 ± 1.62 | 0.9365 | 0.1343 | 0.428 |
| Triplet (semi-hard) | Unrestricted | L2² | α=0.2, P×K=8×4 | 86.47 ± 1.62 | 0.9365 | 0.1343 | 0.428 |
| Triplet (batch-hard) | Unrestricted | cos | α=0.3, P×K=8×4 | 85.40 ± 1.90 | 0.9321 | 0.1447 | 0.4283 |
| Triplet (batch-hard) | Unrestricted | L2² | α=0.3, P×K=8×4 | 85.40 ± 1.90 | 0.9321 | 0.1447 | 0.4283 |
| SphereFace | Unrestricted | cos | m=3, λ annealed to 5 | 78.08 ± 3.02 | 0.8689 | 0.215 | 0.3033 |
| ArcFace | Unrestricted | cos | s=30, m=0.3 | 71.77 ± 1.81 | 0.7883 | 0.2843 | 0.1807 |
