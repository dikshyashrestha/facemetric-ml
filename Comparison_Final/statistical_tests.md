# Statistical comparison

Triplet (semi-hard) has the highest mean View 2 accuracy. The table compares its ten fold accuracies with every other method, along with the three planned ablation comparisons. Positive differences mean that method A was more accurate. Both raw p-values and Holm-adjusted p-values are reported; the correction is applied separately within each comparison group.

Two folds share each trained model, and the training sets overlap, so the fold differences are not fully independent. These tests are exploratory and can overstate the evidence for small differences. One seed also leaves seed-to-seed variation unmeasured. For ten nonzero differences, the smallest exact two-sided Wilcoxon p-value is about 0.002. The restricted/unrestricted Siamese comparison changes the available label information as well as the pair sampling.

| Comparison group | Method A | Method B | Difference (percentage points) | Folds A was better | Paired t-test p | Paired t-test p (Holm) | Wilcoxon p | Wilcoxon p (Holm) |
|---|---|---|---|---|---|---|---|---|
| best vs every other method | Triplet (semi-hard) | Softmax baseline | 8.2 | 10 | 1.5e-06 | 6.02e-06 | 0.00195 | 0.0137 |
| best vs every other method | Triplet (semi-hard) | Softmax + Center Loss | 5.78 | 10 | 2.84e-05 | 5.67e-05 | 0.00195 | 0.0137 |
| best vs every other method | Triplet (semi-hard) | Siamese (restricted) | 23.23 | 10 | 9e-10 | 6.3e-09 | 0.00195 | 0.0137 |
| best vs every other method | Triplet (semi-hard) | Siamese (unrestricted) | 19.98 | 10 | 1.45e-08 | 7.23e-08 | 0.00195 | 0.0137 |
| best vs every other method | Triplet (semi-hard) | Triplet (batch-hard) | 1.07 | 8 | 0.0107 | 0.0107 | 0.0137 | 0.0137 |
| best vs every other method | Triplet (semi-hard) | SphereFace | 8.38 | 10 | 3.99e-06 | 1.2e-05 | 0.00195 | 0.0137 |
| best vs every other method | Triplet (semi-hard) | ArcFace | 14.7 | 10 | 8.4e-09 | 5.04e-08 | 0.00195 | 0.0137 |
| planned ablation pairs | Siamese (unrestricted) | Siamese (restricted) | 3.25 | 8 | 0.0402 | 0.0402 | 0.00781 | 0.0176 |
| planned ablation pairs | Triplet (semi-hard) | Triplet (batch-hard) | 1.07 | 8 | 0.0107 | 0.0214 | 0.0137 | 0.0176 |
| planned ablation pairs | Softmax + Center Loss | Softmax baseline | 2.42 | 9 | 0.000446 | 0.00134 | 0.00586 | 0.0176 |
