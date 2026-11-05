# Paired tests over the 10 View 2 folds (cosine accuracy)

Best mean accuracy: **Triplet (semi-hard)**. Holm correction is applied within each family. Caveats: (1) the two folds scored by the same run share a model, so the 10 differences are not fully independent and p-values are likely optimistic; (2) with n = 10 the smallest two-sided Wilcoxon p is ≈ 0.002; (3) restricted vs unrestricted Siamese also differ in protocol, by design.

| family | method_a | method_b | mean_difference_pct_points | folds_a_better | paired_t_p | paired_t_p_holm | wilcoxon_p | wilcoxon_p_holm |
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
