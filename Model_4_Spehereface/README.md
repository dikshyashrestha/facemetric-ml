# Model 4: SphereFace (A-Softmax)

`sphereface.py` follows the same structure as `Model_1_Siamese/siamese.py`: the same STAGE switch, results layout,
resume behaviour, AdamW settings and View 1 selection rule.

- Normalized class weights with zero bias, so `logit_j = ||x|| cos(theta_j)`.
- For the true class, `psi(theta) = (-1)^k cos(m theta) - 2k`, with `cos(m theta)` from the Chebyshev polynomial.
- Annealing: the target logit is `(lambda cos + psi) / (1 + lambda)`, with
  `lambda = max(5, 1000 * (1 + 0.12 * iteration)^-1)` updated every batch.
- Classes are training identities with >= 2 images (PDF 4.3). Protocol: unrestricted. Score: cosine.

View 1 variants:
- **sphereface**: m = 2, 3, 4 with annealing. The best one goes to View 2.
- **m1_baseline**: m = 1, the normalized-weight softmax with no margin (ablation).
- **no_annealing**: m = 4 without annealing (ablation, for the "what happens without annealing" question).

## Run (from the repository root)

Set `STAGE` at the top of the file, then:

```bash
python -m Model_4_Spehereface.sphereface
```

1. `STAGE = "smoke_test"`: a few batches of everything. Output goes to `outputs/` (gitignored).
2. `STAGE = "view1_tuning"`: 5 runs of 30 epochs. This writes `results/View_1/tuning_summary.csv`.
3. Copy the rank-1 `sphereface` row into `FINAL_CONFIG`, then `STAGE = "view2_final"`: 5 runs.

## Outputs

```
results/
├── View_1/
│   ├── sphereface/m_<2|3|4>_anneal_lr_0.0001/
│   ├── m1_baseline/m_1_anneal_lr_0.0001/
│   ├── no_annealing/m_4_no_anneal_lr_0.0001/
│   │   (config.json, run_complete.json, training_curves.png, training_history.csv, validation_scores.csv)
│   └── tuning_summary.csv
└── View_2/
    └── sphereface/
        ├── run_1 ... run_5/
        ├── pair_scores.csv, fold_accuracy.csv, roc_data.csv, roc.png,
        ├── score_histogram.png, confident_errors.csv
        └── summary.json
```

`training_history.csv` also has `annealing_lambda`, `mean_target_angle_degrees`, `mean_feature_norm`
and `train_classification_accuracy`.
