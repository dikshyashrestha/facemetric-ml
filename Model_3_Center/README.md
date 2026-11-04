# Model 3: Center Loss (+ plain softmax baseline)

`center_loss.py` follows the same structure as `Model_1_Siamese/siamese.py`: the same STAGE switch, results layout,
resume behaviour, AdamW settings and View 1 selection rule.

- **softmax**: cross-entropy only (lambda = 0). This is the plain-softmax baseline for the whole project.
- **center_loss**: `CE + lambda * (1/2) * ||f(x) - c_y||^2`. The centers are updated by hand after every batch
  (alpha = 0.5, Wen et al. 2016), not by the optimizer.
- Classes are training identities with >= 2 images (PDF 4.3). Protocol: unrestricted. Score: cosine.

## Run (from the repository root)

Set `STAGE` at the top of the file, then:

```bash
python -m Model_3_Center.center_loss
```

1. `STAGE = "smoke_test"`: a few batches of everything. Output goes to `outputs/` (gitignored).
2. `STAGE = "view1_tuning"`: lambda in {0, 0.001, 0.003, 0.01, 0.03}, lr 1e-4, 30 epochs.
   This writes `results/View_1/tuning_summary.csv`.
3. Copy the rank-1 row of each variant into `FINAL_CONFIG`, then `STAGE = "view2_final"`: 5 runs per variant.

## Outputs

```
results/
├── View_1/
│   ├── softmax/lambda_0_lr_0.0001/
│   ├── center_loss/lambda_<value>_lr_0.0001/
│   │   (config.json, run_complete.json, training_curves.png, training_history.csv, validation_scores.csv)
│   └── tuning_summary.csv
└── View_2/
    ├── softmax/
    └── center_loss/
        ├── run_1 ... run_5/   (config.json, run_complete.json, training_history.csv, heldout_scores.csv, ...)
        ├── pair_scores.csv    (fold,img1,img2,label,score: 6,000 rows)
        ├── fold_accuracy.csv  (fold,threshold,correct,n_pairs,accuracy)
        ├── roc_data.csv, roc.png, score_histogram.png, confident_errors.csv
        └── summary.json
```

`training_history.csv` also has `classification_loss`, `center_loss`, `weighted_center_loss`,
`mean_intra_class_distance` and `mean_inter_center_distance`. Centers are also tracked when lambda = 0,
so the two variants can be compared.
