# Deep Metric Learning for Face Verification on LFW

MSDS-3100 Applied Deep Learning, Project 1 — Clark University

This project compares five ways of learning face embeddings: contrastive loss with a Siamese network, triplet loss, softmax with Center Loss, SphereFace, and ArcFace. We also include plain softmax as a baseline. The task is to decide whether two LFW images show the same person.

All methods use the same randomly initialized ResNet-18 backbone, a 512-dimensional embedding head, and the same preprocessing and evaluation code. The experiments were trained on LFW only, without face-pretrained weights or external training images.

## Project layout

| Folder | Contents |
|---|---|
| `Common/` | Shared backbone, datasets, transforms, evaluation, and training utilities |
| `Dataset/` | Preprocessing script and local data, protocols, manifests, and image cache |
| `Model_1_Siamese/` | Contrastive loss under restricted and unrestricted training |
| `Model_2_Triplet/` | Semi-hard and batch-hard online triplet mining |
| `Model_3_Center/` | Plain softmax and softmax with Center Loss |
| `Model_4_Spehereface/` | SphereFace, including margin and annealing experiments |
| `Model_5_ArcFace/` | ArcFace, including scale and margin experiments |
| `Comparison_Final/` | Combined results, ROC curves, statistical tests, and comparison script |

The spelling of `Model_4_Spehereface` is retained in the folder and module names. References to "PDF 4.2" in the code and saved configs point to section 4.2 of the course assignment, which set the reduced five-run evaluation scheme described below.

## Setup

Run the commands below from the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` records the direct package versions installed in the local Python 3.12 environment. Historical experiments ran in different environments; each run's `config.json` records its software and hardware. The training code selects MPS, CUDA, or CPU according to availability.

## Preparing LFW

We used the deep-funneled aligned version of LFW from the [Kaggle mirror](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset). Download and extract it under `Dataset/` before running preprocessing. The dataset is not included in the repository or submission.

The extracted files must contain one metadata directory with the original CSV filenames, including `peopleDevTrain.csv`, `peopleDevTest.csv`, `people.csv`, `pairs.csv`, and the development match/mismatch pair CSVs. Within that directory, the script expects one `lfw-deepfunneled/` image directory containing the identity folders. It searches recursively to handle the archive's wrapper directories.

```bash
python Dataset/preprocessdata.py
```

This reconstructs the six official pair/people protocol files and checks their MD5 hashes. It then writes CSV manifests to `Dataset/Manifests/` and resizes the aligned 250×250 images to 112×112 RGB PNGs in `Dataset/Processed/deepfunneled_112/`. No additional face detector or crop is applied. Training uses random horizontal flips and normalization `(pixel - 127.5) / 128`.

To generate only the protocols and manifests:

```bash
python Dataset/preprocessdata.py --metadata-only
```

The metadata CSVs are still needed for this command. Model training and image scoring also need the processed image cache.

## Running the models

Each training file has a `STAGE` setting near the top:

- `"smoke_test"` runs a small check of the training and evaluation pipeline.
- `"view1_tuning"` runs the development experiments.
- `"view2_final"` runs the final evaluation using the selected View 1 settings.

The files are currently set to `"view2_final"`. Set the stage you want, then run the corresponding command from the project root:

| Method | Command |
|---|---|
| Siamese, both protocols | `python -m Model_1_Siamese.siamese` |
| Triplet, both miners | `python -m Model_2_Triplet.triplet_loss` |
| Softmax and Center Loss | `python -m Model_3_Center.center_loss` |
| SphereFace | `python -m Model_4_Spehereface.sphereface` |
| ArcFace | `python -m Model_5_ArcFace.arcface` |

For a new experiment, run View 1 first. Siamese, Center Loss, and SphereFace store the selected settings in `FINAL_CONFIG`; Triplet and ArcFace read their selection from the View 1 tuning summary. Final settings should be chosen from View 1 before running View 2.

Completed runs are reused when their recorded settings match, and interrupted runs can resume from local checkpoints. Running a final-stage command against the supplied completed results therefore does not necessarily retrain the model. Keep existing results separate when starting a new experiment; completed runs should not be mixed with changed settings.

## Evaluation and results

We used the assignment's reduced five-run scheme. Each run trains from scratch on eight View 2 folds and scores two held-out folds: (1, 2), (3, 4), (5, 6), (7, 8), and (9, 10). Training and held-out identities are disjoint. Each fold's decision threshold is selected from the other nine folds' out-of-fold pair scores.

At test time, the embeddings of an image and its horizontal flip are summed and L2-normalized. Pairs are scored with cosine similarity. Siamese and Triplet also report squared Euclidean distance, which agrees with `2 - 2 * cosine` for the normalized embeddings.

| Method | Protocol | Accuracy (%) | AUC | EER | TAR at FAR=1% |
|---|---|---:|---:|---:|---:|
| Triplet, semi-hard | Unrestricted | 86.47 ± 1.62 | 0.9365 | 0.1343 | 0.4280 |
| Triplet, batch-hard | Unrestricted | 85.40 ± 1.90 | 0.9321 | 0.1447 | 0.4283 |
| Softmax + Center Loss | Unrestricted | 80.68 ± 2.50 | 0.8881 | 0.1913 | 0.2537 |
| Softmax baseline | Unrestricted | 78.27 ± 2.35 | 0.8676 | 0.2207 | 0.3117 |
| SphereFace | Unrestricted | 78.08 ± 3.02 | 0.8689 | 0.2150 | 0.3033 |
| ArcFace | Unrestricted | 71.77 ± 1.81 | 0.7883 | 0.2843 | 0.1807 |
| Siamese | Unrestricted | 66.48 ± 3.97 | 0.7134 | 0.3347 | 0.0917 |
| Siamese | Image-restricted | 63.23 ± 2.19 | 0.6761 | 0.3650 | 0.0520 |

Accuracy is the mean ± sample standard deviation across ten folds, using one seed (42). AUC, EER, and TAR are computed from the 6,000 out-of-fold scores. The restricted Siamese row uses pair labels only; the unrestricted methods use training identity labels.

To recompute the main table, fold accuracies, paired tests, training-cost summaries, and combined ROC from the saved results:

```bash
python Comparison_Final/compare_methods.py
```

This writes the comparison outputs to `Comparison_Final/` and runs 204 consistency checks. It does not train models or change their selected settings. It requires the saved model results and the View 2 manifests, which can be regenerated with the metadata-only preprocessing command above. This is the way to reproduce the reported table from the supplied scores; fresh training on different hardware may produce different scores.

## Saved experiment files

View 1 and View 2 results are stored under each model's `results/` folder. The main files are:

- `config.json`: run settings, training-data counts, hardware, and software.
- `training_history.csv` and `training_curves.png`: losses and training diagnostics; View 1 also records validation metrics.
- `run_complete.json`: completed-run summary.
- `heldout_scores.csv`: pairs scored by an individual View 2 run.
- `pair_scores.csv`: all 6,000 final pairs, with columns `fold,img1,img2,label,score`.
- `fold_accuracy.csv`: each fold's threshold and accuracy.
- `summary.json`: aggregate metrics and selected settings.
- `roc.png`, `score_histogram.png`, and `confident_errors.csv`: evaluation plots and error examples.

Siamese and Triplet also save squared-L2 scores and fold results. The aggregate files sit in each variant's View 2 folder; ArcFace uses `results/View_2/` directly.

The comparison uses one seed and a limited training budget. Learning-rate searches were run for Triplet and ArcFace; the other methods used a fixed learning rate. Hardware and batch composition differ across methods, so the recorded training times should not be treated as a hardware-independent cost ranking.

Dataset images, model checkpoints, saved weights, local embedding arrays, and the Python environment are excluded from the submission. Source code, result CSVs, configurations, and the report figures provide the submitted experimental evidence.
