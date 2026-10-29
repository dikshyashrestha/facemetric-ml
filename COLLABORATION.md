# Collaboration and experiment contract

## Ownership and branches

Assign a named owner and reviewer before implementation; no person assignments are assumed here.

| Folder | Responsibility | Owner / reviewer |
|---|---|---|
| Model_1_Siamese | restricted and unrestricted contrastive experiments | Unassigned |
| Model_2_Triplet | semi-hard and batch-hard mining | Unassigned |
| Model_3_Center | center loss and plain-softmax baseline | Unassigned |
| Model_4_Spehereface | SphereFace and margin ablations | Unassigned |
| Model_5_ArcFace | ArcFace and margin ablations | Unassigned |
| Common + Dataset | shared contracts, preprocessing and LFWPeople resolution | Team assignment required |
| Comparison_Final | combined analysis and report | Team assignment required |

Start from current main: `git switch main`, `git pull --ff-only`, then `git switch -c model/triplet-yourname` (choose your assigned model). Work in your folder, commit explicit paths, push with `git push -u origin HEAD`, and open a PR to main. Include changed behavior, exact run instructions, validation, configuration, seed, split and result provenance. Another partner reviews before merge. Do not force-push main. Use an issue/PR to coordinate changes to Common, preprocessing, requirements or protocols; all owners must agree to experiment-affecting changes. Record a shared commit/config version before final evaluation. Invite collaborators in GitHub Settings → Collaborators after publication; repository creation alone does not grant access.

## Preserve shared interfaces

- `FaceEmbeddingNet(embedding_dim=512, dropout=0.4).forward(x, normalize=False)` expects `[B,3,112,112]`, returns `[B,512]`. Loss-specific heads remain in model folders; discard classification heads for verification.
- `LFWIdentityDataset(split="view1_train", held_out_folds=None, train=True)` returns `(tensor, identity_label)` and exposes `labels`, `identity_to_label`, `indices_by_identity`. For View 2 use `split="view2"` and explicitly exclude held-out folds. The loader does NOT apply ≥2-image class filtering; classification owners must filter and rebuild contiguous class labels before training. Triplet singletons can be negatives only.
- `LFWPairDataset(split="view1_validation", fold=None, train=False)` returns `(image1,image2,label,fold)`; 1=same person, 0=different. It exposes `.samples`. For restricted View 2 training explicitly exclude held-out pairs; `fold=k` selects that fold, not its complement.
- All inputs use cached RGB 112×112 images; `(pixel-127.5)/128`, training random horizontal flips, deterministic evaluation. Shared test features sum original and flipped embeddings then L2-normalize.
- `score_pair_dataset` produces cosine `score` and `squared_l2`. `evaluate_view2` requires 6,000 OUT-OF-FOLD rows; it cannot prove training isolation. Other nine folds' out-of-fold scores select each threshold. `save_pair_scores` exports exactly `fold,img1,img2,label,score`; give distance exports separate filenames. `save_evaluation_results` saves per-fold accuracy and summary; `save_roc_plot` plots a combined ROC. Confirm score direction; negative distance is used for ROC.
- Use `seed_everything`, `get_device`, `TrainingMonitor`, `hardware_info`, and `save_json`. MPS memory is a sampled maximum, not a true peak; label it correctly.

## Protocol and fairness

View 1: train on pairsDevTrain (2,200 pairs) for restricted Siamese or peopleDevTrain identities for unrestricted methods; tune/validate on pairsDevTest (1,000 pairs). All hyperparameters, epoch counts and ablation decisions must be fixed here. Never tune on View 2.

View 2: 10 identity-disjoint folds, each 300 same + 300 different pairs. Normally train 10 fresh models per method, excluding the scored fold's identities/pairs. The accepted reduced scheme trains 5 models per method excluding fold pairs (1,2), (3,4), (5,6), (7,8), (9,10), scoring only those held-out folds; apply the same scheme to all methods and disclose it. Select each test-fold threshold from the other 5,400 out-of-fold pair scores; apply unchanged to that fold's 600 pairs. Never reuse a View 1-trained checkpoint for a View 2 run without proving identity isolation: the views reuse the dataset. Shared helpers do not enforce this.

Use the same backbone, inputs, augmentation and test-time features for all methods. Random initialization is current behavior; no face-pretrained weights in the main track. SGD momentum 0.9 and weight decay 5e-4 or AdamW; tune LR per method on View 1, cosine/step schedule. At least 3 seeds if compute permits, otherwise disclose 1. Retain per-seed results and state how aggregation was done. Follow assignment protocol labels (restricted/unrestricted); the phrase “unrestricted, labeled outside data” in the handout does not authorize silently adding external training faces.

## Required experiments and evidence

- Siamese: restricted official training pairs only; no identity-derived extra pairs. Also unrestricted with freely generated 1:1 same/different pairs. Unit embeddings; margin search starts 0.5,0.75,1,1.25; log genuine/impostor distance distributions.
- Triplet: compare semi-hard and batch-hard online mining, P×K sampling; singletons never anchors/positives. Starting alpha 0.1,0.2,0.3; log active-triplet fraction and sampler/mining details.
- Center: train classes with ≥2 images; softmax + center versus lambda=0 softmax baseline. Starting lambda 0.001,0.003,0.01,0.03 and center update alpha 0.5; document separate center updates, intra-class/center and inter-center distances.
- SphereFace: same class filtering, normalized weights/zero bias, monotonic piecewise angular margin, required annealing; compare margins 1,2,3,4 and record annealing schedule. Discuss evidence for behavior without annealing.
- ArcFace: same filtering, normalized features/weights, numerical stability and threshold/easy-margin choice. Tune scale starting 16,30,64; required margins 0,0.2,0.3,0.5 at chosen scale.
- All: save class/image counts per run, hyperparameter search and final config, seeds, epochs, hardware, per-fold wall time and memory, parameter counts and provenance (commit/version, data source, checksum audit). Include any alignment fallback count; current path uses already aligned deep-funneled images without a new detector.

## Results and report handoff

For every main method/variant/seed, preserve the 6,000 pair scores with exact five-column schema; per-fold accuracy and threshold; mean ± sample SD over 10 folds; ROC/AUC; EER; TAR at FAR=0.01; training time and epochs. Siamese and triplet additionally export squared L2 results and empirically confirm squared_l2≈2−2*cosine. Explicitly label protocol and score type in tables. No invented or placeholder metrics presented as measured results.

Main table includes softmax, restricted Siamese, unrestricted Siamese, semi-hard triplet, batch-hard triplet, center, SphereFace and ArcFace. Save all ablation tables, training loss/View 1 accuracy curves, combined ROC with log FAR, genuine/impostor score histograms with a stated threshold/fold, embedding visualizations, and best/worst confident-error galleries with failure analysis. Keep galleries containing face images LOCAL and transfer only via an instructor-approved private submission channel; Git ignores protect Dataset, but review all figures manually for faces. Compare best versus each other method with paired t-test or Wilcoxon over aligned 10 fold accuracies and save p-values; discuss variability and multiple comparisons.

Use small reviewed CSV/JSON/YAML and non-face plots under each model's `results/` and final comparison assets in `Comparison_Final/`. Store checkpoints, embeddings and large intermediate outputs in ignored folders. Model owners must document one real command per method to reproduce the main table.

Report: PDF from Word/LaTeX, 12–20 pages excluding references/appendices, 11-point, single column; abstract 150–250 words; introduction, loss background/equations, dataset/protocol, implementation, results, discussion, ranked comparison/conclusion, references, appendix with full configs and member contributions. Address all nine questions in handout §7.3. Submit source/repo link plus results and final configs in the required archive, excluding dataset and checkpoints.

## Academic honesty

Follow the course's AI policy; confirm it if unavailable. Each partner must understand and verify submitted code/equations, cite original papers and reused code/tools, disclose AI use as required, and accurately record individual contributions. Do not invent experiments, inflate performance, conceal leakage, or describe unimplemented code as complete. Use a private repo and approved sharing channels for coursework and dataset-derived face galleries.
