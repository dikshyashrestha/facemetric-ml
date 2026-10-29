# Clark University — Applied Deep Learning Project 1

Deep metric learning for LFW face verification, MSDS-3100.

## Current status

Existing preprocessing and five Common modules are preserved. All five model folders were empty at repository preparation; training entry points and results do not yet exist. This is a collaboration scaffold, not a completed experiment. See Project_1.pdf for the authoritative assignment and COLLABORATION.md for responsibilities and the required results checklist.

## Layout

- `Common/base_model.py`: FaceEmbeddingNet, modified ResNet-18 and 512-D BN–Dropout–FC–BN head, random initialization.
- `Common/dataset_loader.py`: identity/pair datasets backed by local manifests.
- `Common/dataset_transforms.py`: shared normalization, flips and evaluation transforms.
- `Common/model_evaluation.py`: pair scoring, thresholds, metrics and exports.
- `Common/model_utils.py`: seeds, device selection, timing, memory and JSON helpers.
- `Dataset/preprocessdata.py`: existing Kaggle CSV reconstruction, official protocol MD5 verification and aligned-image cache preparation.
- `Model_1_Siamese`, `Model_2_Triplet`, `Model_3_Center`, `Model_4_Spehereface`, `Model_5_ArcFace`: individual implementations. The existing `Spehereface` spelling is intentionally retained.
- `Comparison_Final`: combined tables, plots, statistical tests and report integration.

## Setup and local data

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python Dataset/preprocessdata.py
```

The preprocessing command requires manually acquired inputs first. Download the LFW dataset from the mirror linked by torchvision: https://www.kaggle.com/datasets/jessicali9530/lfw-dataset . Extract it under `Dataset/lfw_deepfuneeled/`. The script recursively finds exactly one folder containing `peopleDevTrain.csv`, `people.csv`, and `pairs.csv`; that same folder's descendants must contain exactly one `lfw-deepfunneled/` image directory containing `Aaron_Eckhart/`. Keep the original mirror CSV filenames, including match/mismatch pair CSVs. Inspect `find_metadata` and `find_images` if the archive has a different wrapper layout. Archives and images stay local.

Preprocessing reconstructs and MD5-checks six official TXT protocols, writes six manifests under `Dataset/Manifests/`, and converts aligned 250×250 JPEGs to 112×112 RGB PNGs under `Dataset/Processed/deepfunneled_112/`. All generated dataset files are excluded from Git and must be regenerated on each machine. Existing code supports `--metadata-only`; new model code should use plain Python configuration/constants rather than introducing an argument parser.

Use the existing pinned requirements without silent upgrades. Installation and training have not been validated on a clean machine. Dependencies and accelerator support may differ by platform.

## Unresolved professor requirement: torchvision.datasets.LFWPeople

Professor Ahmed El-Sayed explicitly directed the team to use torchvision's LFWPeople. Current preprocessing uses mirror CSVs and current Common loaders subclass torch.utils.data.Dataset; neither instantiates LFWPeople. Therefore compliance is unresolved. The official documentation allows `image_set="deepfunneled"`, advises manual download and `download=False`, and links the same Kaggle mirror. This supports the source choice but does not establish loader compliance. Coordinate a minimal shared adapter using LFWPeople, validate identity/fold equivalence and cache mapping, or obtain explicit instructor acceptance of the current loader before claiming compliance. Do not independently replace Common in each model branch.

Documentation checked October 28, 2026: https://docs.pytorch.org/vision/main/generated/torchvision.datasets.LFWPeople.html

## Training and reproduction

No training commands are claimed yet. Each model owner must add a runnable module, final YAML/JSON configuration and exactly one documented command that reproduces its main-table numbers. Prefer `python -m Model_2_Triplet.train` style entry points with in-file configuration or a fixed configuration file, without a new CLI parser. Add only commands that actually work. See PARTNER_AI_PROMPT.md for onboarding.

Keep this coursework repository private by default. Do not include dataset images, archives, checkpoints, credentials or unreviewed notebook outputs. The report and submission must include real results, references and individual contributions; never fabricated numbers.
