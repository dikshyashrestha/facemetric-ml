# Result consistency checks

204 checks completed; 0 failed.

The checks compare the saved pair scores, fold results, run metadata and View 1 selections. Metrics and thresholds are recalculated from the scores, and the pair ordering and training counts are compared with the manifests.

Training-isolation entries check the recorded run metadata and manifest counts; they are not an independent reconstruction of historical training. The checks also do not establish that every statement in the report is correct.

Each method's detailed results appear below. These are output records from `compare_methods.py`.

## Softmax baseline  (`Model_3_Center/results/View_2/softmax`)

| Check | Result | Details |
|---|---|---|
| pair_scores.csv has exactly fold,img1,img2,label,score | PASS | ['fold', 'img1', 'img2', 'label', 'score'] |
| 6,000 scored pairs | PASS | 6000 |
| pairs, labels and folds identical to official View 2 list (same order) | PASS |  |
| each fold: 600 pairs, 300 genuine + 300 impostor | PASS |  |
| summary fold_to_run = (1,2),(3,4),(5,6),(7,8),(9,10) scheme | PASS | {1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 4, 8: 4, 9: 5, 10: 5} |
| run_1 trained without and scored only folds (1, 2) | PASS |  |
| run_2 trained without and scored only folds (3, 4) | PASS |  |
| run_3 trained without and scored only folds (5, 6) | PASS |  |
| run_4 trained without and scored only folds (7, 8) | PASS |  |
| run_5 trained without and scored only folds (9, 10) | PASS |  |
| five runs' held-out scores reproduce pair_scores.csv exactly | PASS |  |
| run_1 training classes (identities ≥2 images) and images match the 8 training folds; recorded overlap 0 | PASS | recorded (1336, 7240), expected (1336, 7240), overlap 0 |
| run_2 training classes (identities ≥2 images) and images match the 8 training folds; recorded overlap 0 | PASS | recorded (1375, 7558), expected (1375, 7558), overlap 0 |
| run_3 training classes (identities ≥2 images) and images match the 8 training folds; recorded overlap 0 | PASS | recorded (1374, 7770), expected (1374, 7770), overlap 0 |
| run_4 training classes (identities ≥2 images) and images match the 8 training folds; recorded overlap 0 | PASS | recorded (1311, 7081), expected (1311, 7081), overlap 0 |
| run_5 training classes (identities ≥2 images) and images match the 8 training folds; recorded overlap 0 | PASS | recorded (1324, 7007), expected (1324, 7007), overlap 0 |
| summary.json cosine metrics reproduced from pair_scores.csv | PASS | {"accuracy_mean": "0.0e+00", "accuracy_std": "0.0e+00", "auc": "0.0e+00", "eer": "0.0e+00", "tar_at_far_0.01": "0.0e+00"} |
| fold_accuracy.csv thresholds/accuracies match the recomputation | PASS |  |
| thresholds independently re-derived from the other nine folds (5,400 pairs) | PASS |  |
| a fold's threshold does not depend on its own labels (fold 1 labels flipped) | PASS |  |
| score direction: cosine, higher = same person (genuine mean > impostor mean) | PASS | genuine 0.257, impostor 0.001 |
| View 2 hyperparameters = rank-1 row of the View 1 tuning summary | PASS | View 1 rank 1: {'lambda': '0.0', 'learning_rate': '0.0001', 'epochs': '30'} |
| every View 2 run trained for the View 1-selected number of epochs | PASS | [30, 30, 30, 30, 30] |

## Softmax + Center Loss  (`Model_3_Center/results/View_2/center_loss`)

| Check | Result | Details |
|---|---|---|
| pair_scores.csv has exactly fold,img1,img2,label,score | PASS | ['fold', 'img1', 'img2', 'label', 'score'] |
| 6,000 scored pairs | PASS | 6000 |
| pairs, labels and folds identical to official View 2 list (same order) | PASS |  |
| each fold: 600 pairs, 300 genuine + 300 impostor | PASS |  |
| summary fold_to_run = (1,2),(3,4),(5,6),(7,8),(9,10) scheme | PASS | {1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 4, 8: 4, 9: 5, 10: 5} |
| run_1 trained without and scored only folds (1, 2) | PASS |  |
| run_2 trained without and scored only folds (3, 4) | PASS |  |
| run_3 trained without and scored only folds (5, 6) | PASS |  |
| run_4 trained without and scored only folds (7, 8) | PASS |  |
| run_5 trained without and scored only folds (9, 10) | PASS |  |
| five runs' held-out scores reproduce pair_scores.csv exactly | PASS |  |
| run_1 training classes (identities ≥2 images) and images match the 8 training folds; recorded overlap 0 | PASS | recorded (1336, 7240), expected (1336, 7240), overlap 0 |
| run_2 training classes (identities ≥2 images) and images match the 8 training folds; recorded overlap 0 | PASS | recorded (1375, 7558), expected (1375, 7558), overlap 0 |
| run_3 training classes (identities ≥2 images) and images match the 8 training folds; recorded overlap 0 | PASS | recorded (1374, 7770), expected (1374, 7770), overlap 0 |
| run_4 training classes (identities ≥2 images) and images match the 8 training folds; recorded overlap 0 | PASS | recorded (1311, 7081), expected (1311, 7081), overlap 0 |
| run_5 training classes (identities ≥2 images) and images match the 8 training folds; recorded overlap 0 | PASS | recorded (1324, 7007), expected (1324, 7007), overlap 0 |
| summary.json cosine metrics reproduced from pair_scores.csv | PASS | {"accuracy_mean": "0.0e+00", "accuracy_std": "0.0e+00", "auc": "0.0e+00", "eer": "0.0e+00", "tar_at_far_0.01": "0.0e+00"} |
| fold_accuracy.csv thresholds/accuracies match the recomputation | PASS |  |
| thresholds independently re-derived from the other nine folds (5,400 pairs) | PASS |  |
| a fold's threshold does not depend on its own labels (fold 1 labels flipped) | PASS |  |
| score direction: cosine, higher = same person (genuine mean > impostor mean) | PASS | genuine 0.548, impostor 0.013 |
| View 2 hyperparameters = rank-1 row of the View 1 tuning summary | PASS | View 1 rank 1: {'lambda': '0.01', 'learning_rate': '0.0001', 'epochs': '30'} |
| every View 2 run trained for the View 1-selected number of epochs | PASS | [30, 30, 30, 30, 30] |

## Siamese (restricted)  (`Model_1_Siamese/results/View_2/restricted`)

| Check | Result | Details |
|---|---|---|
| pair_scores.csv has exactly fold,img1,img2,label,score | PASS | ['fold', 'img1', 'img2', 'label', 'score'] |
| 6,000 scored pairs | PASS | 6000 |
| pairs, labels and folds identical to official View 2 list (same order) | PASS |  |
| each fold: 600 pairs, 300 genuine + 300 impostor | PASS |  |
| summary fold_to_run = (1,2),(3,4),(5,6),(7,8),(9,10) scheme | PASS | {1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 4, 8: 4, 9: 5, 10: 5} |
| run_1 trained without and scored only folds (1, 2) | PASS |  |
| run_2 trained without and scored only folds (3, 4) | PASS |  |
| run_3 trained without and scored only folds (5, 6) | PASS |  |
| run_4 trained without and scored only folds (7, 8) | PASS |  |
| run_5 trained without and scored only folds (9, 10) | PASS |  |
| five runs' held-out scores reproduce pair_scores.csv exactly | PASS |  |
| run_1 training official training pairs and genuine pairs match the 8 training folds; recorded overlap 0 | PASS | recorded (4800, 2400), expected (4800, 2400), overlap 0 |
| run_2 training official training pairs and genuine pairs match the 8 training folds; recorded overlap 0 | PASS | recorded (4800, 2400), expected (4800, 2400), overlap 0 |
| run_3 training official training pairs and genuine pairs match the 8 training folds; recorded overlap 0 | PASS | recorded (4800, 2400), expected (4800, 2400), overlap 0 |
| run_4 training official training pairs and genuine pairs match the 8 training folds; recorded overlap 0 | PASS | recorded (4800, 2400), expected (4800, 2400), overlap 0 |
| run_5 training official training pairs and genuine pairs match the 8 training folds; recorded overlap 0 | PASS | recorded (4800, 2400), expected (4800, 2400), overlap 0 |
| summary.json cosine metrics reproduced from pair_scores.csv | PASS | {"accuracy_mean": "0.0e+00", "accuracy_std": "0.0e+00", "auc": "0.0e+00", "eer": "0.0e+00", "tar_at_far_0.01": "0.0e+00"} |
| fold_accuracy.csv thresholds/accuracies match the recomputation | PASS |  |
| thresholds independently re-derived from the other nine folds (5,400 pairs) | PASS |  |
| a fold's threshold does not depend on its own labels (fold 1 labels flipped) | PASS |  |
| score direction: cosine, higher = same person (genuine mean > impostor mean) | PASS | genuine 0.795, impostor 0.697 |
| pair_scores_squared_l2.csv: same pairs, same five columns | PASS |  |
| squared L2 = 2 − 2·cosine for unit vectors | PASS | max \|diff\| 4.8e-07 |
| score direction: squared L2, lower = same person | PASS |  |
| summary.json squared-L2 metrics reproduced | PASS | {"accuracy_mean": "0.0e+00", "accuracy_std": "0.0e+00", "auc": "0.0e+00", "eer": "0.0e+00", "tar_at_far_0.01": "0.0e+00"} |
| fold_accuracy_squared_l2.csv matches the recomputation | PASS |  |
| View 2 hyperparameters = rank-1 row of the View 1 tuning summary | PASS | View 1 rank 1: {'margin': '1.25', 'learning_rate': '0.0001', 'epochs': '30'} |
| every View 2 run trained for the View 1-selected number of epochs | PASS | [30, 30, 30, 30, 30] |

## Siamese (unrestricted)  (`Model_1_Siamese/results/View_2/unrestricted`)

| Check | Result | Details |
|---|---|---|
| pair_scores.csv has exactly fold,img1,img2,label,score | PASS | ['fold', 'img1', 'img2', 'label', 'score'] |
| 6,000 scored pairs | PASS | 6000 |
| pairs, labels and folds identical to official View 2 list (same order) | PASS |  |
| each fold: 600 pairs, 300 genuine + 300 impostor | PASS |  |
| summary fold_to_run = (1,2),(3,4),(5,6),(7,8),(9,10) scheme | PASS | {1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 4, 8: 4, 9: 5, 10: 5} |
| run_1 trained without and scored only folds (1, 2) | PASS |  |
| run_2 trained without and scored only folds (3, 4) | PASS |  |
| run_3 trained without and scored only folds (5, 6) | PASS |  |
| run_4 trained without and scored only folds (7, 8) | PASS |  |
| run_5 trained without and scored only folds (9, 10) | PASS |  |
| five runs' held-out scores reproduce pair_scores.csv exactly | PASS |  |
| run_1 training identities and images match the 8 training folds; recorded overlap 0 | PASS | recorded (4593, 10497), expected (4593, 10497), overlap 0 |
| run_2 training identities and images match the 8 training folds; recorded overlap 0 | PASS | recorded (4637, 10820), expected (4637, 10820), overlap 0 |
| run_3 training identities and images match the 8 training folds; recorded overlap 0 | PASS | recorded (4655, 11051), expected (4655, 11051), overlap 0 |
| run_4 training identities and images match the 8 training folds; recorded overlap 0 | PASS | recorded (4551, 10321), expected (4551, 10321), overlap 0 |
| run_5 training identities and images match the 8 training folds; recorded overlap 0 | PASS | recorded (4560, 10243), expected (4560, 10243), overlap 0 |
| summary.json cosine metrics reproduced from pair_scores.csv | PASS | {"accuracy_mean": "0.0e+00", "accuracy_std": "0.0e+00", "auc": "0.0e+00", "eer": "0.0e+00", "tar_at_far_0.01": "0.0e+00"} |
| fold_accuracy.csv thresholds/accuracies match the recomputation | PASS |  |
| thresholds independently re-derived from the other nine folds (5,400 pairs) | PASS |  |
| a fold's threshold does not depend on its own labels (fold 1 labels flipped) | PASS |  |
| score direction: cosine, higher = same person (genuine mean > impostor mean) | PASS | genuine 0.792, impostor 0.648 |
| pair_scores_squared_l2.csv: same pairs, same five columns | PASS |  |
| squared L2 = 2 − 2·cosine for unit vectors | PASS | max \|diff\| 4.8e-07 |
| score direction: squared L2, lower = same person | PASS |  |
| summary.json squared-L2 metrics reproduced | PASS | {"accuracy_mean": "0.0e+00", "accuracy_std": "0.0e+00", "auc": "0.0e+00", "eer": "0.0e+00", "tar_at_far_0.01": "0.0e+00"} |
| fold_accuracy_squared_l2.csv matches the recomputation | PASS |  |
| View 2 hyperparameters = rank-1 row of the View 1 tuning summary | PASS | View 1 rank 1: {'margin': '1.25', 'learning_rate': '0.0001', 'epochs': '30'} |
| every View 2 run trained for the View 1-selected number of epochs | PASS | [30, 30, 30, 30, 30] |

## Triplet (semi-hard)  (`Model_2_Triplet/results/View_2/semi_hard`)

| Check | Result | Details |
|---|---|---|
| pair_scores.csv has exactly fold,img1,img2,label,score | PASS | ['fold', 'img1', 'img2', 'label', 'score'] |
| 6,000 scored pairs | PASS | 6000 |
| pairs, labels and folds identical to official View 2 list (same order) | PASS |  |
| each fold: 600 pairs, 300 genuine + 300 impostor | PASS |  |
| summary fold_to_run = (1,2),(3,4),(5,6),(7,8),(9,10) scheme | PASS | {1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 4, 8: 4, 9: 5, 10: 5} |
| run_1 trained without and scored only folds (1, 2) | PASS |  |
| run_2 trained without and scored only folds (3, 4) | PASS |  |
| run_3 trained without and scored only folds (5, 6) | PASS |  |
| run_4 trained without and scored only folds (7, 8) | PASS |  |
| run_5 trained without and scored only folds (9, 10) | PASS |  |
| five runs' held-out scores reproduce pair_scores.csv exactly | PASS |  |
| run_1 training identities and images match the 8 training folds; recorded overlap 0 | PASS | recorded (4593, 10497), expected (4593, 10497), overlap 0 |
| run_2 training identities and images match the 8 training folds; recorded overlap 0 | PASS | recorded (4637, 10820), expected (4637, 10820), overlap 0 |
| run_3 training identities and images match the 8 training folds; recorded overlap 0 | PASS | recorded (4655, 11051), expected (4655, 11051), overlap 0 |
| run_4 training identities and images match the 8 training folds; recorded overlap 0 | PASS | recorded (4551, 10321), expected (4551, 10321), overlap 0 |
| run_5 training identities and images match the 8 training folds; recorded overlap 0 | PASS | recorded (4560, 10243), expected (4560, 10243), overlap 0 |
| summary.json cosine metrics reproduced from pair_scores.csv | PASS | {"accuracy_mean": "0.0e+00", "accuracy_std": "0.0e+00", "auc": "0.0e+00", "eer": "0.0e+00", "tar_at_far_0.01": "0.0e+00"} |
| fold_accuracy.csv thresholds/accuracies match the recomputation | PASS |  |
| thresholds independently re-derived from the other nine folds (5,400 pairs) | PASS |  |
| a fold's threshold does not depend on its own labels (fold 1 labels flipped) | PASS |  |
| score direction: cosine, higher = same person (genuine mean > impostor mean) | PASS | genuine 0.681, impostor 0.007 |
| pair_scores_squared_l2.csv: same pairs, same five columns | PASS |  |
| squared L2 = 2 − 2·cosine for unit vectors | PASS | max \|diff\| 4.8e-07 |
| score direction: squared L2, lower = same person | PASS |  |
| summary.json squared-L2 metrics reproduced | PASS | {"accuracy_mean": "0.0e+00", "accuracy_std": "0.0e+00", "auc": "0.0e+00", "eer": "0.0e+00", "tar_at_far_0.01": "0.0e+00"} |
| fold_accuracy_squared_l2.csv matches the recomputation | PASS |  |
| View 2 hyperparameters = rank-1 row of the View 1 tuning summary | PASS | View 1 rank 1: {'margin': '0.2', 'learning_rate': '0.0001', 'epochs': '30'} |
| every View 2 run trained for the View 1-selected number of epochs | PASS | [30, 30, 30, 30, 30] |

## Triplet (batch-hard)  (`Model_2_Triplet/results/View_2/batch_hard`)

| Check | Result | Details |
|---|---|---|
| pair_scores.csv has exactly fold,img1,img2,label,score | PASS | ['fold', 'img1', 'img2', 'label', 'score'] |
| 6,000 scored pairs | PASS | 6000 |
| pairs, labels and folds identical to official View 2 list (same order) | PASS |  |
| each fold: 600 pairs, 300 genuine + 300 impostor | PASS |  |
| summary fold_to_run = (1,2),(3,4),(5,6),(7,8),(9,10) scheme | PASS | {1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 4, 8: 4, 9: 5, 10: 5} |
| run_1 trained without and scored only folds (1, 2) | PASS |  |
| run_2 trained without and scored only folds (3, 4) | PASS |  |
| run_3 trained without and scored only folds (5, 6) | PASS |  |
| run_4 trained without and scored only folds (7, 8) | PASS |  |
| run_5 trained without and scored only folds (9, 10) | PASS |  |
| five runs' held-out scores reproduce pair_scores.csv exactly | PASS |  |
| run_1 training identities and images match the 8 training folds; recorded overlap 0 | PASS | recorded (4593, 10497), expected (4593, 10497), overlap 0 |
| run_2 training identities and images match the 8 training folds; recorded overlap 0 | PASS | recorded (4637, 10820), expected (4637, 10820), overlap 0 |
| run_3 training identities and images match the 8 training folds; recorded overlap 0 | PASS | recorded (4655, 11051), expected (4655, 11051), overlap 0 |
| run_4 training identities and images match the 8 training folds; recorded overlap 0 | PASS | recorded (4551, 10321), expected (4551, 10321), overlap 0 |
| run_5 training identities and images match the 8 training folds; recorded overlap 0 | PASS | recorded (4560, 10243), expected (4560, 10243), overlap 0 |
| summary.json cosine metrics reproduced from pair_scores.csv | PASS | {"accuracy_mean": "0.0e+00", "accuracy_std": "0.0e+00", "auc": "0.0e+00", "eer": "0.0e+00", "tar_at_far_0.01": "0.0e+00"} |
| fold_accuracy.csv thresholds/accuracies match the recomputation | PASS |  |
| thresholds independently re-derived from the other nine folds (5,400 pairs) | PASS |  |
| a fold's threshold does not depend on its own labels (fold 1 labels flipped) | PASS |  |
| score direction: cosine, higher = same person (genuine mean > impostor mean) | PASS | genuine 0.547, impostor 0.009 |
| pair_scores_squared_l2.csv: same pairs, same five columns | PASS |  |
| squared L2 = 2 − 2·cosine for unit vectors | PASS | max \|diff\| 4.8e-07 |
| score direction: squared L2, lower = same person | PASS |  |
| summary.json squared-L2 metrics reproduced | PASS | {"accuracy_mean": "0.0e+00", "accuracy_std": "0.0e+00", "auc": "0.0e+00", "eer": "0.0e+00", "tar_at_far_0.01": "0.0e+00"} |
| fold_accuracy_squared_l2.csv matches the recomputation | PASS |  |
| View 2 hyperparameters = rank-1 row of the View 1 tuning summary | PASS | View 1 rank 1: {'margin': '0.3', 'learning_rate': '0.0001', 'epochs': '30'} |
| every View 2 run trained for the View 1-selected number of epochs | PASS | [30, 30, 30, 30, 30] |

## SphereFace  (`Model_4_Spehereface/results/View_2/sphereface`)

| Check | Result | Details |
|---|---|---|
| pair_scores.csv has exactly fold,img1,img2,label,score | PASS | ['fold', 'img1', 'img2', 'label', 'score'] |
| 6,000 scored pairs | PASS | 6000 |
| pairs, labels and folds identical to official View 2 list (same order) | PASS |  |
| each fold: 600 pairs, 300 genuine + 300 impostor | PASS |  |
| summary fold_to_run = (1,2),(3,4),(5,6),(7,8),(9,10) scheme | PASS | {1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 4, 8: 4, 9: 5, 10: 5} |
| run_1 trained without and scored only folds (1, 2) | PASS |  |
| run_2 trained without and scored only folds (3, 4) | PASS |  |
| run_3 trained without and scored only folds (5, 6) | PASS |  |
| run_4 trained without and scored only folds (7, 8) | PASS |  |
| run_5 trained without and scored only folds (9, 10) | PASS |  |
| five runs' held-out scores reproduce pair_scores.csv exactly | PASS |  |
| run_1 training classes (identities ≥2 images) and images match the 8 training folds; recorded overlap 0 | PASS | recorded (1336, 7240), expected (1336, 7240), overlap 0 |
| run_2 training classes (identities ≥2 images) and images match the 8 training folds; recorded overlap 0 | PASS | recorded (1375, 7558), expected (1375, 7558), overlap 0 |
| run_3 training classes (identities ≥2 images) and images match the 8 training folds; recorded overlap 0 | PASS | recorded (1374, 7770), expected (1374, 7770), overlap 0 |
| run_4 training classes (identities ≥2 images) and images match the 8 training folds; recorded overlap 0 | PASS | recorded (1311, 7081), expected (1311, 7081), overlap 0 |
| run_5 training classes (identities ≥2 images) and images match the 8 training folds; recorded overlap 0 | PASS | recorded (1324, 7007), expected (1324, 7007), overlap 0 |
| summary.json cosine metrics reproduced from pair_scores.csv | PASS | {"accuracy_mean": "0.0e+00", "accuracy_std": "0.0e+00", "auc": "0.0e+00", "eer": "0.0e+00", "tar_at_far_0.01": "0.0e+00"} |
| fold_accuracy.csv thresholds/accuracies match the recomputation | PASS |  |
| thresholds independently re-derived from the other nine folds (5,400 pairs) | PASS |  |
| a fold's threshold does not depend on its own labels (fold 1 labels flipped) | PASS |  |
| score direction: cosine, higher = same person (genuine mean > impostor mean) | PASS | genuine 0.276, impostor 0.004 |
| View 2 hyperparameters = rank-1 row of the View 1 tuning summary | PASS | View 1 rank 1: {'margin': '3', 'learning_rate': '0.0001', 'epochs': '30'} |
| every View 2 run trained for the View 1-selected number of epochs | PASS | [30, 30, 30, 30, 30] |

## ArcFace  (`Model_5_ArcFace/results/View_2`)

| Check | Result | Details |
|---|---|---|
| pair_scores.csv has exactly fold,img1,img2,label,score | PASS | ['fold', 'img1', 'img2', 'label', 'score'] |
| 6,000 scored pairs | PASS | 6000 |
| pairs, labels and folds identical to official View 2 list (same order) | PASS |  |
| each fold: 600 pairs, 300 genuine + 300 impostor | PASS |  |
| summary fold_to_run = (1,2),(3,4),(5,6),(7,8),(9,10) scheme | PASS | {1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 4, 8: 4, 9: 5, 10: 5} |
| run_1 trained without and scored only folds (1, 2) | PASS |  |
| run_2 trained without and scored only folds (3, 4) | PASS |  |
| run_3 trained without and scored only folds (5, 6) | PASS |  |
| run_4 trained without and scored only folds (7, 8) | PASS |  |
| run_5 trained without and scored only folds (9, 10) | PASS |  |
| five runs' held-out scores reproduce pair_scores.csv exactly | PASS |  |
| run_1 training classes (identities ≥2 images) and images match the 8 training folds; recorded overlap 0 | PASS | recorded (1336, 7240), expected (1336, 7240), overlap 0 |
| run_2 training classes (identities ≥2 images) and images match the 8 training folds; recorded overlap 0 | PASS | recorded (1375, 7558), expected (1375, 7558), overlap 0 |
| run_3 training classes (identities ≥2 images) and images match the 8 training folds; recorded overlap 0 | PASS | recorded (1374, 7770), expected (1374, 7770), overlap 0 |
| run_4 training classes (identities ≥2 images) and images match the 8 training folds; recorded overlap 0 | PASS | recorded (1311, 7081), expected (1311, 7081), overlap 0 |
| run_5 training classes (identities ≥2 images) and images match the 8 training folds; recorded overlap 0 | PASS | recorded (1324, 7007), expected (1324, 7007), overlap 0 |
| summary.json cosine metrics reproduced from pair_scores.csv | PASS | {"accuracy_mean": "0.0e+00", "accuracy_std": "0.0e+00", "auc": "0.0e+00", "eer": "0.0e+00", "tar_at_far_0.01": "0.0e+00"} |
| fold_accuracy.csv thresholds/accuracies match the recomputation | PASS |  |
| thresholds independently re-derived from the other nine folds (5,400 pairs) | PASS |  |
| a fold's threshold does not depend on its own labels (fold 1 labels flipped) | PASS |  |
| score direction: cosine, higher = same person (genuine mean > impostor mean) | PASS | genuine 0.151, impostor 0.014 |
| View 2 hyperparameters = rank-1 row of the View 1 tuning summary | PASS | View 1 rank 1: {'scale': '30', 'margin': '0.3', 'learning_rate': '0.0001', 'epochs': '28'} |
| every View 2 run trained for the View 1-selected number of epochs | PASS | [28, 28, 28, 28, 28] |
