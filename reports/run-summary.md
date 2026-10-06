# Suggested Runs Execution Summary

Run date: 2026-06-27

## Source review

- `Guideline.md`: requires data audit before training, clean taxonomy, group-aware split, duplicate checks, metadata coverage, and train-only augmentation.
- `Data preprocessing.docx`: includes Central Vietnam metadata examples, disease-specific augmentation hypotheses, label/severity schema, and seasonal distribution assumptions.
- `eda.ipynb`: provides the pre-training gate logic for manifest scan, image readability, duplicate checks, metadata coverage, image quality stats, and proposed holdout.
- `Suggested Runs.pdf`: P0 items are EX05, EX06, EX07, and EX16. The tracker notes that `hispa` and `dead_heart` should be excluded pending agronomist review.

## Executed runs

### EX05 - Data audit

- Status: done.
- Images scanned: 10,407.
- Raw classes: 10.
- Unreadable images: 0.
- Exact duplicate rows removed with keep-first policy: 74.
- Exact duplicate group members: 146.
- Perceptual duplicate candidate rows: 255.
- Pest/review rows excluded from clean baselines: 3,036 (`hispa`, `dead_heart`).
- Clean 8-class manifest after exact duplicate removal: 7,315 images.
- Limitation: no `field_id`, `province`, or `season` in supplied metadata.

Key outputs:

- `ex05_data_audit/manifest_audited.csv`
- `ex05_data_audit/manifest_clean_8class_no_exact_dupes.csv`
- `ex05_data_audit/ex05_summary.json`
- `ex05_data_audit/duplicate_exact_sha1.csv`
- `ex05_data_audit/duplicate_perceptual_dhash_candidates.csv`
- `ex05_data_audit/collection_shortcut_summary.csv`
- `ex05_data_audit/plot_*.png`

### EX06 - MobileNet reference quick run

- Status: done as quick CPU run.
- Architecture: `mobilenet_v3_small`, ImageNet weights, frozen backbone.
- Classes: clean 8-class disease/normal set.
- Train/val: 35 train + 5 val per class, 280 train / 40 val total.
- Epochs: 2.
- Final val accuracy: 0.25.
- Final val macro F1: 0.2088.

Key outputs:

- `ex06_mobilenet_reference_quick/train_log.csv`
- `ex06_mobilenet_reference_quick/validation_predictions.csv`
- `ex06_mobilenet_reference_quick/ex06_summary.json`
- `ex06_mobilenet_reference_quick/ex06_mobilenet_v3_small.pt`

### EX07 - DenseNet121 primary quick run

- Status: done as quick CPU run.
- Architecture: `densenet121`, ImageNet weights, frozen backbone.
- Classes: clean 8-class disease/normal set.
- Train/val: 40 train + 10 val per class, 320 train / 80 val total.
- Epochs: 2.
- Final val accuracy: 0.30.
- Final val macro F1: 0.2560.

Key outputs:

- `ex07_densenet_primary_quick/train_log.csv`
- `ex07_densenet_primary_quick/validation_predictions.csv`
- `ex07_densenet_primary_quick/ex07_summary.json`
- `ex07_densenet_primary_quick/ex07_densenet121.pt`

### EX16 - Robustness gate proxy

- Status: done as proxy-only evaluation.
- Reason: true field/province/season holdout cannot be run because the supplied metadata lacks those fields.
- Proxy groups used: `variety`, `age`.
- Worst proxy groups are listed in `ex16_proxy_robustness/ex16_summary.json`.

Key outputs:

- `ex16_proxy_robustness/proxy_group_metrics.csv`
- `ex16_proxy_robustness/ex16_summary.json`

## Important limitations

- EX06 and EX07 are quick frozen-backbone CPU runs, not the full training budget from the tracker.
- No field-aware, province-aware, or season-aware split can be produced from the current metadata.
- Metrics are intentionally not treated as final model performance.
- `DenseNet121_Rice_Disease_Pipeline.ipynb` had a hard-coded Kaggle token; it was replaced with an environment-variable read.
