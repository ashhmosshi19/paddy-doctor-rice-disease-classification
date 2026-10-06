# Paddy Doctor: Rice Disease Classification

An image-only rice-leaf disease classification study built around a reproducible audit, label policy, duplicate review, grouped validation, and transfer-learning baselines.

## What this repository demonstrates

- data-readiness checks before training;
- a versioned eight-class baseline taxonomy;
- exact and perceptual duplicate analysis;
- proxy group-aware splitting when field metadata is unavailable;
- frozen-backbone experiments with MobileNet, DenseNet, EfficientNet, and ResNet;
- a Kaggle-style submission flow.

## Results snapshot

The audit scanned 10,407 images and produced a clean eight-class manifest with 7,315 images after exact-duplicate removal. It identified 255 perceptual-duplicate candidates that still need human review.

On the available proxy split, the best full frozen-feature run was an EfficientNet-B0 experiment with macro F1 0.3749 and accuracy 0.6088. These are diagnostic results, not evidence of field deployment performance: the source metadata does not provide reliable field, province, or season identifiers.

## Reproducing the work

The original competition dataset and model checkpoints are intentionally not committed because of size and redistribution constraints. Place an approved local dataset under the paths expected by the scripts, then run:

```bash
uv sync
uv run python scripts/run_suggested_p0.py
uv run python scripts/create_kaggle_submission.py
```

See `label_map.yaml` and `reports/` for the taxonomy decisions, experiment summaries, and limitations.

## Responsible interpretation

This is a baseline research/engineering project. It does not claim agronomic diagnosis or generalization to Central Vietnam. The quarantined `hispa` and `dead_heart` labels require expert review before being treated as disease classes.
