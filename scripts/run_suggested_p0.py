from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


CANONICAL_8 = [
    "bacterial_leaf_blight",
    "bacterial_leaf_streak",
    "bacterial_panicle_blight",
    "blast",
    "brown_spot",
    "downy_mildew",
    "normal",
    "tungro",
]

PEST_OR_REVIEW = {"hispa", "dead_heart"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def dhash(image: Image.Image, hash_size: int = 8) -> str:
    gray = ImageOps.grayscale(image).resize(
        (hash_size + 1, hash_size), Image.Resampling.LANCZOS
    )
    pixels = np.asarray(gray, dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    bit_string = "".join("1" if bit else "0" for bit in bits.flatten())
    return f"{int(bit_string, 2):0{hash_size * hash_size // 4}x}"


def image_stats(path: Path) -> dict:
    rec: dict[str, object] = {"is_readable": False, "error": ""}
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im = im.convert("RGB")
            arr = np.asarray(im)
            h, w = arr.shape[:2]
            gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
            max_rgb = arr.max(axis=2).astype(float)
            min_rgb = arr.min(axis=2).astype(float)
            saturation = np.divide(max_rgb - min_rgb, np.maximum(max_rgb, 1.0))
            sharpness = float(
                np.var(np.diff(gray, axis=0)) + np.var(np.diff(gray, axis=1))
            )
            rec.update(
                {
                    "is_readable": True,
                    "width": int(w),
                    "height": int(h),
                    "aspect_ratio": round(w / h, 4),
                    "brightness_mean": round(float(gray.mean()), 3),
                    "brightness_std": round(float(gray.std()), 3),
                    "saturation_mean": round(float(saturation.mean()), 4),
                    "sharpness_proxy": round(sharpness, 3),
                    "file_bytes": path.stat().st_size,
                    "sha1": hashlib.sha1(path.read_bytes()).hexdigest(),
                    "dhash": dhash(im),
                }
            )
    except Exception as exc:  # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {exc}"
    return rec


def ensure_dirs(base: Path) -> dict[str, Path]:
    dirs = {
        "root": base,
        "ex05": base / "ex05_data_audit",
        "ex06": base / "ex06_mobilenet_reference_quick",
        "ex07": base / "ex07_densenet_primary_quick",
        "ex16": base / "ex16_proxy_robustness",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def build_manifest(train_root: Path, train_csv: Path) -> pd.DataFrame:
    meta = pd.read_csv(train_csv)
    meta["image_id"] = meta["image_id"].astype(str)
    meta["label"] = meta["label"].astype(str)
    rows = []
    for path in sorted(train_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        rows.append(
            {
                "image_id": path.name,
                "relative_path": path.relative_to(train_root).as_posix(),
                "image_path": str(path.resolve()),
                "folder_label": path.parent.name,
            }
        )
    manifest = pd.DataFrame(rows)
    manifest = manifest.merge(meta, on="image_id", how="left")
    manifest["primary_disease"] = manifest["label"].fillna(manifest["folder_label"])
    manifest["source_dataset"] = "local_kaggle_paddy"
    manifest["capture_session_id"] = manifest["variety"].fillna("unknown").astype(str) + "_age_" + manifest["age"].fillna("unknown").astype(str)
    manifest["field_id"] = ""
    manifest["province"] = ""
    manifest["season"] = ""
    manifest["review_status"] = np.where(
        manifest["primary_disease"].isin(PEST_OR_REVIEW), "needs_expert_review", "accepted"
    )
    return manifest


def run_ex05(train_root: Path, train_csv: Path, out_dir: Path) -> pd.DataFrame:
    manifest = build_manifest(train_root, train_csv)
    manifest.to_csv(out_dir / "manifest_scanned.csv", index=False)

    stats = []
    t0 = time.time()
    for i, row in manifest.iterrows():
        rec = image_stats(Path(row["image_path"]))
        rec["image_id"] = row["image_id"]
        rec["relative_path"] = row["relative_path"]
        stats.append(rec)
        if (i + 1) % 1000 == 0:
            print(f"EX05 scanned {i + 1}/{len(manifest)} images")
    image_stats_df = pd.DataFrame(stats)
    image_stats_df.to_csv(out_dir / "image_stats.csv", index=False)

    audited = manifest.merge(
        image_stats_df.drop(columns=["relative_path"], errors="ignore"),
        on="image_id",
        how="left",
    )
    audited["is_pest_or_review"] = audited["primary_disease"].isin(PEST_OR_REVIEW)
    audited["is_clean_8class"] = audited["primary_disease"].isin(CANONICAL_8)
    audited["is_exact_duplicate"] = audited.duplicated("sha1", keep="first")
    audited["is_exact_duplicate_group_member"] = audited.duplicated("sha1", keep=False)
    audited["include_ex06_ex07"] = (
        audited["is_clean_8class"]
        & audited["is_readable"].fillna(False)
        & ~audited["is_exact_duplicate"]
    )

    exact_dupes = audited.loc[audited["is_exact_duplicate_group_member"]].sort_values(
        ["sha1", "relative_path"]
    )
    perceptual = audited.loc[audited.duplicated("dhash", keep=False)].sort_values(
        ["dhash", "relative_path"]
    )
    exact_dupes.to_csv(out_dir / "duplicate_exact_sha1.csv", index=False)
    perceptual.to_csv(out_dir / "duplicate_perceptual_dhash_candidates.csv", index=False)
    audited.to_csv(out_dir / "manifest_audited.csv", index=False)
    audited.loc[audited["include_ex06_ex07"]].to_csv(
        out_dir / "manifest_clean_8class_no_exact_dupes.csv", index=False
    )

    class_counts = audited["primary_disease"].value_counts().sort_index().to_dict()
    clean_counts = (
        audited.loc[audited["include_ex06_ex07"], "primary_disease"]
        .value_counts()
        .sort_index()
        .to_dict()
    )
    summary = {
        "experiment_id": "EX05",
        "status": "done",
        "elapsed_seconds": round(time.time() - t0, 2),
        "n_original_images": int(len(audited)),
        "n_classes_raw": int(audited["primary_disease"].nunique()),
        "class_counts_raw": {k: int(v) for k, v in class_counts.items()},
        "clean_8class_counts_after_exact_duplicate_drop": {
            k: int(v) for k, v in clean_counts.items()
        },
        "unreadable_images": int((~audited["is_readable"].fillna(False)).sum()),
        "exact_duplicate_rows_all_members": int(len(exact_dupes)),
        "exact_duplicate_rows_removed_keep_first": int(audited["is_exact_duplicate"].sum()),
        "perceptual_duplicate_candidate_rows": int(len(perceptual)),
        "pest_or_review_rows": int(audited["is_pest_or_review"].sum()),
        "metadata_missing_province_pct": 100.0,
        "metadata_missing_season_pct": 100.0,
        "group_proxy_used": "capture_session_id = variety + age",
        "gate_notes": [
            "No field_id/province/season in the supplied Kaggle-style metadata.",
            "hispa and dead_heart are excluded from clean disease baselines.",
            "Exact duplicate rows are kept in the audit file but excluded from EX06/EX07 manifests.",
        ],
    }
    (out_dir / "ex05_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return audited


@dataclass
class RunConfig:
    experiment_id: str
    architecture: str
    train_per_class: int
    val_per_class: int
    epochs: int
    batch_size: int
    seed: int
    image_size: int = 224
    lr: float = 1e-3
    weighted_loss: bool = False
    augmentation_mode: str = "basic"


def make_split(manifest: pd.DataFrame, config: RunConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = random.Random(config.seed)
    train_parts = []
    val_parts = []
    for label in CANONICAL_8:
        group = manifest.loc[manifest["primary_disease"] == label].copy()
        paths = group.index.tolist()
        rng.shuffle(paths)
        val_idx = paths[: config.val_per_class]
        train_idx = paths[
            config.val_per_class : config.val_per_class + config.train_per_class
        ]
        train_parts.append(group.loc[train_idx])
        val_parts.append(group.loc[val_idx])
    train_df = pd.concat(train_parts).sample(frac=1.0, random_state=config.seed)
    val_df = pd.concat(val_parts).sample(frac=1.0, random_state=config.seed)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def run_torch_training(
    manifest_csv: Path,
    out_dir: Path,
    config: RunConfig,
) -> Path:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision.transforms as T
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models

    class RiceDataset(Dataset):
        def __init__(self, frame: pd.DataFrame, transform):
            self.frame = frame.reset_index(drop=True)
            self.transform = transform
            self.label_to_idx = {label: i for i, label in enumerate(CANONICAL_8)}

        def __len__(self):
            return len(self.frame)

        def __getitem__(self, idx):
            row = self.frame.iloc[idx]
            with Image.open(row["image_path"]) as im:
                image = im.convert("RGB")
            return self.transform(image), self.label_to_idx[row["primary_disease"]]

    manifest = pd.read_csv(manifest_csv)
    if "proxy_split" in manifest.columns:
        train_df = manifest.loc[manifest["proxy_split"] == "train"].copy()
        val_df = manifest.loc[manifest["proxy_split"] == "validation"].copy()
        if train_df.empty or val_df.empty:
            raise ValueError("proxy_split manifest must contain both train and validation rows.")
    else:
        train_df, val_df = make_split(manifest, config)
    train_df.to_csv(out_dir / "train_split.csv", index=False)
    val_df.to_csv(out_dir / "val_split.csv", index=False)

    train_transforms = [
        T.Resize((config.image_size, config.image_size)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
    ]
    if config.augmentation_mode == "central_weather_safe":
        train_transforms.extend(
            [
                T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))], p=0.25),
                T.RandomAdjustSharpness(sharpness_factor=0.7, p=0.25),
                T.RandomPerspective(distortion_scale=0.18, p=0.20),
                T.RandomAutocontrast(p=0.20),
                T.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.15, hue=0.03),
            ]
        )
    train_transforms.extend(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    train_tf = T.Compose(train_transforms)
    val_tf = T.Compose(
        [
            T.Resize((config.image_size, config.image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_loader = DataLoader(
        RiceDataset(train_df, train_tf),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        RiceDataset(val_df, val_tf),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    if config.architecture == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        model = models.mobilenet_v3_small(weights=weights)
        for param in model.features.parameters():
            param.requires_grad = False
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CANONICAL_8))
    elif config.architecture == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT
        model = models.densenet121(weights=weights)
        for param in model.features.parameters():
            param.requires_grad = False
        model.classifier = nn.Linear(model.classifier.in_features, len(CANONICAL_8))
    elif config.architecture == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
        for param in model.features.parameters():
            param.requires_grad = False
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CANONICAL_8))
    elif config.architecture == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        for name, param in model.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, len(CANONICAL_8))
    elif config.architecture == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT
        model = models.mobilenet_v3_large(weights=weights)
        for param in model.features.parameters():
            param.requires_grad = False
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CANONICAL_8))
    else:
        raise ValueError(config.architecture)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if config.weighted_loss:
        counts = train_df["primary_disease"].value_counts().reindex(CANONICAL_8, fill_value=0)
        weights = counts.sum() / (len(CANONICAL_8) * counts.clip(lower=1))
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(weights.values, dtype=torch.float32, device=device)
        )
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=config.lr, weight_decay=1e-4
    )

    history = []
    best_macro_f1 = -1.0
    best_path = out_dir / f"{config.experiment_id.lower()}_{config.architecture}.pt"
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = 0.0
        y_train_true = []
        y_train_pred = []
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * int(labels.size(0))
            y_train_true.extend(labels.cpu().numpy().tolist())
            y_train_pred.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())

        model.eval()
        val_loss = 0.0
        y_true = []
        y_pred = []
        y_prob = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                probs = torch.softmax(logits, dim=1)
                val_loss += float(loss.item()) * int(labels.size(0))
                y_true.extend(labels.cpu().numpy().tolist())
                y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
                y_prob.extend(probs.cpu().numpy().tolist())

        epoch_row = {
            "epoch": epoch,
            "train_loss": train_loss / len(train_df),
            "val_loss": val_loss / len(val_df),
            "train_accuracy": accuracy_score(y_train_true, y_train_pred),
            "val_accuracy": accuracy_score(y_true, y_pred),
            "val_balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "val_macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "val_macro_recall": recall_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
        }
        history.append(epoch_row)
        print(json.dumps(epoch_row, indent=2))
        if epoch_row["val_macro_f1"] > best_macro_f1:
            best_macro_f1 = epoch_row["val_macro_f1"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": CANONICAL_8,
                    "config": config.__dict__,
                },
                best_path,
            )

    pred_df = val_df.copy()
    pred_df["y_true"] = [CANONICAL_8[i] for i in y_true]
    pred_df["y_pred"] = [CANONICAL_8[i] for i in y_pred]
    pred_df["confidence"] = [float(max(row)) for row in y_prob]
    pred_df.to_csv(out_dir / "validation_predictions.csv", index=False)

    pd.DataFrame(history).to_csv(out_dir / "train_log.csv", index=False)
    report = classification_report(
        y_true, y_pred, target_names=CANONICAL_8, zero_division=0, output_dict=True
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CANONICAL_8))))
    summary = {
        "experiment_id": config.experiment_id,
        "status": "done_quick_cpu_run",
        "architecture": config.architecture,
        "device": str(device),
        "classes": CANONICAL_8,
        "train_images": int(len(train_df)),
        "val_images": int(len(val_df)),
        "epochs": config.epochs,
        "weighted_loss": config.weighted_loss,
        "augmentation_mode": config.augmentation_mode,
        "best_macro_f1": float(best_macro_f1),
        "final_metrics": history[-1],
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "limitations": [
            "Quick frozen-backbone CPU run, not the full suggested training budget.",
            "No field_id/province/season holdout is possible from the supplied metadata.",
        ],
    }
    (out_dir / f"{config.experiment_id.lower()}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return best_path


def run_ex16(ex07_dir: Path, out_dir: Path) -> None:
    pred_path = ex07_dir / "validation_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    df = pd.read_csv(pred_path)
    rows = []
    group_cols = [col for col in ["variety", "age"] if col in df.columns]
    for col in group_cols:
        for value, group in df.groupby(col, dropna=False):
            if len(group) < 5:
                continue
            rows.append(
                {
                    "group_column": col,
                    "group_value": str(value),
                    "n": int(len(group)),
                    "accuracy": accuracy_score(group["y_true"], group["y_pred"]),
                    "macro_f1": f1_score(
                        group["y_true"], group["y_pred"], average="macro", zero_division=0
                    ),
                    "macro_recall": recall_score(
                        group["y_true"], group["y_pred"], average="macro", zero_division=0
                    ),
                    "mean_confidence": float(group["confidence"].mean()),
                }
            )
    proxy = pd.DataFrame(rows).sort_values(["group_column", "macro_f1", "n"])
    proxy.to_csv(out_dir / "proxy_group_metrics.csv", index=False)
    summary = {
        "experiment_id": "EX16",
        "status": "done_proxy_only",
        "source_predictions": str(pred_path),
        "group_columns": group_cols,
        "note": "The requested field/province/season robustness gate cannot be run because those columns are absent. This proxy uses variety and age from train.csv.",
        "worst_groups": proxy.head(10).to_dict(orient="records") if len(proxy) else [],
    }
    (out_dir / "ex16_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", default="data/train_images")
    parser.add_argument("--train-csv", default="train.csv")
    parser.add_argument("--output-root", default="reports/suggested_runs")
    parser.add_argument("--skip-models", action="store_true")
    parser.add_argument("--ex07-train-per-class", type=int, default=40)
    parser.add_argument("--ex07-val-per-class", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    dirs = ensure_dirs(Path(args.output_root))
    audited = run_ex05(Path(args.train_root), Path(args.train_csv), dirs["ex05"])
    manifest_clean = dirs["ex05"] / "manifest_clean_8class_no_exact_dupes.csv"

    if args.skip_models:
        return

    ex06_cfg = RunConfig(
        experiment_id="EX06",
        architecture="mobilenet_v3_small",
        train_per_class=35,
        val_per_class=5,
        epochs=args.epochs,
        batch_size=16,
        seed=42,
        lr=1e-3,
    )
    run_torch_training(manifest_clean, dirs["ex06"], ex06_cfg)

    ex07_cfg = RunConfig(
        experiment_id="EX07",
        architecture="densenet121",
        train_per_class=args.ex07_train_per_class,
        val_per_class=args.ex07_val_per_class,
        epochs=args.epochs,
        batch_size=8,
        seed=43,
        lr=1e-3,
    )
    run_torch_training(manifest_clean, dirs["ex07"], ex07_cfg)
    run_ex16(dirs["ex07"], dirs["ex16"])


if __name__ == "__main__":
    main()
