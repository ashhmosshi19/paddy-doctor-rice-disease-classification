from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision import models


BASE = Path("reports/suggested_runs")
MANIFEST = BASE / "image_only_fix" / "image_only_hash_split_manifest.csv"
OUT = BASE / "image_only_fix" / "densenet_gpu_no_weights"
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = [
    "bacterial_leaf_blight",
    "bacterial_leaf_streak",
    "bacterial_panicle_blight",
    "blast",
    "brown_spot",
    "downy_mildew",
    "normal",
    "tungro",
]

IMG_SIZE = 224
BATCH_SIZE = 48
PHASE1_EPOCHS = 5
PHASE2_EPOCHS = 7
SEED = 7307


class RiceDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.label_to_idx = {label: i for i, label in enumerate(CLASSES)}

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        with Image.open(row["image_path"]) as im:
            image = im.convert("RGB")
        return self.transform(image), self.label_to_idx[row["primary_disease"]], idx


def build_transforms():
    train_tf = T.Compose(
        [
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(18),
            T.ColorJitter(brightness=0.16, contrast=0.16, saturation=0.10, hue=0.02),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_tf = T.Compose(
        [
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_tf, val_tf


def make_loaders():
    frame = pd.read_csv(MANIFEST)
    train_df = frame.loc[frame["split"] == "train"].copy()
    val_df = frame.loc[frame["split"] == "validation"].copy()
    train_df.to_csv(OUT / "train_split.csv", index=False)
    val_df.to_csv(OUT / "val_split.csv", index=False)
    train_tf, val_tf = build_transforms()
    train_loader = DataLoader(
        RiceDataset(train_df, train_tf),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        RiceDataset(val_df, val_tf),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    return train_df, val_df, train_loader, val_loader


def build_model():
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    model.classifier = nn.Linear(model.classifier.in_features, len(CLASSES))
    return model


def set_phase(model: nn.Module, phase: int):
    for param in model.features.parameters():
        param.requires_grad = False
    if phase == 2:
        for name, param in model.features.named_parameters():
            if name.startswith("denseblock4") or name.startswith("norm5"):
                param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    y_true, y_pred, y_prob, indices_all = [], [], [], []
    with torch.no_grad():
        for images, labels, indices in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)
            total_loss += float(loss.item()) * int(labels.size(0))
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            y_prob.extend(probs.cpu().numpy().tolist())
            indices_all.extend(indices.numpy().tolist())
    return {
        "loss": total_loss / len(loader.dataset),
        "y_true": np.array(y_true),
        "y_pred": np.array(y_pred),
        "y_prob": np.array(y_prob),
        "indices": indices_all,
    }


def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    y_true, y_pred = [], []
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=True):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.item()) * int(labels.size(0))
        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())
    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def row_metrics(epoch, phase, lr, train_stats, val_stats):
    y_true = val_stats["y_true"]
    y_pred = val_stats["y_pred"]
    return {
        "epoch": epoch,
        "phase": phase,
        "lr": lr,
        "train_loss": train_stats["loss"],
        "train_accuracy": train_stats["accuracy"],
        "val_loss": val_stats["loss"],
        "val_accuracy": accuracy_score(y_true, y_pred),
        "val_balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "val_macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "val_macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def write_reports(summary: dict, report: dict):
    rows = [
        "| Lớp | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in CLASSES:
        m = report[label]
        rows.append(
            f"| `{label}` | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1-score']:.4f} | {int(m['support'])} |"
        )

    text = f"""# Báo cáo DenseNet GPU image-only fixed

Ngày tạo: 2026-06-27

## Sửa lỗi đã áp dụng

- Loại metadata không cần thiết khỏi manifest train.
- Không dùng `field_id`, `province`, `season`, `variety`, `age`, `capture_session_id`.
- Split mới dùng nhãn + `dhash` để giảm duplicate leakage.
- Bỏ class-weighted loss để tránh over-predict lớp hiếm như `downy_mildew`.

## Thiết lập

- Model: DenseNet121.
- GPU: {summary["device"]}.
- Train: {summary["train_images"]} ảnh.
- Validation: {summary["val_images"]} ảnh.
- Batch size: {summary["batch_size"]}.
- Phase 1: {summary["phase1_epochs"]} epoch train classifier head.
- Phase 2: {summary["phase2_epochs"]} epoch fine-tune `denseblock4` + `norm5`.
- Loss: plain cross entropy.

## Kết quả tốt nhất

| Metric | Value |
|---|---:|
| Best epoch | {summary["best_epoch"]} |
| Best macro F1 | {summary["best_macro_f1"]:.4f} |
| Validation accuracy | {summary["best_metrics"]["val_accuracy"]:.4f} |
| Validation balanced accuracy | {summary["best_metrics"]["val_balanced_accuracy"]:.4f} |
| Validation macro recall | {summary["best_metrics"]["val_macro_recall"]:.4f} |
| Runtime seconds | {summary["elapsed_seconds"]:.2f} |

## Per-class validation report

{chr(10).join(rows)}

## Nhận xét

- Đây là run sửa lỗi hợp lý nhất sau khi bỏ metadata giả và bỏ class weights.
- Nếu accuracy/macro F1 ổn hơn bản cũ, nên dùng split/manifest này cho các lần benchmark tiếp theo.
- Nếu lớp nào vẫn F1 thấp, bước tiếp theo là tạo error gallery cho riêng lớp đó.
"""
    (OUT / "densenet_gpu_image_only_fixed_report_vi.md").write_text(text, encoding="utf-8")
    with (BASE / "report.md").open("a", encoding="utf-8") as f:
        f.write("\n\n---\n\n")
        f.write(text)
    with (BASE / "report_vi.md").open("a", encoding="utf-8") as f:
        f.write("\n\n---\n\n")
        f.write(text)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Use the cu128 PyTorch command in repro_command.md.")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    train_df, val_df, train_loader, val_loader = make_loaders()
    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    history = []
    best = {"macro_f1": -1.0, "epoch": None, "path": OUT / "best_densenet121_gpu_no_weights.pt"}
    start = time.time()
    global_epoch = 0
    for phase, epochs, lr in [(1, PHASE1_EPOCHS, 1e-3), (2, PHASE2_EPOCHS, 1e-4)]:
        set_phase(model, phase)
        optimizer = optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=1e-4,
        )
        for _ in range(epochs):
            global_epoch += 1
            train_stats = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
            val_stats = evaluate(model, val_loader, criterion, device)
            row = row_metrics(global_epoch, phase, lr, train_stats, val_stats)
            history.append(row)
            print(json.dumps(row, indent=2))
            if row["val_macro_f1"] > best["macro_f1"]:
                best.update({"macro_f1": row["val_macro_f1"], "epoch": global_epoch})
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "classes": CLASSES,
                        "epoch": global_epoch,
                        "history": history,
                    },
                    best["path"],
                )

    pd.DataFrame(history).to_csv(OUT / "train_log.csv", index=False)
    checkpoint = torch.load(best["path"], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    val_stats = evaluate(model, val_loader, criterion, device)
    y_true, y_pred, y_prob = val_stats["y_true"], val_stats["y_pred"], val_stats["y_prob"]
    pred_df = val_df.iloc[val_stats["indices"]].copy()
    pred_df["y_true"] = [CLASSES[i] for i in y_true]
    pred_df["y_pred"] = [CLASSES[i] for i in y_pred]
    pred_df["confidence"] = y_prob.max(axis=1)
    pred_df.to_csv(OUT / "validation_predictions.csv", index=False)

    report = classification_report(
        y_true, y_pred, target_names=CLASSES, zero_division=0, output_dict=True
    )
    summary = {
        "experiment_id": "EX07_DENSENET_GPU_IMAGE_ONLY_FIXED",
        "status": "done_gpu_image_only_no_metadata_no_class_weights",
        "architecture": "densenet121",
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "train_images": int(len(train_df)),
        "val_images": int(len(val_df)),
        "batch_size": BATCH_SIZE,
        "phase1_epochs": PHASE1_EPOCHS,
        "phase2_epochs": PHASE2_EPOCHS,
        "total_epochs": PHASE1_EPOCHS + PHASE2_EPOCHS,
        "loss": "plain_cross_entropy",
        "split": "image_only_label_stratified_dhash_group_split",
        "best_epoch": int(best["epoch"]),
        "best_macro_f1": float(best["macro_f1"]),
        "best_metrics": history[int(best["epoch"]) - 1],
        "classification_report": report,
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(len(CLASSES)))
        ).tolist(),
        "elapsed_seconds": round(time.time() - start, 2),
    }
    (OUT / "densenet_gpu_image_only_fixed_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    write_reports(summary, report)


if __name__ == "__main__":
    main()
