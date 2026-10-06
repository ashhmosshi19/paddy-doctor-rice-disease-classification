from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = Path("DenseNet121_Rice_Disease_Pipeline.ipynb")


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip("\n").split("\n")],
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.strip("\n").split("\n")],
    }


cells = [
    md(
        """
# Rice Disease Classification - DenseNet121 From Scratch

Đây là notebook template cho học sinh tự xây dựng DenseNet121 bằng PyTorch, tham khảo tinh thần từ `Lab_5_1_ViT_template.ipynb`.

Nhiệm vụ của học sinh:

1. Hoàn thiện các khối được đánh dấu `### START CODE HERE ###` và `### END CODE HERE ###`.
2. Không dùng `torchvision.models.densenet121`, không dùng pretrained ImageNet.
3. Tự viết từng block chính của DenseNet: `DenseLayer`, `DenseBlock`, `TransitionLayer`, `DenseNetFromScratch`.
4. Chạy lại toàn bộ notebook sau khi đã thay hết `NotImplementedError`.

Notebook này dùng pipeline **image-only** vì dataset hiện tại không có metadata hiện trường đáng tin cậy. Các cột như `field_id`, `province`, `season`, `variety`, `age` không được dùng.
"""
    ),
    md(
        """
## 1. Import thư viện

Cell này chỉ dùng PyTorch cơ bản và `torchvision.transforms` cho augmentation. Phần model DenseNet sẽ tự viết từ đầu, không gọi `torchvision.models`.
"""
    ),
    code(
        """
# Nếu thiếu thư viện trong notebook runtime, bỏ comment dòng dưới.
# !pip install torch torchvision pandas numpy pillow scikit-learn matplotlib seaborn tqdm

from pathlib import Path
import json
import random
import time

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)
"""
    ),
    md("## 2. Cấu hình thí nghiệm"),
    code(
        """
PROJECT_ROOT = Path(".").resolve()
MANIFEST_PATH = PROJECT_ROOT / "reports" / "suggested_runs" / "image_only_fix" / "image_only_hash_split_manifest.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "suggested_runs" / "notebook_densenet_from_scratch"
TEST_DIR = PROJECT_ROOT / "data" / "test_images"

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0
SEED = 42

NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
USE_CLASS_WEIGHTS = False

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

seed_everything()
"""
    ),
    md("## 3. Load manifest image-only"),
    code(
        """
manifest = pd.read_csv(MANIFEST_PATH)
required_columns = {"image_path", "primary_disease", "split"}
missing = required_columns - set(manifest.columns)
if missing:
    raise ValueError(f"Manifest thiếu cột: {missing}")

CLASSES = sorted(manifest["primary_disease"].unique().tolist())
LABEL_TO_IDX = {label: idx for idx, label in enumerate(CLASSES)}
IDX_TO_LABEL = {idx: label for label, idx in LABEL_TO_IDX.items()}

train_df = manifest[manifest["split"] == "train"].reset_index(drop=True)
val_df = manifest[manifest["split"] == "val"].reset_index(drop=True)

print("Classes:", CLASSES)
print("Train:", len(train_df), "Val:", len(val_df))
display(pd.crosstab(manifest["primary_disease"], manifest["split"]))
"""
    ),
    md(
        """
## 4. Dataset và DataLoader

Bài tập: hoàn thiện `RiceDiseaseDataset.__getitem__` để đọc ảnh, transform, encode label và trả về `(image, label, index)`.
"""
    ),
    code(
        """
train_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.2),
    T.RandomRotation(degrees=12),
    T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class RiceDiseaseDataset(Dataset):
    def __init__(self, frame, label_to_idx, transform=None):
        self.frame = frame.reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]

        # ### START CODE HERE ### (Approx. 5-8 lines)
        # Gợi ý:
        # 1. Lấy đường dẫn ảnh từ row["image_path"].
        # 2. Mở ảnh bằng PIL và chuyển sang RGB.
        # 3. Áp dụng self.transform nếu có.
        # 4. Đổi primary_disease thành chỉ số class bằng self.label_to_idx.
        # 5. Trả về image, label, index.
        raise NotImplementedError("Hoàn thiện RiceDiseaseDataset.__getitem__")
        # ### END CODE HERE ###

train_dataset = RiceDiseaseDataset(train_df, LABEL_TO_IDX, train_transform)
val_dataset = RiceDiseaseDataset(val_df, LABEL_TO_IDX, val_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

print("Train batches:", len(train_loader))
print("Val batches:", len(val_loader))
"""
    ),
    md("## 5. Visualize batch mẫu"),
    code(
        """
def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = tensor.cpu() * std + mean
    return image.clamp(0, 1)

def show_batch(loader, n=8):
    images, labels, _ = next(iter(loader))
    n = min(n, len(images))
    rows = 2
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(14, 7))
    axes = axes.ravel()
    for i in range(n):
        img = denormalize(images[i]).permute(1, 2, 0).numpy()
        axes[i].imshow(img)
        axes[i].set_title(IDX_TO_LABEL[int(labels[i])], fontsize=9)
        axes[i].axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    plt.tight_layout()

show_batch(train_loader)
"""
    ),
    md(
        """
## 6. DenseNet121 từ đầu

Cấu trúc chuẩn của DenseNet121:

- Stem: `7x7 conv -> BN -> ReLU -> MaxPool`
- Dense block 1: 6 dense layers
- Transition 1
- Dense block 2: 12 dense layers
- Transition 2
- Dense block 3: 24 dense layers
- Transition 3
- Dense block 4: 16 dense layers
- Global average pooling + classifier

Trong `DenseLayer`, output của layer mới được nối channel-wise với input cũ: `torch.cat([x, new_features], dim=1)`.
"""
    ),
    code(
        """
class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size=4, drop_rate=0.0):
        super().__init__()
        inter_channels = bn_size * growth_rate

        # ### START CODE HERE ### (Approx. 6-8 lines)
        # Gợi ý kiến trúc:
        # BN -> ReLU -> Conv2d 1x1
        # BN -> ReLU -> Conv2d 3x3 padding=1
        # Dropout là optional, dùng nn.Dropout2d nếu drop_rate > 0.
        raise NotImplementedError("Hoàn thiện DenseLayer.__init__")
        # ### END CODE HERE ###

    def forward(self, x):
        # ### START CODE HERE ### (Approx. 4-6 lines)
        # Gợi ý:
        # new_features = conv3x3(relu(bn(conv1x1(relu(bn(x))))))
        # nếu có dropout thì áp dụng lên new_features
        # return torch.cat([x, new_features], dim=1)
        raise NotImplementedError("Hoàn thiện DenseLayer.forward")
        # ### END CODE HERE ###


class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, bn_size=4, drop_rate=0.0):
        super().__init__()

        # ### START CODE HERE ### (Approx. 4-6 lines)
        # Gợi ý:
        # Tạo nn.ModuleList gồm num_layers DenseLayer.
        # Channel đầu vào của layer thứ i là in_channels + i * growth_rate.
        raise NotImplementedError("Hoàn thiện DenseBlock.__init__")
        # ### END CODE HERE ###

    def forward(self, x):
        # ### START CODE HERE ### (Approx. 2-4 lines)
        # Gợi ý: lần lượt đưa x qua từng DenseLayer.
        raise NotImplementedError("Hoàn thiện DenseBlock.forward")
        # ### END CODE HERE ###


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # ### START CODE HERE ### (Approx. 4-6 lines)
        # Gợi ý kiến trúc:
        # BN -> ReLU -> Conv2d 1x1 -> AvgPool2d(kernel_size=2, stride=2)
        raise NotImplementedError("Hoàn thiện TransitionLayer.__init__")
        # ### END CODE HERE ###

    def forward(self, x):
        # ### START CODE HERE ### (Approx. 1 line)
        raise NotImplementedError("Hoàn thiện TransitionLayer.forward")
        # ### END CODE HERE ###


class DenseNetFromScratch(nn.Module):
    def __init__(
        self,
        num_classes,
        growth_rate=32,
        block_layers=(6, 12, 24, 16),
        num_init_features=64,
        bn_size=4,
        drop_rate=0.0,
        compression=0.5,
    ):
        super().__init__()

        # ### START CODE HERE ### (Approx. 25-35 lines)
        # Gợi ý:
        # 1. Tạo stem bằng nn.Sequential:
        #    Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)
        #    BatchNorm2d -> ReLU -> MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 2. Duyệt qua block_layers để tạo DenseBlock và TransitionLayer.
        # 3. Sau mỗi DenseBlock, số channel tăng thêm num_layers * growth_rate.
        # 4. Sau mỗi TransitionLayer, số channel giảm theo compression.
        # 5. Cuối mạng: BatchNorm2d -> ReLU -> AdaptiveAvgPool2d((1, 1)) -> Linear.
        raise NotImplementedError("Hoàn thiện DenseNetFromScratch.__init__")
        # ### END CODE HERE ###

    def forward(self, x):
        # ### START CODE HERE ### (Approx. 4-6 lines)
        # Gợi ý:
        # features = self.features(x)
        # out = self.final_relu(self.final_norm(features))
        # out = self.avgpool(out)
        # flatten và đưa qua classifier.
        raise NotImplementedError("Hoàn thiện DenseNetFromScratch.forward")
        # ### END CODE HERE ###


def build_densenet121_from_scratch(num_classes=len(CLASSES)):
    return DenseNetFromScratch(
        num_classes=num_classes,
        growth_rate=32,
        block_layers=(6, 12, 24, 16),
        num_init_features=64,
        bn_size=4,
        drop_rate=0.0,
        compression=0.5,
    )


def count_trainable(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


model = build_densenet121_from_scratch().to(device)
total_params, trainable_params = count_trainable(model)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
"""
    ),
    md("## 7. Loss và metrics"),
    code(
        """
def make_criterion(train_frame, use_class_weights=False):
    if not use_class_weights:
        return nn.CrossEntropyLoss()

    counts = train_frame["primary_disease"].value_counts().reindex(CLASSES, fill_value=0)
    weights = counts.sum() / (len(CLASSES) * counts.clip(lower=1))
    print("Class weights:")
    display(weights)
    return nn.CrossEntropyLoss(weight=torch.tensor(weights.values, dtype=torch.float32, device=device))

criterion = make_criterion(train_df, USE_CLASS_WEIGHTS)
scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }
"""
    ),
    md(
        """
## 8. Training loop tự viết

Bài tập: hoàn thiện một epoch train và evaluate. Không cần viết lại optimizer hay loss, chỉ cần đúng vòng lặp PyTorch chuẩn.
"""
    ),
    code(
        """
def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels, _ in tqdm(dataloader, desc="Train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # ### START CODE HERE ### (Approx. 10-14 lines)
        # Gợi ý:
        # optimizer.zero_grad(set_to_none=True)
        # autocast nếu có CUDA
        # logits = model(images)
        # loss = criterion(logits, labels)
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        # cập nhật total_loss, all_preds, all_labels
        raise NotImplementedError("Hoàn thiện train_one_epoch")
        # ### END CODE HERE ###

    metrics = compute_metrics(all_labels, all_preds)
    return total_loss / len(dataloader.dataset), metrics


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs, all_indices = [], [], [], []

    with torch.no_grad():
        for images, labels, indices in tqdm(dataloader, desc="Eval", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # ### START CODE HERE ### (Approx. 8-12 lines)
            # Gợi ý:
            # logits = model(images)
            # loss = criterion(logits, labels)
            # probs = torch.softmax(logits, dim=1)
            # preds = logits.argmax(dim=1)
            # cập nhật total_loss, all_preds, all_labels, all_probs, all_indices
            raise NotImplementedError("Hoàn thiện evaluate")
            # ### END CODE HERE ###

    metrics = compute_metrics(all_labels, all_preds)
    predictions = pd.DataFrame({
        "row_index": all_indices,
        "true_idx": all_labels,
        "pred_idx": all_preds,
        "true_label": [IDX_TO_LABEL[i] for i in all_labels],
        "pred_label": [IDX_TO_LABEL[i] for i in all_preds],
    })
    for class_idx, class_name in IDX_TO_LABEL.items():
        predictions[f"prob_{class_name}"] = [prob[class_idx] for prob in all_probs]

    return total_loss / len(dataloader.dataset), metrics, predictions
"""
    ),
    md("## 9. Fit model"),
    code(
        """
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

history = []
best_macro_f1 = -1.0
best_path = OUTPUT_DIR / "best_densenet121_from_scratch.pt"

for epoch in range(1, NUM_EPOCHS + 1):
    start_time = time.time()
    train_loss, train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
    val_loss, val_metrics, val_predictions = evaluate(model, val_loader, criterion, device)
    scheduler.step()

    row = {
        "epoch": epoch,
        "lr": optimizer.param_groups[0]["lr"],
        "train_loss": train_loss,
        "val_loss": val_loss,
        **{f"train_{k}": v for k, v in train_metrics.items()},
        **{f"val_{k}": v for k, v in val_metrics.items()},
        "seconds": time.time() - start_time,
    }
    history.append(row)
    print(
        f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
        f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
        f"val_macro_f1={val_metrics['macro_f1']:.4f} | val_acc={val_metrics['accuracy']:.4f}"
    )

    if val_metrics["macro_f1"] > best_macro_f1:
        best_macro_f1 = val_metrics["macro_f1"]
        torch.save(
            {
                "model_state": model.state_dict(),
                "classes": CLASSES,
                "image_size": IMAGE_SIZE,
                "epoch": epoch,
                "best_macro_f1": best_macro_f1,
            },
            best_path,
        )
        val_predictions.to_csv(OUTPUT_DIR / "validation_predictions.csv", index=False)

history_df = pd.DataFrame(history)
history_df.to_csv(OUTPUT_DIR / "train_log.csv", index=False)
print("Best macro F1:", best_macro_f1)
print("Saved:", best_path)
"""
    ),
    md("## 10. Learning curves"),
    code(
        """
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history_df["epoch"], history_df["train_loss"], label="train")
axes[0].plot(history_df["epoch"], history_df["val_loss"], label="val")
axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(history_df["epoch"], history_df["val_macro_f1"], label="val macro F1")
axes[1].plot(history_df["epoch"], history_df["val_accuracy"], label="val accuracy")
axes[1].set_title("Validation metrics")
axes[1].set_xlabel("Epoch")
axes[1].legend()
plt.tight_layout()
plt.show()
"""
    ),
    md("## 11. Evaluation cuối"),
    code(
        """
checkpoint = torch.load(best_path, map_location=device)
model.load_state_dict(checkpoint["model_state"])
val_loss, val_metrics, val_predictions = evaluate(model, val_loader, criterion, device)

print("Validation metrics:")
print(json.dumps(val_metrics, indent=2, ensure_ascii=False))

print("\\nClassification report:")
print(classification_report(
    val_predictions["true_label"],
    val_predictions["pred_label"],
    labels=CLASSES,
    zero_division=0,
))

cm = confusion_matrix(val_predictions["true_label"], val_predictions["pred_label"], labels=CLASSES)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

summary = {
    "best_macro_f1": float(best_macro_f1),
    "final_val_loss": float(val_loss),
    "final_val_metrics": {k: float(v) for k, v in val_metrics.items()},
    "classes": CLASSES,
    "note": "DenseNet121 from scratch; no torchvision.models.densenet121; image-only manifest.",
}
with open(OUTPUT_DIR / "densenet121_from_scratch_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
"""
    ),
    md("## 12. Inference demo và submission"),
    code(
        """
def predict_single_image(model, image_path, transform, device):
    model.eval()

    # ### START CODE HERE ### (Approx. 7-10 lines)
    # Gợi ý:
    # mở ảnh RGB, transform, thêm batch dimension
    # đưa tensor lên device
    # chạy model trong torch.no_grad()
    # softmax lấy xác suất, argmax lấy class dự đoán
    # return predicted_label, confidence
    raise NotImplementedError("Hoàn thiện predict_single_image")
    # ### END CODE HERE ###


test_images = sorted(TEST_DIR.glob("*.jpg")) + sorted(TEST_DIR.glob("*.png"))
print("Test images:", len(test_images))

submission_rows = []
for image_path in tqdm(test_images, desc="Predict test"):
    label, confidence = predict_single_image(model, image_path, val_transform, device)
    submission_rows.append({
        "image_id": image_path.stem,
        "label": label,
        "confidence": confidence,
    })

submission = pd.DataFrame(submission_rows)
submission.to_csv(OUTPUT_DIR / "submission_from_scratch.csv", index=False)
display(submission.head())
"""
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Wrote {NOTEBOOK}")
