from __future__ import annotations

import json
import sys
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_suggested_p0 import CANONICAL_8, RunConfig


BASE = Path("reports/suggested_runs")
ROOT = BASE / "full_feature_final_experiments"
MANIFEST = BASE / "next_steps" / "proxy_group_split_manifest.csv"
HEAD_EPOCHS = 80


RUNS = [
    RunConfig("EX06_FF", "mobilenet_v3_small", 0, 0, HEAD_EPOCHS, 512, 5106),
    RunConfig("EX07_FF", "densenet121", 0, 0, HEAD_EPOCHS, 512, 5107),
    RunConfig("EX08_FF", "efficientnet_b0", 0, 0, HEAD_EPOCHS, 512, 5108),
    RunConfig("EX09_FF", "resnet50", 0, 0, HEAD_EPOCHS, 512, 5109),
    RunConfig("EX10_FF", "mobilenet_v3_large", 0, 0, HEAD_EPOCHS, 512, 5110),
    RunConfig(
        "EX11_FF",
        "mobilenet_v3_small",
        0,
        0,
        HEAD_EPOCHS,
        512,
        5111,
        weighted_loss=True,
    ),
    RunConfig(
        "EX12_FF",
        "mobilenet_v3_small",
        0,
        0,
        HEAD_EPOCHS,
        512,
        5112,
        augmentation_mode="central_weather_safe",
    ),
]


class RiceDataset:
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


def make_transforms(config: RunConfig):
    import torch
    import torchvision.transforms as T

    torch.manual_seed(config.seed)
    train_steps = [
        T.Resize((config.image_size, config.image_size)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
    ]
    if config.augmentation_mode == "central_weather_safe":
        train_steps.extend(
            [
                T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))], p=0.25),
                T.RandomAdjustSharpness(sharpness_factor=0.7, p=0.25),
                T.RandomPerspective(distortion_scale=0.18, p=0.20),
                T.RandomAutocontrast(p=0.20),
                T.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.15, hue=0.03),
            ]
        )
    train_steps.extend(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_steps = [
        T.Resize((config.image_size, config.image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    return T.Compose(train_steps), T.Compose(val_steps)


def load_backbone(config: RunConfig):
    import torch
    import torch.nn as nn
    from torchvision import models

    if config.architecture == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

        class Extractor(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.features = model.features
                self.avgpool = model.avgpool

            def forward(self, x):
                x = self.features(x)
                x = self.avgpool(x)
                return torch.flatten(x, 1)

        return Extractor(model)

    if config.architecture == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)

        class Extractor(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.features = model.features
                self.avgpool = model.avgpool

            def forward(self, x):
                x = self.features(x)
                x = self.avgpool(x)
                return torch.flatten(x, 1)

        return Extractor(model)

    if config.architecture == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)

        class Extractor(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.features = model.features
                self.relu = nn.ReLU(inplace=True)
                self.pool = nn.AdaptiveAvgPool2d((1, 1))

            def forward(self, x):
                x = self.features(x)
                x = self.relu(x)
                x = self.pool(x)
                return torch.flatten(x, 1)

        return Extractor(model)

    if config.architecture == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

        class Extractor(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.features = model.features
                self.avgpool = model.avgpool

            def forward(self, x):
                x = self.features(x)
                x = self.avgpool(x)
                return torch.flatten(x, 1)

        return Extractor(model)

    if config.architecture == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        class Extractor(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.body = nn.Sequential(*list(model.children())[:-1])

            def forward(self, x):
                x = self.body(x)
                return torch.flatten(x, 1)

        return Extractor(model)

    raise ValueError(config.architecture)


def extract_features(frame: pd.DataFrame, transform, extractor, device, batch_size: int):
    import torch
    from torch.utils.data import DataLoader

    dataset = RiceDataset(frame, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    features = []
    labels = []
    extractor.eval()
    with torch.no_grad():
        for images, y in loader:
            images = images.to(device)
            feats = extractor(images).cpu().numpy()
            features.append(feats)
            labels.append(y.numpy())
    return np.concatenate(features), np.concatenate(labels)


def train_head(x_train, y_train, x_val, y_val, config: RunConfig, out_dir: Path) -> dict:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    x_val_t = torch.tensor(x_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.long, device=device)

    loader = DataLoader(
        TensorDataset(x_train_t, y_train_t),
        batch_size=config.batch_size,
        shuffle=True,
    )
    head = nn.Linear(x_train.shape[1], len(CANONICAL_8)).to(device)
    if config.weighted_loss:
        counts = np.bincount(y_train, minlength=len(CANONICAL_8)).clip(min=1)
        weights = counts.sum() / (len(CANONICAL_8) * counts)
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32, device=device)
        )
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(head.parameters(), lr=config.lr, weight_decay=1e-4)

    history = []
    best = {"macro_f1": -1.0, "state": None, "epoch": None}
    for epoch in range(1, config.epochs + 1):
        head.train()
        train_loss = 0.0
        train_true = []
        train_pred = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * int(yb.size(0))
            train_true.extend(yb.cpu().numpy().tolist())
            train_pred.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())

        head.eval()
        with torch.no_grad():
            logits = head(x_val_t)
            val_loss = float(criterion(logits, y_val_t).item())
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            y_pred = probs.argmax(axis=1)

        row = {
            "epoch": epoch,
            "train_loss": train_loss / len(y_train),
            "val_loss": val_loss,
            "train_accuracy": accuracy_score(train_true, train_pred),
            "val_accuracy": accuracy_score(y_val, y_pred),
            "val_balanced_accuracy": balanced_accuracy_score(y_val, y_pred),
            "val_macro_f1": f1_score(y_val, y_pred, average="macro", zero_division=0),
            "val_macro_recall": recall_score(y_val, y_pred, average="macro", zero_division=0),
        }
        history.append(row)
        if row["val_macro_f1"] > best["macro_f1"]:
            best = {
                "macro_f1": row["val_macro_f1"],
                "state": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                "epoch": epoch,
            }
        if epoch % 10 == 0 or epoch == 1 or epoch == config.epochs:
            print(json.dumps(row, indent=2))

    if best["state"] is not None:
        head.load_state_dict(best["state"])
    head.eval()
    with torch.no_grad():
        logits = head(x_val_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        y_pred = probs.argmax(axis=1)

    torch.save(
        {
            "head_state": head.state_dict(),
            "classes": CANONICAL_8,
            "config": config.__dict__,
            "best_epoch": best["epoch"],
        },
        out_dir / f"{config.experiment_id.lower()}_{config.architecture}_head.pt",
    )
    pd.DataFrame(history).to_csv(out_dir / "train_log.csv", index=False)
    pred_df = pd.DataFrame(
        {
            "y_true": [CANONICAL_8[i] for i in y_val],
            "y_pred": [CANONICAL_8[i] for i in y_pred],
            "confidence": probs.max(axis=1),
        }
    )
    pred_df.to_csv(out_dir / "validation_predictions.csv", index=False)

    final_metrics = history[-1]
    best_metrics = history[int(best["epoch"]) - 1]
    return {
        "experiment_id": config.experiment_id,
        "status": "done_full_frozen_feature_benchmark",
        "architecture": config.architecture,
        "device": str(device),
        "classes": CANONICAL_8,
        "train_images": int(len(y_train)),
        "val_images": int(len(y_val)),
        "epochs": config.epochs,
        "weighted_loss": config.weighted_loss,
        "augmentation_mode": config.augmentation_mode,
        "best_epoch": int(best["epoch"]),
        "best_macro_f1": float(best["macro_f1"]),
        "best_metrics": best_metrics,
        "final_metrics": final_metrics,
        "classification_report": classification_report(
            y_val, y_pred, target_names=CANONICAL_8, zero_division=0, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(
            y_val, y_pred, labels=list(range(len(CANONICAL_8)))
        ).tolist(),
        "limitations": [
            "Frozen backbone features with trained linear head, used because full end-to-end CPU runs were too slow.",
            "No field_id/province/season holdout is possible from the supplied metadata.",
        ],
    }


def run_config(config: RunConfig) -> dict:
    import torch

    out_dir = ROOT / config.experiment_id.lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"{config.experiment_id.lower()}_summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    frame = pd.read_csv(MANIFEST)
    train_df = frame.loc[frame["proxy_split"] == "train"].copy()
    val_df = frame.loc[frame["proxy_split"] == "validation"].copy()
    train_df.to_csv(out_dir / "train_split.csv", index=False)
    val_df.to_csv(out_dir / "val_split.csv", index=False)

    train_tf, val_tf = make_transforms(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = load_backbone(config).to(device)
    for p in extractor.parameters():
        p.requires_grad = False

    t0 = time.time()
    print(f"Extracting features for {config.experiment_id} {config.architecture}")
    x_train, y_train = extract_features(train_df, train_tf, extractor, device, batch_size=64)
    x_val, y_val = extract_features(val_df, val_tf, extractor, device, batch_size=64)
    np.save(out_dir / "train_features.npy", x_train)
    np.save(out_dir / "train_labels.npy", y_train)
    np.save(out_dir / "val_features.npy", x_val)
    np.save(out_dir / "val_labels.npy", y_val)

    print(f"Training head for {config.experiment_id}")
    summary = train_head(x_train, y_train, x_val, y_val, config, out_dir)
    summary["feature_dim"] = int(x_train.shape[1])
    summary["elapsed_seconds"] = round(time.time() - t0, 2)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def per_class_table(summary: dict) -> str:
    rows = [
        "| Lớp | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in summary["classes"]:
        m = summary["classification_report"][label]
        rows.append(
            f"| `{label}` | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1-score']:.4f} | {int(m['support'])} |"
        )
    return "\n".join(rows)


def write_report(summaries: list[dict]) -> None:
    ranked = sorted(
        summaries,
        key=lambda x: (x["best_macro_f1"], x["best_metrics"]["val_accuracy"]),
        reverse=True,
    )
    best = ranked[0]
    result_table = "\n".join(
        [
            "| Hạng | Run | Model | Train | Val | Head epochs | Best epoch | Weighted | Augmentation | Best macro F1 | Best acc | Best balanced acc | Best recall |",
            "|---:|---|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|",
        ]
        + [
            f"| {i + 1} | {s['experiment_id']} | `{s['architecture']}` | {s['train_images']} | {s['val_images']} | {s['epochs']} | {s['best_epoch']} | {s['weighted_loss']} | `{s['augmentation_mode']}` | {s['best_macro_f1']:.4f} | {s['best_metrics']['val_accuracy']:.4f} | {s['best_metrics']['val_balanced_accuracy']:.4f} | {s['best_metrics']['val_macro_recall']:.4f} |"
            for i, s in enumerate(ranked)
        ]
    )
    text = f"""# Báo cáo chốt mô hình tối ưu - full frozen-feature EX06-EX12

Ngày tạo: 2026-06-27

## Thiết lập chạy cuối

- Dữ liệu: toàn bộ `proxy_group_split_manifest.csv`.
- Train: 6,451 ảnh.
- Validation: 864 ảnh.
- Split: proxy group `capture_session_id = variety + age`, không rò rỉ group.
- Cách chạy: frozen backbone feature extraction + linear head 80 epoch.
- Lý do dùng cách này: full end-to-end 6 epoch trên CPU bị nghẽn ở DenseNet; frozen-feature benchmark vẫn dùng toàn bộ dataset và phù hợp với các backbone đang frozen.
- Bài toán: image-only 8 lớp, bỏ `hispa` và `dead_heart`.

## Bảng chốt mô hình

{result_table}

## Mô hình tối ưu nên chọn

Chọn: **{best['experiment_id']} - `{best['architecture']}`**.

Lý do chọn:

- Có `best_macro_f1` cao nhất trên cùng full proxy split.
- Không cần metadata hiện trường.
- Là lựa chọn tốt nhất trong bộ EX06-EX12 với điều kiện dataset hiện tại.

## Per-class report của mô hình được chọn

{per_class_table(best)}

## Lỗi/giới hạn hiện tại

1. Dataset không có `field_id`, `province`, `season`, `capture_session_id` thật.
   - Không thể kết luận tổng quát hóa theo ruộng/tỉnh/mùa vụ.

2. Validation proxy bị lệch lớp.
   - `brown_spot` và `downy_mildew` có support validation thấp.
   - Macro F1 vẫn là metric nên dùng, nhưng cần đọc cùng support từng lớp.

3. Đây là frozen-feature benchmark.
   - Dùng để chốt hướng mô hình tối ưu.
   - Muốn mô hình production hơn thì cần fine-tune mô hình được chọn bằng end-to-end training hoặc GPU.

4. Phần preprocessing metadata trong `Data preprocessing.docx` không dùng được đúng nghĩa vì dataset thiếu metadata.
   - Bỏ qua metadata fusion và province/season holdout.
   - Giữ lại preprocessing image-only: resize, normalize, train-only augmentation, duplicate policy, label policy.

## Cách cải thiện tiếp theo

1. Fine-tune end-to-end mô hình được chọn trên GPU.
2. Tạo split mạnh hơn bằng `variety + age + dhash_group`.
3. Sinh error gallery cho các mẫu validation sai.
4. Nếu muốn đúng Central Vietnam dataset, bổ sung metadata hiện trường thật.

## Artifacts

- Root: `reports/suggested_runs/full_feature_final_experiments/`
- Report: `reports/suggested_runs/final_model_selection_feature_report_vi.md`
- Summary từng run: `reports/suggested_runs/full_feature_final_experiments/<run>/<run>_summary.json`
"""
    (BASE / "final_model_selection_feature_report_vi.md").write_text(text, encoding="utf-8")
    with (BASE / "report.md").open("a", encoding="utf-8") as f:
        f.write("\n\n---\n\n")
        f.write(text)
    with (BASE / "report_vi.md").open("a", encoding="utf-8") as f:
        f.write("\n\n---\n\n")
        f.write(text)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    summaries = []
    for cfg in RUNS:
        summaries.append(run_config(cfg))
    write_report(summaries)


if __name__ == "__main__":
    main()
