from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_suggested_p0 import RunConfig, run_torch_training


BASE = Path("reports/suggested_runs")
NEXT = BASE / "next_steps"
SUBSET = NEXT / "proxy_group_training_subset.csv"
FULL_PROXY = NEXT / "proxy_group_split_manifest.csv"


RUNS = [
    (
        "EX08",
        BASE / "ex08_efficientnet_b0_proxy",
        SUBSET,
        RunConfig(
            experiment_id="EX08",
            architecture="efficientnet_b0",
            train_per_class=120,
            val_per_class=30,
            epochs=4,
            batch_size=32,
            seed=3008,
            lr=1e-3,
        ),
    ),
    (
        "EX09",
        BASE / "ex09_resnet50_proxy",
        SUBSET,
        RunConfig(
            experiment_id="EX09",
            architecture="resnet50",
            train_per_class=120,
            val_per_class=30,
            epochs=4,
            batch_size=32,
            seed=3009,
            lr=1e-3,
        ),
    ),
    (
        "EX10",
        BASE / "ex10_mobilenet_v3_large_proxy",
        SUBSET,
        RunConfig(
            experiment_id="EX10",
            architecture="mobilenet_v3_large",
            train_per_class=120,
            val_per_class=30,
            epochs=4,
            batch_size=32,
            seed=3010,
            lr=1e-3,
        ),
    ),
    (
        "EX11",
        BASE / "ex11_mobilenet_weighted_proxy_full",
        FULL_PROXY,
        RunConfig(
            experiment_id="EX11",
            architecture="mobilenet_v3_small",
            train_per_class=0,
            val_per_class=0,
            epochs=2,
            batch_size=32,
            seed=3011,
            lr=1e-3,
            weighted_loss=True,
        ),
    ),
    (
        "EX12",
        BASE / "ex12_mobilenet_central_aug_proxy",
        SUBSET,
        RunConfig(
            experiment_id="EX12",
            architecture="mobilenet_v3_small",
            train_per_class=120,
            val_per_class=30,
            epochs=4,
            batch_size=32,
            seed=3012,
            lr=1e-3,
            augmentation_mode="central_weather_safe",
        ),
    ),
]


def load_summary(out_dir: Path, experiment_id: str) -> dict:
    path = out_dir / f"{experiment_id.lower()}_summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def write_additional_report(summaries: list[dict]) -> None:
    rows = []
    for item in summaries:
        metrics = item["final_metrics"]
        rows.append(
            {
                "id": item["experiment_id"],
                "architecture": item["architecture"],
                "train_images": item["train_images"],
                "val_images": item["val_images"],
                "epochs": item["epochs"],
                "weighted_loss": item.get("weighted_loss", False),
                "augmentation_mode": item.get("augmentation_mode", "basic"),
                "best_macro_f1": item["best_macro_f1"],
                "final_acc": metrics["val_accuracy"],
                "final_macro_f1": metrics["val_macro_f1"],
                "final_macro_recall": metrics["val_macro_recall"],
            }
        )
    rows = sorted(rows, key=lambda x: x["best_macro_f1"], reverse=True)
    table = "\n".join(
        [
            "| Run | Model | Train | Val | Epochs | Weighted | Augmentation | Best F1 | Final acc | Final F1 | Final recall |",
            "|---|---|---:|---:|---:|---|---|---:|---:|---:|---:|",
        ]
        + [
            f"| {r['id']} | `{r['architecture']}` | {r['train_images']} | {r['val_images']} | {r['epochs']} | {r['weighted_loss']} | `{r['augmentation_mode']}` | {r['best_macro_f1']:.4f} | {r['final_acc']:.4f} | {r['final_macro_f1']:.4f} | {r['final_macro_recall']:.4f} |"
            for r in rows
        ]
    )
    text = f"""# Báo cáo chạy thêm các EX image-only

Ngày tạo: 2026-06-27

## Bối cảnh

Bạn đã duyệt `duplicate_review_gallery_top_groups.jpg`, vì vậy tôi tiếp tục theo hướng không dùng metadata hiện trường. Dataset hiện tại không có `province`, `season`, `field_id`, `capture_session_id` thật, nên các thí nghiệm cần metadata như EX13 không chạy được đúng nghĩa.

Phần `Data preprocessing.docx` vẫn dùng được một phần:

- Dùng được: label policy, quarantine nhãn mơ hồ, resize, normalize, train-only augmentation, duplicate review, split trước augmentation.
- Không dùng được đúng nghĩa: phân tích theo tỉnh/mùa/vụ, metadata fusion, leave-one-province/season-out, severity/risk-factor modeling.

## Kết quả các run mới

{table}

## Nhận xét nhanh

- Các run trên đều là image-only và vẫn chạy CPU/frozen backbone, vì vậy chưa phải final benchmark.
- EX11 dùng weighted loss trên proxy split đầy đủ, nhưng do thiếu field metadata nên vẫn chỉ là ablation kỹ thuật.
- EX12 dùng augmentation chung kiểu điều kiện miền Trung ở mức an toàn. Không dùng class-specific augmentation vì nếu không có metadata hiện trường và review chuyên gia thì dễ tạo shortcut nhãn.
- EX13 metadata branch bị bỏ qua vì dataset không có metadata đủ tin cậy.

## Cách cải thiện khi không có metadata

1. Dùng group proxy chặt hơn:
   - Hiện dùng `variety + age`.
   - Có thể gom thêm theo near-duplicate hash để đảm bảo ảnh gần trùng không qua cả train và val.

2. Tăng chất lượng split:
   - Freeze một validation set theo proxy group.
   - Giữ cùng split cho tất cả architecture để so sánh công bằng.
   - Báo cáo rõ đây là proxy split, không phải field split.

3. Tăng chất lượng training image-only:
   - Unfreeze từng phần backbone sau vài epoch.
   - Dùng early stopping theo macro F1.
   - Dùng balanced sampler hoặc class weights trên full clean manifest.
   - Lưu confusion matrix và error gallery.

4. Dùng augmentation an toàn:
   - Shared augmentation cho mọi lớp: brightness/contrast nhẹ, blur nhẹ, perspective nhẹ, JPEG/noise nhẹ.
   - Không dùng augmentation riêng theo từng bệnh nếu chưa có bằng chứng thu thập thật.

5. Tạo metadata tối thiểu từ ảnh nếu bắt buộc:
   - Chỉ có thể tạo pseudo metadata như `source_folder`, `variety`, `age`, `dhash_group`, `image_quality_bin`.
   - Không nên gọi đó là metadata hiện trường.

## Yêu cầu cần làm thêm

- Nếu muốn claim Central Vietnam field dataset: cần thu/bổ sung `field_id`, `capture_session_id`, `province`, `season`.
- Nếu chỉ làm Kaggle baseline: bỏ qua metadata branch, tập trung image-only benchmark và ghi rõ giới hạn.
- Cần chuyên gia xác nhận `label_map.yaml`, nhất là `hispa`, `dead_heart`, và tên tiếng Việt của từng bệnh.
"""
    (BASE / "additional_experiments_report_vi.md").write_text(text, encoding="utf-8")
    with (BASE / "report.md").open("a", encoding="utf-8") as f:
        f.write("\n\n---\n\n")
        f.write(text)
    with (BASE / "report_vi.md").open("a", encoding="utf-8") as f:
        f.write("\n\n---\n\n")
        f.write(text)


def main() -> None:
    summaries = []
    for name, out_dir, manifest, cfg in RUNS:
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Running {name} -> {out_dir}")
        run_torch_training(manifest, out_dir, cfg)
        summaries.append(load_summary(out_dir, cfg.experiment_id))
    write_additional_report(summaries)


if __name__ == "__main__":
    main()
