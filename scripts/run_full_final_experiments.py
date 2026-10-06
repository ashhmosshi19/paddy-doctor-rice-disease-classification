from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_suggested_p0 import RunConfig, run_torch_training


BASE = Path("reports/suggested_runs")
FULL_ROOT = BASE / "full_final_experiments"
MANIFEST = BASE / "next_steps" / "proxy_group_split_manifest.csv"
EPOCHS = 6


RUNS = [
    RunConfig("EX06_FULL", "mobilenet_v3_small", 0, 0, EPOCHS, 32, 4106),
    RunConfig("EX07_FULL", "densenet121", 0, 0, EPOCHS, 32, 4107),
    RunConfig("EX08_FULL", "efficientnet_b0", 0, 0, EPOCHS, 32, 4108),
    RunConfig("EX09_FULL", "resnet50", 0, 0, EPOCHS, 32, 4109),
    RunConfig("EX10_FULL", "mobilenet_v3_large", 0, 0, EPOCHS, 32, 4110),
    RunConfig(
        "EX11_FULL",
        "mobilenet_v3_small",
        0,
        0,
        EPOCHS,
        32,
        4111,
        weighted_loss=True,
    ),
    RunConfig(
        "EX12_FULL",
        "mobilenet_v3_small",
        0,
        0,
        EPOCHS,
        32,
        4112,
        augmentation_mode="central_weather_safe",
    ),
]


def summary_path(out_dir: Path, experiment_id: str) -> Path:
    return out_dir / f"{experiment_id.lower()}_summary.json"


def read_summary(out_dir: Path, experiment_id: str) -> dict:
    return json.loads(summary_path(out_dir, experiment_id).read_text(encoding="utf-8"))


def per_class_table(summary: dict) -> str:
    report = summary["classification_report"]
    rows = [
        "| Lớp | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in summary["classes"]:
        m = report[label]
        rows.append(
            f"| `{label}` | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1-score']:.4f} | {int(m['support'])} |"
        )
    return "\n".join(rows)


def write_final_report(summaries: list[dict]) -> None:
    rows = []
    for summary in summaries:
        final = summary["final_metrics"]
        rows.append(
            {
                "id": summary["experiment_id"],
                "model": summary["architecture"],
                "train": summary["train_images"],
                "val": summary["val_images"],
                "epochs": summary["epochs"],
                "weighted": summary["weighted_loss"],
                "augmentation": summary["augmentation_mode"],
                "best_f1": summary["best_macro_f1"],
                "final_acc": final["val_accuracy"],
                "final_bal_acc": final["val_balanced_accuracy"],
                "final_f1": final["val_macro_f1"],
                "final_recall": final["val_macro_recall"],
            }
        )
    rows = sorted(rows, key=lambda r: (r["best_f1"], r["final_f1"]), reverse=True)
    best = rows[0]
    best_summary = next(s for s in summaries if s["experiment_id"] == best["id"])

    result_table = "\n".join(
        [
            "| Hạng | Run | Model | Train | Val | Epochs | Weighted | Augmentation | Best macro F1 | Final acc | Final balanced acc | Final macro F1 | Final macro recall |",
            "|---:|---|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|",
        ]
        + [
            f"| {i + 1} | {r['id']} | `{r['model']}` | {r['train']} | {r['val']} | {r['epochs']} | {r['weighted']} | `{r['augmentation']}` | {r['best_f1']:.4f} | {r['final_acc']:.4f} | {r['final_bal_acc']:.4f} | {r['final_f1']:.4f} | {r['final_recall']:.4f} |"
            for i, r in enumerate(rows)
        ]
    )

    text = f"""# Báo cáo chốt mô hình tối ưu - full EX06-EX12

Ngày tạo: 2026-06-27

## Thiết lập chạy cuối

- Dữ liệu: `reports/suggested_runs/next_steps/proxy_group_split_manifest.csv`.
- Tập train: 6,451 ảnh.
- Tập validation: 864 ảnh.
- Split: proxy group theo `capture_session_id = variety + age`, không có group crossing giữa train và validation.
- Số epoch: {EPOCHS}.
- Thiết bị: CPU theo môi trường PyTorch hiện tại.
- Bài toán: image-only 8 lớp, đã loại `hispa` và `dead_heart`.
- Metadata hiện trường: không có, nên EX13 metadata branch bị bỏ qua.

## Bảng kết quả cuối

{result_table}

## Mô hình được đề xuất

Mô hình nên chọn trong điều kiện hiện tại: **{best['id']} - `{best['model']}`**.

Lý do:

- Có `best_macro_f1` cao nhất trong các EX full chạy cùng split.
- Dùng cùng train/validation proxy split với các mô hình còn lại nên so sánh công bằng hơn các run subset trước đó.
- Không phụ thuộc metadata hiện trường, phù hợp dataset hiện tại.

## Per-class report của mô hình được chọn

{per_class_table(best_summary)}

## Các lỗi/giới hạn vẫn còn

1. Dataset không có `field_id`, `capture_session_id` thật, `province`, `season`.
   - Không thể claim tổng quát hóa theo ruộng/tỉnh/mùa vụ.
   - Proxy `variety + age` chỉ là split tạm để giảm rò rỉ.

2. Validation proxy bị lệch lớp.
   - `brown_spot` và `downy_mildew` có support validation rất thấp trong proxy split.
   - Macro F1 vì vậy nhạy với vài ảnh sai/đúng ở các lớp ít mẫu.

3. Các model vẫn chủ yếu là frozen-backbone transfer learning.
   - Đây là benchmark chọn hướng, chưa phải mô hình production cuối cùng.
   - Nếu muốn đẩy hiệu năng, bước tiếp theo nên fine-tune từng phần backbone của mô hình được chọn.

4. `Data preprocessing.docx` chỉ dùng được phần image-only.
   - Dùng được: resize, normalization, train-only augmentation, label policy, duplicate policy.
   - Không dùng được đúng nghĩa: metadata fusion, severity/risk modeling, province/season holdout.

## Đề xuất cải thiện tiếp theo

1. Fine-tune mô hình được chọn:
   - Unfreeze 1-2 block cuối.
   - Giảm learning rate xuống `1e-4` hoặc `5e-5`.
   - Dùng early stopping theo macro F1.

2. Tạo error gallery:
   - Lấy các ảnh validation sai của mô hình được chọn.
   - Gom theo lớp thật/lớp dự đoán.
   - Dùng để kiểm tra nhãn sai, ảnh mơ hồ, hoặc lớp dễ nhầm.

3. Tạo split chặt hơn nếu vẫn không có metadata:
   - Dùng `variety + age + dhash_group`.
   - Tránh near-duplicate crossing tốt hơn split hiện tại.

4. Nếu muốn đúng hướng Central Vietnam:
   - Bắt buộc bổ sung `field_id`, `capture_session_id`, `province`, `season`.
   - Nếu không bổ sung, chỉ nên gọi đây là Kaggle image-only baseline.

## Artifact

- Root: `reports/suggested_runs/full_final_experiments/`
- Report này: `reports/suggested_runs/final_model_selection_report_vi.md`
- Checkpoint mô hình được chọn nằm trong thư mục run tương ứng dưới root trên.
"""
    (BASE / "final_model_selection_report_vi.md").write_text(text, encoding="utf-8")
    with (BASE / "report.md").open("a", encoding="utf-8") as f:
        f.write("\n\n---\n\n")
        f.write(text)
    with (BASE / "report_vi.md").open("a", encoding="utf-8") as f:
        f.write("\n\n---\n\n")
        f.write(text)


def main() -> None:
    FULL_ROOT.mkdir(parents=True, exist_ok=True)
    summaries = []
    for cfg in RUNS:
        out_dir = FULL_ROOT / cfg.experiment_id.lower()
        out_dir.mkdir(parents=True, exist_ok=True)
        if summary_path(out_dir, cfg.experiment_id).exists():
            print(f"Skipping existing {cfg.experiment_id}")
        else:
            print(f"Running {cfg.experiment_id}: {cfg.architecture}")
            run_torch_training(MANIFEST, out_dir, cfg)
        summaries.append(read_summary(out_dir, cfg.experiment_id))
    write_final_report(summaries)


if __name__ == "__main__":
    main()
