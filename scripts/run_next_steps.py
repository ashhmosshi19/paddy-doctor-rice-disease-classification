from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_suggested_p0 import CANONICAL_8, RunConfig, run_torch_training


BASE = Path("reports/suggested_runs")
AUDIT_DIR = BASE / "ex05_data_audit"
NEXT_DIR = BASE / "next_steps"
NEXT_DIR.mkdir(parents=True, exist_ok=True)


def write_label_map() -> None:
    label_map = """version: v1_clean_8class
status: draft_needs_agronomist_review
target_task: single_label_rice_disease_classification
canonical_labels:
  - id: 0
    name: bacterial_leaf_blight
    vietnamese_display_name: bạc lá
    status: accepted_for_baseline
  - id: 1
    name: bacterial_leaf_streak
    vietnamese_display_name: sọc lá vi khuẩn
    status: accepted_for_baseline
  - id: 2
    name: bacterial_panicle_blight
    vietnamese_display_name: bạc lá/bạc bông vi khuẩn
    status: accepted_for_baseline
  - id: 3
    name: blast
    vietnamese_display_name: đạo ôn
    status: accepted_for_baseline
  - id: 4
    name: brown_spot
    vietnamese_display_name: đốm nâu
    status: accepted_for_baseline
  - id: 5
    name: downy_mildew
    vietnamese_display_name: sương mai
    status: accepted_for_baseline
  - id: 6
    name: normal
    vietnamese_display_name: khỏe mạnh/bình thường
    status: accepted_for_baseline
  - id: 7
    name: tungro
    vietnamese_display_name: tungro
    status: accepted_for_baseline
quarantined_labels:
  - name: hispa
    reason: pest_label_not_clean_disease_class
    required_action: agronomist_review_before_model_inclusion
  - name: dead_heart
    reason: pest_or_symptom_label_not_clean_disease_class
    required_action: agronomist_review_before_model_inclusion
unknown_policy:
  other_unknown: keep_out_of_first_supervised_baseline
  needs_expert_review: keep_out_of_first_supervised_baseline
split_policy:
  preferred_group_key: field_id_or_capture_session_id
  current_proxy_group_key: capture_session_id_from_variety_age
metadata_limitations:
  - province_missing
  - season_missing
  - real_field_id_missing
  - real_capture_session_id_missing
"""
    Path("label_map.yaml").write_text(label_map, encoding="utf-8")


def summarize_duplicate_candidates() -> dict:
    dupes = pd.read_csv(AUDIT_DIR / "duplicate_perceptual_dhash_candidates.csv")
    if dupes.empty:
        summary = {"groups": 0, "rows": 0}
        (NEXT_DIR / "duplicate_review_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return summary

    group_rows = []
    for dhash, group in dupes.groupby("dhash"):
        labels = sorted(group["primary_disease"].dropna().unique().tolist())
        sha1_count = int(group["sha1"].nunique())
        group_rows.append(
            {
                "dhash": dhash,
                "rows": int(len(group)),
                "unique_sha1": sha1_count,
                "labels": "|".join(labels),
                "cross_label": len(labels) > 1,
                "exact_duplicate_inside_group": sha1_count < len(group),
                "example_paths": " | ".join(group["relative_path"].head(6).tolist()),
            }
        )
    group_summary = pd.DataFrame(group_rows).sort_values(
        ["cross_label", "rows"], ascending=[False, False]
    )
    group_summary.to_csv(NEXT_DIR / "duplicate_perceptual_group_summary.csv", index=False)

    summary = {
        "candidate_rows": int(len(dupes)),
        "candidate_groups": int(group_summary["dhash"].nunique()),
        "cross_label_groups": int(group_summary["cross_label"].sum()),
        "groups_with_exact_duplicates": int(group_summary["exact_duplicate_inside_group"].sum()),
        "largest_group_size": int(group_summary["rows"].max()),
    }
    (NEXT_DIR / "duplicate_review_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    make_duplicate_gallery(dupes, group_summary)
    return summary


def make_duplicate_gallery(dupes: pd.DataFrame, group_summary: pd.DataFrame) -> None:
    top_groups = group_summary.head(12)["dhash"].tolist()
    thumbs = []
    for dhash in top_groups:
        group = dupes.loc[dupes["dhash"] == dhash].head(4)
        for _, row in group.iterrows():
            try:
                with Image.open(row["image_path"]) as im:
                    im = im.convert("RGB")
                    im.thumbnail((160, 160))
                    canvas = Image.new("RGB", (180, 220), "white")
                    canvas.paste(im, ((180 - im.width) // 2, 5))
                    draw = ImageDraw.Draw(canvas)
                    text = f"{row['primary_disease']}\n{row['image_id']}\n{dhash[:8]}"
                    draw.multiline_text((8, 170), text, fill=(0, 0, 0), spacing=2)
                    thumbs.append(canvas)
            except Exception:
                continue
    if not thumbs:
        return
    cols = 4
    rows = int(np.ceil(len(thumbs) / cols))
    sheet = Image.new("RGB", (cols * 180, rows * 220), "white")
    for idx, thumb in enumerate(thumbs):
        x = (idx % cols) * 180
        y = (idx // cols) * 220
        sheet.paste(thumb, (x, y))
    sheet.save(NEXT_DIR / "duplicate_review_gallery_top_groups.jpg", quality=90)


def create_proxy_group_split() -> dict:
    clean = pd.read_csv(AUDIT_DIR / "manifest_clean_8class_no_exact_dupes.csv")
    clean["capture_session_id"] = clean["capture_session_id"].fillna(
        clean["variety"].astype(str) + "_age_" + clean["age"].astype(str)
    )
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=2026)
    train_idx, val_idx = next(
        splitter.split(clean, y=clean["primary_disease"], groups=clean["capture_session_id"])
    )
    split = clean.copy()
    split["proxy_split"] = "train"
    split.iloc[val_idx, split.columns.get_loc("proxy_split")] = "validation"
    split.to_csv(NEXT_DIR / "proxy_group_split_manifest.csv", index=False)

    leakage = split.groupby("capture_session_id")["proxy_split"].nunique()
    crosstab = pd.crosstab(split["primary_disease"], split["proxy_split"])
    crosstab.to_csv(NEXT_DIR / "proxy_group_split_class_counts.csv")
    summary = {
        "rows": int(len(split)),
        "train_rows": int((split["proxy_split"] == "train").sum()),
        "validation_rows": int((split["proxy_split"] == "validation").sum()),
        "proxy_group_key": "capture_session_id = variety + age",
        "groups_total": int(split["capture_session_id"].nunique()),
        "groups_crossing_splits": int((leakage > 1).sum()),
        "validation_classes_missing": int((crosstab.get("validation", pd.Series(dtype=int)) == 0).sum()),
        "class_counts": {
            label: {k: int(v) for k, v in row.dropna().to_dict().items()}
            for label, row in crosstab.iterrows()
        },
    }
    (NEXT_DIR / "proxy_group_split_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def make_proxy_subset(split_manifest: Path, out_csv: Path, train_per_class=120, val_per_class=30) -> None:
    frame = pd.read_csv(split_manifest)
    rng = random.Random(2027)
    parts = []
    for label in CANONICAL_8:
        train_g = frame[(frame["primary_disease"] == label) & (frame["proxy_split"] == "train")]
        val_g = frame[(frame["primary_disease"] == label) & (frame["proxy_split"] == "validation")]
        train_rows = train_g.sample(
            n=min(train_per_class, len(train_g)), random_state=rng.randint(1, 999999)
        )
        val_rows = val_g.sample(
            n=min(val_per_class, len(val_g)), random_state=rng.randint(1, 999999)
        )
        parts.extend([train_rows, val_rows])
    subset = pd.concat(parts, ignore_index=True)
    subset.to_csv(out_csv, index=False)


def train_proxy_mobilenet() -> None:
    out_dir = BASE / "ex06_mobilenet_proxy_group_extended"
    out_dir.mkdir(parents=True, exist_ok=True)
    subset_csv = NEXT_DIR / "proxy_group_training_subset.csv"
    make_proxy_subset(NEXT_DIR / "proxy_group_split_manifest.csv", subset_csv)
    run_torch_training(
        subset_csv,
        out_dir,
        RunConfig(
            experiment_id="EX06_PROXY",
            architecture="mobilenet_v3_small",
            train_per_class=120,
            val_per_class=30,
            epochs=4,
            batch_size=32,
            seed=2028,
            lr=1e-3,
        ),
    )


def write_report(label_summary: dict, duplicate_summary: dict, split_summary: dict) -> None:
    ex06_proxy = json.loads(
        (BASE / "ex06_mobilenet_proxy_group_extended" / "ex06_proxy_summary.json").read_text(
            encoding="utf-8"
        )
    )
    report = f"""# Báo cáo tiếp tục next steps

Ngày tạo: 2026-06-27

## 1. Việc đã chạy thêm

Tôi đã tiếp tục các bước kế tiếp từ `reports/suggested_runs/report.md`:

1. Tạo `label_map.yaml` có dấu, version `v1_clean_8class`, kèm chính sách quarantine cho `hispa` và `dead_heart`.
2. Phân tích lại nhóm ảnh gần trùng trong `duplicate_perceptual_dhash_candidates.csv`.
3. Tạo split tạm thời theo proxy group `capture_session_id = variety + age`.
4. Chạy thêm một MobileNetV3 trên subset lấy từ proxy group split để kiểm tra mức độ nhạy với split không rò rỉ group proxy.

## 2. Kết quả label map

File mới: `label_map.yaml`

Trạng thái: `draft_needs_agronomist_review`.

8 lớp được dùng cho baseline:

- `bacterial_leaf_blight` - bạc lá
- `bacterial_leaf_streak` - sọc lá vi khuẩn
- `bacterial_panicle_blight` - bạc lá/bạc bông vi khuẩn
- `blast` - đạo ôn
- `brown_spot` - đốm nâu
- `downy_mildew` - sương mai
- `normal` - khỏe mạnh/bình thường
- `tungro` - tungro

2 nhãn đang bị quarantine:

- `hispa`: nhãn sâu hại, không phải lớp bệnh sạch cho baseline hiện tại.
- `dead_heart`: nhãn sâu hại/triệu chứng, cần chuyên gia nông học xác nhận trước khi đưa vào taxonomy.

## 3. Kết quả duplicate review

| Hạng mục | Giá trị |
|---|---:|
| Số dòng candidate gần trùng | {duplicate_summary["candidate_rows"]} |
| Số nhóm dHash candidate | {duplicate_summary["candidate_groups"]} |
| Nhóm có nhiều nhãn khác nhau | {duplicate_summary["cross_label_groups"]} |
| Nhóm chứa exact duplicate bên trong | {duplicate_summary["groups_with_exact_duplicates"]} |
| Kích thước nhóm lớn nhất | {duplicate_summary["largest_group_size"]} |

File sinh ra:

- `reports/suggested_runs/next_steps/duplicate_perceptual_group_summary.csv`
- `reports/suggested_runs/next_steps/duplicate_review_summary.json`
- `reports/suggested_runs/next_steps/duplicate_review_gallery_top_groups.jpg`

Lưu ý: đây mới là review bằng hash/metadata. Các nhóm gần trùng vẫn cần người xem ảnh thật, đặc biệt các nhóm có nhiều nhãn khác nhau.

## 4. Proxy group split

Vì dữ liệu hiện tại không có `field_id`, `province`, `season` hoặc `capture_session_id` thật, tôi tạo split tạm thời bằng:

`capture_session_id = variety + age`

Kết quả:

| Hạng mục | Giá trị |
|---|---:|
| Tổng dòng clean 8-class | {split_summary["rows"]} |
| Train rows | {split_summary["train_rows"]} |
| Validation rows | {split_summary["validation_rows"]} |
| Tổng proxy groups | {split_summary["groups_total"]} |
| Groups bị rò rỉ qua cả train và validation | {split_summary["groups_crossing_splits"]} |
| Số lớp thiếu trong validation | {split_summary["validation_classes_missing"]} |

File sinh ra:

- `reports/suggested_runs/next_steps/proxy_group_split_manifest.csv`
- `reports/suggested_runs/next_steps/proxy_group_split_class_counts.csv`
- `reports/suggested_runs/next_steps/proxy_group_split_summary.json`

## 5. EX06 proxy group MobileNet run

Run mới: `EX06_PROXY`

| Hạng mục | Giá trị |
|---|---:|
| Output folder | `reports/suggested_runs/ex06_mobilenet_proxy_group_extended` |
| Architecture | `{ex06_proxy["architecture"]}` |
| Device | `{ex06_proxy["device"]}` |
| Train images | {ex06_proxy["train_images"]} |
| Validation images | {ex06_proxy["val_images"]} |
| Epochs | {ex06_proxy["epochs"]} |
| Best macro F1 | {ex06_proxy["best_macro_f1"]:.4f} |
| Final val accuracy | {ex06_proxy["final_metrics"]["val_accuracy"]:.4f} |
| Final val balanced accuracy | {ex06_proxy["final_metrics"]["val_balanced_accuracy"]:.4f} |
| Final val macro F1 | {ex06_proxy["final_metrics"]["val_macro_f1"]:.4f} |
| Final val macro recall | {ex06_proxy["final_metrics"]["val_macro_recall"]:.4f} |

Artifacts:

- `reports/suggested_runs/ex06_mobilenet_proxy_group_extended/train_log.csv`
- `reports/suggested_runs/ex06_mobilenet_proxy_group_extended/train_split.csv`
- `reports/suggested_runs/ex06_mobilenet_proxy_group_extended/val_split.csv`
- `reports/suggested_runs/ex06_mobilenet_proxy_group_extended/validation_predictions.csv`
- `reports/suggested_runs/ex06_mobilenet_proxy_group_extended/ex06_proxy_summary.json`
- `reports/suggested_runs/ex06_mobilenet_proxy_group_extended/ex06_proxy_mobilenet_v3_small.pt`

## 6. Lỗi và giới hạn hiện tại

### Lỗi/thiếu dữ liệu nghiêm trọng

1. Thiếu `province`, `season`, `field_id`, `capture_session_id` thật.
   - Hệ quả: chưa thể chạy đúng group-aware field holdout hoặc leave-one-province/season-out như guideline.
   - Split proxy `variety + age` chỉ là biện pháp tạm, không chứng minh được khả năng tổng quát hóa ngoài đồng ruộng miền Trung.

2. `hispa` và `dead_heart` chưa thuộc taxonomy bệnh sạch.
   - Hệ quả: nếu đưa thẳng vào model bệnh lúa sẽ làm lệch bài toán.
   - Hiện tại hai nhãn này đã bị loại khỏi baseline 8-class.

3. Còn {duplicate_summary["candidate_rows"]} dòng near-duplicate candidate.
   - Hệ quả: nếu các ảnh gần trùng lọt qua train/validation thì metric có thể bị thổi phồng.
   - Tôi đã sinh gallery và group summary, nhưng cần người review ảnh thật.

4. Model vẫn đang chạy CPU với frozen backbone.
   - Hệ quả: kết quả hiện tại là diagnostic, chưa phải full benchmark.
   - Chưa có fine-tuning nhiều pha, early stopping nghiêm túc, calibration, hoặc error gallery đầy đủ.

### Lỗi/yếu điểm về đánh giá

1. Chưa thể báo cáo đúng EX16 field/province/season robustness.
2. Chưa có test set có nhãn; `test_images.zip` chỉ có ảnh test chưa có label nên chưa tính được metric test thật.
3. Các run trước đó dùng subset nhỏ, không phải toàn bộ 7,315 ảnh sạch.
4. Chưa có review chuyên gia cho label tiếng Việt và các lớp bệnh dễ nhầm.

## 7. Yêu cầu cần thực hiện thêm

Để dự án đạt đúng yêu cầu trong guideline, cần bổ sung hoặc quyết định các điểm sau:

1. Bổ sung metadata tối thiểu cho mỗi ảnh:
   - `field_id`
   - `capture_session_id`
   - `province`
   - `season`
   - `growth_stage`
   - `severity_score`
   - `weather_condition`

2. Nhờ chuyên gia nông học xác nhận:
   - `label_map.yaml`
   - cách xử lý `hispa`
   - cách xử lý `dead_heart`
   - tên tiếng Việt của từng lớp
   - các trường hợp đồng nhiễm hoặc ảnh mơ hồ

3. Review thủ công file/galllery duplicate:
   - `duplicate_review_gallery_top_groups.jpg`
   - `duplicate_perceptual_group_summary.csv`

4. Nếu vẫn chỉ dùng Kaggle dataset:
   - ghi rõ đây là baseline tham chiếu, không phải Central Vietnam field dataset.
   - không claim generalization theo tỉnh/mùa/vụ.

5. Nếu muốn tiếp tục training nghiêm túc:
   - chạy full EX06/EX07 trên toàn bộ manifest sạch.
   - thêm unfreeze/fine-tuning backbone.
   - thêm early stopping.
   - lưu confusion matrix, error gallery, calibration/reject threshold.
   - sau khi có metadata thật, chạy lại group split theo `field_id` hoặc `capture_session_id` thật.

## 8. Kết luận

Tình trạng hiện tại: pipeline đã có audit, label map draft, duplicate review artifacts, proxy group split, và thêm một run MobileNet theo proxy group. Tuy nhiên, lỗi lớn nhất không nằm ở code mà nằm ở dữ liệu: thiếu metadata hiện trường và thiếu xác nhận taxonomy. Vì vậy kết quả model hiện tại chỉ nên xem là baseline kỹ thuật, chưa phải kết quả triển khai cho miền Trung Việt Nam.
"""
    (BASE / "report_vi.md").write_text(report, encoding="utf-8")
    (BASE / "report.md").write_text(report, encoding="utf-8")


def main() -> None:
    write_label_map()
    duplicate_summary = summarize_duplicate_candidates()
    split_summary = create_proxy_group_split()
    train_proxy_mobilenet()
    write_report({}, duplicate_summary, split_summary)


if __name__ == "__main__":
    main()
