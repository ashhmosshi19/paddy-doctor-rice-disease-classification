from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


BASE = Path("reports/suggested_runs")
AUDIT = BASE / "ex05_data_audit" / "manifest_clean_8class_no_exact_dupes.csv"
OUT = BASE / "image_only_fix"
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

KEEP_COLUMNS = ["image_id", "relative_path", "image_path", "primary_disease", "dhash"]


def read_rows() -> list[dict]:
    with AUDIT.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    clean = []
    for row in rows:
        if row.get("primary_disease") not in CLASSES:
            continue
        if row.get("include_ex06_ex07", "").lower() != "true":
            continue
        clean.append({col: row.get(col, "") for col in KEEP_COLUMNS})
    return clean


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def make_hash_group_split(rows: list[dict], val_fraction: float = 0.2, seed: int = 20260627):
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["dhash"] or row["image_id"]].append(row)

    group_infos = []
    for group_id, group_rows in groups.items():
        counts = Counter(row["primary_disease"] for row in group_rows)
        majority_label = counts.most_common(1)[0][0]
        group_infos.append(
            {
                "group_id": group_id,
                "majority_label": majority_label,
                "rows": group_rows,
                "size": len(group_rows),
                "labels": dict(counts),
            }
        )

    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = defaultdict(list)
    for info in group_infos:
        by_label[info["majority_label"]].append(info)
    for infos in by_label.values():
        rng.shuffle(infos)

    val_group_ids = set()
    class_counts = Counter(row["primary_disease"] for row in rows)
    target_val = {label: max(1, round(class_counts[label] * val_fraction)) for label in CLASSES}
    current_val = Counter()

    for label in CLASSES:
        infos = by_label[label]
        for info in infos:
            if current_val[label] >= target_val[label]:
                continue
            val_group_ids.add(info["group_id"])
            for row in info["rows"]:
                current_val[row["primary_disease"]] += 1

    split_rows = []
    for row in rows:
        out = dict(row)
        out["split"] = "validation" if (row["dhash"] or row["image_id"]) in val_group_ids else "train"
        split_rows.append(out)

    leakage = 0
    for group_id, group_rows in groups.items():
        splits = {
            "validation" if group_id in val_group_ids else "train"
            for _ in group_rows
        }
        if len(splits) > 1:
            leakage += 1

    split_counts = defaultdict(Counter)
    for row in split_rows:
        split_counts[row["primary_disease"]][row["split"]] += 1

    return split_rows, {
        "rows": len(rows),
        "columns_kept": KEEP_COLUMNS,
        "columns_removed_reason": "Removed blank/untrusted metadata and audit-only columns from the training manifest.",
        "split_method": "image_only_label_stratified_dhash_group_split",
        "split_group": "dhash",
        "groups_total": len(groups),
        "groups_crossing_splits": leakage,
        "train_rows": sum(1 for row in split_rows if row["split"] == "train"),
        "validation_rows": sum(1 for row in split_rows if row["split"] == "validation"),
        "class_counts": {label: dict(split_counts[label]) for label in CLASSES},
    }


def main() -> None:
    rows = read_rows()
    write_csv(OUT / "manifest_image_only.csv", rows, KEEP_COLUMNS)
    split_rows, summary = make_hash_group_split(rows)
    write_csv(OUT / "image_only_hash_split_manifest.csv", split_rows, KEEP_COLUMNS + ["split"])
    (OUT / "image_only_fix_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "# Image-only fix",
        "",
        "Đã tạo manifest train không còn các cột metadata không có thật.",
        "",
        "## File sinh ra",
        "",
        "- `manifest_image_only.csv`: chỉ giữ `image_id`, `relative_path`, `image_path`, `primary_disease`, `dhash`.",
        "- `image_only_hash_split_manifest.csv`: thêm cột kỹ thuật `split` để train/validation.",
        "- `image_only_fix_summary.json`: thống kê split.",
        "",
        "## Lý do",
        "",
        "- Bỏ `field_id`, `province`, `season` vì toàn bộ rỗng.",
        "- Bỏ `variety`, `age`, `capture_session_id` khỏi manifest train vì đây không phải metadata hiện trường đáng tin.",
        "- Bỏ các cột audit như brightness/sharpness/file_bytes khỏi training manifest vì chúng không phải input mô hình.",
        "- Dùng `dhash` làm group kỹ thuật để giảm nguy cơ ảnh gần trùng đi qua cả train và validation.",
    ]
    (OUT / "image_only_fix_report_vi.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
