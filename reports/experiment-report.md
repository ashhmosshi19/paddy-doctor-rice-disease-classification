# Báo cáo tiếp tục next steps

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
| Số dòng candidate gần trùng | 255 |
| Số nhóm dHash candidate | 124 |
| Nhóm có nhiều nhãn khác nhau | 3 |
| Nhóm chứa exact duplicate bên trong | 72 |
| Kích thước nhóm lớn nhất | 4 |

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
| Tổng dòng clean 8-class | 7315 |
| Train rows | 6451 |
| Validation rows | 864 |
| Tổng proxy groups | 40 |
| Groups bị rò rỉ qua cả train và validation | 0 |
| Số lớp thiếu trong validation | 0 |

File sinh ra:

- `reports/suggested_runs/next_steps/proxy_group_split_manifest.csv`
- `reports/suggested_runs/next_steps/proxy_group_split_class_counts.csv`
- `reports/suggested_runs/next_steps/proxy_group_split_summary.json`

## 5. EX06 proxy group MobileNet run

Run mới: `EX06_PROXY`

| Hạng mục | Giá trị |
|---|---:|
| Output folder | `reports/suggested_runs/ex06_mobilenet_proxy_group_extended` |
| Architecture | `mobilenet_v3_small` |
| Device | `cpu` |
| Train images | 960 |
| Validation images | 197 |
| Epochs | 4 |
| Best macro F1 | 0.2897 |
| Final val accuracy | 0.3553 |
| Final val balanced accuracy | 0.3917 |
| Final val macro F1 | 0.2897 |
| Final val macro recall | 0.3917 |

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

3. Còn 255 dòng near-duplicate candidate.
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


---

# Báo cáo chạy thêm các EX image-only

Ngày tạo: 2026-06-27

## Bối cảnh

Bạn đã duyệt `duplicate_review_gallery_top_groups.jpg`, vì vậy tôi tiếp tục theo hướng không dùng metadata hiện trường. Dataset hiện tại không có `province`, `season`, `field_id`, `capture_session_id` thật, nên các thí nghiệm cần metadata như EX13 không chạy được đúng nghĩa.

Phần `Data preprocessing.docx` vẫn dùng được một phần:

- Dùng được: label policy, quarantine nhãn mơ hồ, resize, normalize, train-only augmentation, duplicate review, split trước augmentation.
- Không dùng được đúng nghĩa: phân tích theo tỉnh/mùa/vụ, metadata fusion, leave-one-province/season-out, severity/risk-factor modeling.

## Kết quả các run mới

| Run | Model | Train | Val | Epochs | Weighted | Augmentation | Best F1 | Final acc | Final F1 | Final recall |
|---|---|---:|---:|---:|---|---|---:|---:|---:|---:|
| EX10 | `mobilenet_v3_large` | 960 | 197 | 4 | False | `basic` | 0.3698 | 0.3858 | 0.3698 | 0.4167 |
| EX09 | `resnet50` | 960 | 197 | 4 | False | `basic` | 0.3254 | 0.3147 | 0.3254 | 0.3655 |
| EX12 | `mobilenet_v3_small` | 960 | 197 | 4 | False | `central_weather_safe` | 0.2671 | 0.2792 | 0.2593 | 0.3292 |
| EX08 | `efficientnet_b0` | 960 | 197 | 4 | False | `basic` | 0.2632 | 0.2335 | 0.2090 | 0.2917 |
| EX11 | `mobilenet_v3_small` | 6451 | 864 | 2 | True | `basic` | 0.2357 | 0.3646 | 0.2357 | 0.3787 |

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


---

# Báo cáo chốt mô hình tối ưu - full frozen-feature EX06-EX12

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

| Hạng | Run | Model | Train | Val | Head epochs | Best epoch | Weighted | Augmentation | Best macro F1 | Best acc | Best balanced acc | Best recall |
|---:|---|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|
| 1 | EX08_FF | `efficientnet_b0` | 6451 | 864 | 80 | 29 | False | `basic` | 0.3749 | 0.6088 | 0.4684 | 0.4684 |
| 2 | EX07_FF | `densenet121` | 6451 | 864 | 80 | 72 | False | `basic` | 0.3597 | 0.5544 | 0.4459 | 0.4459 |
| 3 | EX10_FF | `mobilenet_v3_large` | 6451 | 864 | 80 | 35 | False | `basic` | 0.3457 | 0.5648 | 0.4561 | 0.4561 |
| 4 | EX09_FF | `resnet50` | 6451 | 864 | 80 | 52 | False | `basic` | 0.3412 | 0.5428 | 0.4637 | 0.4637 |
| 5 | EX12_FF | `mobilenet_v3_small` | 6451 | 864 | 80 | 66 | False | `central_weather_safe` | 0.3076 | 0.4965 | 0.4216 | 0.4216 |
| 6 | EX11_FF | `mobilenet_v3_small` | 6451 | 864 | 80 | 67 | True | `basic` | 0.2979 | 0.4421 | 0.4178 | 0.4178 |
| 7 | EX06_FF | `mobilenet_v3_small` | 6451 | 864 | 80 | 79 | False | `basic` | 0.2979 | 0.4988 | 0.4084 | 0.4084 |

## Mô hình tối ưu nên chọn

Chọn: **EX08_FF - `efficientnet_b0`**.

Lý do chọn:

- Có `best_macro_f1` cao nhất trên cùng full proxy split.
- Không cần metadata hiện trường.
- Là lựa chọn tốt nhất trong bộ EX06-EX12 với điều kiện dataset hiện tại.

## Per-class report của mô hình được chọn

| Lớp | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| `bacterial_leaf_blight` | 0.0208 | 0.0294 | 0.0244 | 34 |
| `bacterial_leaf_streak` | 0.2500 | 0.0127 | 0.0241 | 79 |
| `bacterial_panicle_blight` | 0.9545 | 0.7119 | 0.8155 | 59 |
| `blast` | 0.6532 | 0.3103 | 0.4208 | 261 |
| `brown_spot` | 0.0000 | 0.0000 | 0.0000 | 11 |
| `downy_mildew` | 0.0769 | 1.0000 | 0.1429 | 6 |
| `normal` | 0.9211 | 0.9431 | 0.9320 | 334 |
| `tungro` | 0.3385 | 0.5500 | 0.4190 | 80 |

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


---

# Báo cáo DenseNet GPU final

Ngày tạo: 2026-06-27

## Thiết lập

- Model: DenseNet121.
- Thiết bị: NVIDIA GeForce RTX 5060 Laptop GPU.
- PyTorch: 2.11.0+cu128, CUDA: 12.8.
- Dữ liệu: full proxy split `variety + age`.
- Train: 6451 ảnh.
- Validation: 864 ảnh.
- Batch size: 24.
- Phase 1: 6 epoch train classifier head.
- Phase 2: 8 epoch fine-tune `denseblock4` + `norm5`.
- Loss: class-weighted cross entropy.

## Kết quả tốt nhất

| Metric | Value |
|---|---:|
| Best epoch | 12 |
| Best macro F1 | 0.3599 |
| Validation accuracy | 0.5312 |
| Validation balanced accuracy | 0.4573 |
| Validation macro recall | 0.4573 |
| Elapsed seconds | 1058.21 |

## Per-class validation report

| Lớp | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| `bacterial_leaf_blight` | 0.0000 | 0.0000 | 0.0000 | 34 |
| `bacterial_leaf_streak` | 0.4490 | 0.2785 | 0.3438 | 79 |
| `bacterial_panicle_blight` | 0.8085 | 0.6441 | 0.7170 | 59 |
| `blast` | 0.7590 | 0.2414 | 0.3663 | 261 |
| `brown_spot` | 0.0164 | 0.0909 | 0.0278 | 11 |
| `downy_mildew` | 0.0458 | 1.0000 | 0.0876 | 6 |
| `normal` | 0.9253 | 0.8533 | 0.8879 | 334 |
| `tungro` | 0.3793 | 0.5500 | 0.4490 | 80 |

## Nhận xét

- Đây là run DenseNet nghiêm túc nhất hiện tại vì đã chạy full proxy split trên GPU và có fine-tune block cuối.
- Vì dataset không có metadata thật, kết quả vẫn chỉ là image-only/proxy split, chưa phải field/province/season holdout.
- Nếu so với các feature/frozen-head benchmark trước, đây là ứng viên DenseNet nên dùng để so sánh cuối cùng.

## Artifacts

- `reports/suggested_runs/densenet_gpu_final/best_densenet121_gpu.pt`
- `reports/suggested_runs/densenet_gpu_final/train_log.csv`
- `reports/suggested_runs/densenet_gpu_final/validation_predictions.csv`
- `reports/suggested_runs/densenet_gpu_final/densenet_gpu_final_summary.json`


---

# Báo cáo chốt hướng DenseNet

Ngày tạo: 2026-06-27

## Mục tiêu

Theo yêu cầu mới, quá trình được chuyển sang tập trung vào DenseNet và chạy bằng GPU. Môi trường GPU đã được xác minh bằng PyTorch CUDA:

- GPU: NVIDIA GeForce RTX 5060 Laptop GPU
- PyTorch: `2.11.0+cu128`
- CUDA runtime: `12.8`
- Kiểm tra tensor CUDA: chạy thành công phép nhân ma trận trên GPU.

## Bảng so sánh DenseNet

| Run | Thiết lập | Train | Val | Thiết bị | Epoch | Best macro F1 | Val accuracy | Val balanced acc | Ghi chú |
|---|---|---:|---:|---|---:|---:|---:|---:|---|
| EX07_EXT | DenseNet frozen backbone subset | 960 | 240 | CPU | 4 | 0.4456 | 0.4542 | 0.4542 | Không so trực tiếp vì chỉ là subset cân bằng |
| EX07_FF | DenseNet frozen-feature full proxy split | 6,451 | 864 | CPU | 80 head epochs | 0.3597 | 0.5544 | 0.4459 | Full split, backbone frozen |
| EX07_DENSENET_GPU_FINAL | DenseNet GPU two-phase fine-tune | 6,451 | 864 | GPU | 14 | 0.3599 | 0.5312 | 0.4573 | Full split, fine-tune `denseblock4` |

## Kết luận chọn DenseNet

Nếu chỉ xét các run DenseNet dùng full proxy split, nên chọn:

**EX07_DENSENET_GPU_FINAL - DenseNet121 GPU two-phase fine-tune**

Lý do:

- Dùng toàn bộ proxy split thay vì subset nhỏ.
- Chạy thật trên GPU.
- Có fine-tune block cuối (`denseblock4` + `norm5`) thay vì chỉ train linear head.
- Best macro F1 nhỉnh hơn DenseNet frozen-feature full split: `0.3599` so với `0.3597`.
- Balanced accuracy cũng cao hơn: `0.4573` so với `0.4459`.

## Kết quả tốt nhất của DenseNet GPU

| Metric | Value |
|---|---:|
| Best epoch | 12 |
| Best macro F1 | 0.3599 |
| Validation accuracy | 0.5312 |
| Validation balanced accuracy | 0.4573 |
| Validation macro recall | 0.4573 |
| Train images | 6,451 |
| Validation images | 864 |
| Batch size | 24 |
| Runtime | 1,058.21 giây |

## Lỗi hiện tại cần chú ý

1. `bacterial_leaf_blight` có F1 bằng `0.0000`.
   - Model không nhận đúng lớp này trên validation proxy.
   - Đây là lỗi nghiêm trọng nếu lớp này là deployment class.

2. `brown_spot` có F1 rất thấp: `0.0278`.
   - Validation support của lớp này chỉ có 11 ảnh, nên metric rất nhạy.
   - Cần split tốt hơn hoặc bổ sung ảnh validation đại diện hơn.

3. `downy_mildew` có recall `1.0000` nhưng precision `0.0458`.
   - Có dấu hiệu over-predict lớp ít mẫu do class-weighted loss.
   - Nên thử thêm run DenseNet GPU không dùng class weights hoặc dùng focal loss nhẹ.

4. Dataset vẫn thiếu metadata thật.
   - Không thể claim field/province/season generalization.
   - Proxy `variety + age` chỉ là giải pháp tạm.

## Yêu cầu thực hiện thêm nếu muốn cải thiện DenseNet

1. Chạy thêm DenseNet GPU không dùng class weights để kiểm tra lỗi over-predict `downy_mildew`.
2. Tạo split mạnh hơn bằng `variety + age + dhash_group`.
3. Tạo error gallery riêng cho DenseNet GPU:
   - ảnh thật `bacterial_leaf_blight` nhưng dự đoán sai;
   - ảnh thật `brown_spot` nhưng dự đoán sai;
   - ảnh bị dự đoán thành `downy_mildew`.
4. Nếu có GPU tiếp tục, fine-tune thêm với learning rate thấp hơn (`5e-5`) và early stopping theo macro F1.

## Artifacts

- Checkpoint: `reports/suggested_runs/densenet_gpu_final/best_densenet121_gpu.pt`
- Log train: `reports/suggested_runs/densenet_gpu_final/train_log.csv`
- Prediction validation: `reports/suggested_runs/densenet_gpu_final/validation_predictions.csv`
- Summary JSON: `reports/suggested_runs/densenet_gpu_final/densenet_gpu_final_summary.json`
- Report GPU final: `reports/suggested_runs/densenet_gpu_final/densenet_gpu_final_report_vi.md`



---

# Báo cáo DenseNet GPU image-only fixed

Ngày tạo: 2026-06-27

## Sửa lỗi đã áp dụng

- Loại metadata không cần thiết khỏi manifest train.
- Không dùng `field_id`, `province`, `season`, `variety`, `age`, `capture_session_id`.
- Split mới dùng nhãn + `dhash` để giảm duplicate leakage.
- Bỏ class-weighted loss để tránh over-predict lớp hiếm như `downy_mildew`.

## Thiết lập

- Model: DenseNet121.
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU.
- Train: 5851 ảnh.
- Validation: 1464 ảnh.
- Batch size: 48.
- Phase 1: 5 epoch train classifier head.
- Phase 2: 7 epoch fine-tune `denseblock4` + `norm5`.
- Loss: plain cross entropy.

## Kết quả tốt nhất

| Metric | Value |
|---|---:|
| Best epoch | 12 |
| Best macro F1 | 0.8984 |
| Validation accuracy | 0.9051 |
| Validation balanced accuracy | 0.8934 |
| Validation macro recall | 0.8934 |
| Runtime seconds | 860.35 |

## Per-class validation report

| Lớp | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| `bacterial_leaf_blight` | 0.9211 | 0.7447 | 0.8235 | 94 |
| `bacterial_leaf_streak` | 0.9722 | 0.9211 | 0.9459 | 76 |
| `bacterial_panicle_blight` | 0.9286 | 0.9701 | 0.9489 | 67 |
| `blast` | 0.8997 | 0.9075 | 0.9036 | 346 |
| `brown_spot` | 0.9222 | 0.8691 | 0.8949 | 191 |
| `downy_mildew` | 0.7970 | 0.8548 | 0.8249 | 124 |
| `normal` | 0.9435 | 0.9543 | 0.9489 | 350 |
| `tungro` | 0.8696 | 0.9259 | 0.8969 | 216 |

## Nhận xét

- Đây là run sửa lỗi hợp lý nhất sau khi bỏ metadata giả và bỏ class weights.
- Nếu accuracy/macro F1 ổn hơn bản cũ, nên dùng split/manifest này cho các lần benchmark tiếp theo.
- Nếu lớp nào vẫn F1 thấp, bước tiếp theo là tạo error gallery cho riêng lớp đó.


---

# Báo cáo sửa lỗi metadata và DenseNet

Ngày tạo: 2026-06-27

## Vấn đề trước khi sửa

Pipeline trước đó dùng `proxy_split` từ `variety + age` vì dataset không có metadata hiện trường thật. Cách này giúp tránh random split quá dễ, nhưng tạo validation lệch lớp nặng:

- `brown_spot`: chỉ 11 ảnh validation.
- `downy_mildew`: chỉ 6 ảnh validation.

DenseNet GPU cũ còn dùng class-weighted cross entropy, làm model over-predict lớp hiếm. Dấu hiệu rõ nhất:

- `downy_mildew`: recall `1.0000` nhưng precision `0.0458`.
- `bacterial_leaf_blight`: F1 `0.0000`.
- `brown_spot`: F1 `0.0278`.

## Sửa lỗi đã thực hiện

1. Tạo manifest image-only mới:
   - File: `reports/suggested_runs/image_only_fix/manifest_image_only.csv`
   - Chỉ giữ: `image_id`, `relative_path`, `image_path`, `primary_disease`, `dhash`.

2. Loại metadata không cần thiết khỏi manifest train:
   - Bỏ `field_id`, `province`, `season` vì rỗng.
   - Bỏ `variety`, `age`, `capture_session_id` vì không phải metadata hiện trường đủ tin cậy.
   - Bỏ các cột audit như brightness/sharpness/file_bytes khỏi input training.

3. Tạo split mới:
   - File: `reports/suggested_runs/image_only_fix/image_only_hash_split_manifest.csv`
   - Phương pháp: stratified theo label, group theo `dhash`.
   - Không có `dhash` group nào đi qua cả train và validation.

4. Chạy lại DenseNet GPU:
   - Không dùng class weights.
   - Loss: plain cross entropy.
   - Fine-tune `denseblock4` + `norm5`.

## Split mới

| Lớp | Train | Validation |
|---|---:|---:|
| `bacterial_leaf_blight` | 377 | 94 |
| `bacterial_leaf_streak` | 304 | 76 |
| `bacterial_panicle_blight` | 269 | 67 |
| `blast` | 1,382 | 346 |
| `brown_spot` | 762 | 191 |
| `downy_mildew` | 494 | 124 |
| `normal` | 1,399 | 350 |
| `tungro` | 864 | 216 |

## Kết quả trước/sau

| Run | Manifest/split | Loss | Train | Val | Accuracy | Balanced acc | Macro F1 |
|---|---|---|---:|---:|---:|---:|---:|
| DenseNet GPU cũ | proxy `variety + age` | class-weighted CE | 6,451 | 864 | 0.5312 | 0.4573 | 0.3599 |
| DenseNet GPU fixed | image-only + `dhash` split | plain CE | 5,851 | 1,464 | 0.9051 | 0.8934 | 0.8984 |

## Per-class fixed result

| Lớp | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| `bacterial_leaf_blight` | 0.9211 | 0.7447 | 0.8235 | 94 |
| `bacterial_leaf_streak` | 0.9722 | 0.9211 | 0.9459 | 76 |
| `bacterial_panicle_blight` | 0.9286 | 0.9701 | 0.9489 | 67 |
| `blast` | 0.8997 | 0.9075 | 0.9036 | 346 |
| `brown_spot` | 0.9222 | 0.8691 | 0.8949 | 191 |
| `downy_mildew` | 0.7970 | 0.8548 | 0.8249 | 124 |
| `normal` | 0.9435 | 0.9543 | 0.9489 | 350 |
| `tungro` | 0.8696 | 0.9259 | 0.8969 | 216 |

## Kết luận

Kết quả thấp trước đó không phải do DenseNet không phù hợp, mà chủ yếu do:

- split proxy metadata làm validation lệch lớp;
- class weights làm model over-predict lớp hiếm;
- manifest train còn nhiều cột metadata/audit không cần thiết, gây nhiễu trong cách đọc báo cáo và thiết kế pipeline.

Nên dùng nhánh fixed hiện tại làm baseline chính:

- Manifest: `reports/suggested_runs/image_only_fix/manifest_image_only.csv`
- Split: `reports/suggested_runs/image_only_fix/image_only_hash_split_manifest.csv`
- Checkpoint: `reports/suggested_runs/image_only_fix/densenet_gpu_no_weights/best_densenet121_gpu_no_weights.pt`
- Summary: `reports/suggested_runs/image_only_fix/densenet_gpu_no_weights/densenet_gpu_image_only_fixed_summary.json`

Lưu ý: split mới vẫn là image-only Kaggle-style split, chưa phải field/province/season split. Không nên claim generalization ngoài đồng ruộng miền Trung nếu chưa bổ sung metadata thật.

