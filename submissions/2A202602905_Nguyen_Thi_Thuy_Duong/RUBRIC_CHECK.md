# Đối chiếu rubric trước khi nộp

**MSSV:** 2A202602905  
**Họ tên:** Nguyễn Thị Thùy Dương  
Đối chiếu README mục 2/4/5, GUIDE mục 6 và RUBRIC của chính repository. Đây là kiểm tra bằng chứng, không phải điểm do giảng viên xác nhận.

| Mục | Kết quả kiểm tra | Bằng chứng và giới hạn |
|---|---|---|
| P1 | Đủ sáu sản phẩm | results.xlsx, report.md, curves/, code/, README.md, predictions/ |
| P2 | Có số liệu chạy thật, truy ngược được | runs/ có config/history/summary; curves/ cho từng B/T/F; evidence/ có COMPLETE, log Kaggle, selection, CSV và kết quả tính lại |
| P3 | Đúng fold 0 | train/val/test 10.501/3.501/3.507; giao rỗng, hợp 17.509; nhãn gốc; kiểm tra trong package_verification.json |
| P4 | Đủ dự đoán final và baseline | Mỗi nhóm ba seed test, đủ 3.507 ảnh; kèm val và uncal. Tổng 34 CSV đã kiểm tra định dạng/nhãn |
| A — 12đ | Đủ thiết lập chính; giới hạn bằng chứng augmentation | Chọn bằng val, selection khóa trước test. EDA mẫu/phân bố và Table 1 trong báo cáo. Log uniform loss 2,197224617, loss step 0 = 3,512908, step 80 = 0,000001237, overfit batch accuracy = 1,0. Ảnh augmentation hiển thị trên notebook online, chưa xuất riêng. Seed/config/pretrained tag/version được lưu |
| B — 12đ | Đạt số lượng và nhóm kiến trúc | ResNet50, ResNeXt50, ConvNeXtTiny, DeiT-S, EfficientNet-B0; cùng nền/split/seed. Sheet Backbones đủ cột và latency bổ sung; GMAC transformer từ thop có thể thiếu attention, đã nêu hạn chế |
| C — 16đ | Đạt bốn trục, có kết hợp | A khởi tạo; B augmentation; C loss; F EMA, mỗi trục có nền và biến thể. T01–T09 thay một yếu tố; T10 kết hợp và giảm so với T06. Không khẳng định ablation một seed vượt nhiễu; final/baseline có std ba seed |
| D — 12đ | Có bốn biến thể thực sự ngoài I00 và phép fusion kiểm tra | Hflip probability, hflip logits, temperature, FP16. I04 fusion là no-op trên ConvNeXt LayerNorm và không tính như một cải tiến độc lập. Tất cả có p50/p95/p99, warmup 10, 100 lượt, đồng bộ CUDA; batch 1 và bổ sung batch 32. Temperature fit trên val; latency forward chưa bao quát hệ thống robot |
| E — 8đ | Đủ bảy sheet và cột GUIDE | Backbones/Training/Inference/Final/PerClass/Latency/Summary; mean/std và delta bằng công thức, cache kiểm tra khớp. Summary top 10 seed 0 có chi phí/latency. Chưa mở bằng Microsoft Excel; công thức đã kiểm tra bằng Artifact Tool |
| F — 4đ | Đủ ảnh từng nhãn run B/T/F | 22 nhãn run, 19 lần train riêng biệt do tái sử dụng có provenance. Đồ thị loss train/val, F1 val/LR. Nhận xét chưa hội tụ của scratch và regularization kết hợp trong báo cáo |
| G — 12đ | Đủ dàn ý; phân tích ảnh cặp khó còn giới hạn | Chín mục, bảng và figure, confusion ba seed, per-class mean/std, tám ảnh lỗi. Bổ sung số nhầm Chinee Apple → Snake Weed 4/2/8 và ngược lại 3/3/2. Chưa có ảnh riêng của chính cặp này, nên không tự nhận đạt trọn điểm phân tích lỗi. Một fold/ngẫu nhiên theo ảnh có thể lạc quan; ngân sách và lệch miền đã nêu |
| H — 4đ | Có implementation và hướng dẫn tái lập | Không còn NotImplementedError trong code/ bài nộp; một hàm train dùng chung; eval.py nguyên bản. Log Kaggle ghi 8 core tests pass. Notebook source nhúng code được đối chiếu khớp code đã chạy. Phụ thuộc phụ chưa pin đầy đủ nên không khẳng định tái lập bit-for-bit |
| I — 20đ | Tự chấm bằng eval.py: 19/20 | I1=7; I2=4; I3=4; I4a=1; I4b=1; I5=2. Delta F1 0,003509 vượt std nhưng dưới 0,01 nên I2 không đạt 5/5. I5 dùng F01 test F1 0,971512 và TTA p95 batch 1 = 14,518995 ms Tesla T4, forward-only |
| Thưởng | Không tự nhận điểm thưởng | Không có nhiều fold/DINOv2/distillation/ONNX/Grad-CAM hoặc đánh giá miền mới đã chạy |

Không tự cộng A–H thành điểm chính thức. Các giới hạn về ảnh cặp khó, augmentation, profiler và tái lập có thể ảnh hưởng điểm giảng viên cho. Không chạy lại test hoặc đổi cấu hình sau phân tích.

## Git và dữ liệu

Chỉ push `origin` (https://github.com/ntthduong/K4-Track4-Day2-Deeplearning-Advance), không push `upstream` của VinUni-AI20k. Không tạo PR upstream trong lần cập nhật này.

Không commit dataset ảnh, checkpoint, ZIP xuất Kaggle hay môi trường Python. Repo và ZIP chỉ giữ ảnh minh hoạ đã ghép trong `curves/error_examples.png` cùng metadata lỗi, không kèm tám JPEG nguồn trong error_images/.

`starter/` của đề giữ nguyên pseudo-code: kiểm tra NotImplementedError áp dụng cho **code/ bài nộp**, không áp dụng cho template starter/ của ban tổ chức. Bộ tests/ của đề kiểm tra eval và template, tách biệt với tám core tests implementation đã chạy thật trên Kaggle.

Đã chạy lại `python -X utf8 -m unittest discover -s tests -v` từ root repo: **38 tests pass**. Log nằm trong `evidence/repository_tests.txt`. Thư viện test được cài riêng ở thư mục local outputs/, không đưa vào Git hoặc gói nộp.
