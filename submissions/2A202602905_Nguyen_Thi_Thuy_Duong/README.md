# Bài nộp DeepWeeds – K4 Track4 Day2

**MSSV:** 2A202602905  
**Họ tên:** Nguyễn Thị Thùy Dương  
**Tài khoản Kaggle/GitHub:** ntthduong  
**Thư mục bài nộp:** `2A202602905_Nguyen_Thi_Thuy_Duong`

Nguồn: [Kaggle notebook version 5, ID 355122553](https://www.kaggle.com/code/ntthduong/k4-track4-day2-deepweeds-lab?scriptVersionId=355122553). Version 5 đã hoàn tất 13.218,1 giây; 5 backbone nhập từ version 4. Bản notebook trong code là **source để tái lập**, không phải file có output đã tải từ Kaggle. Các file `.py` nhúng trong source đã đối chiếu khớp với code thật của version 5.

## Kết quả đã chốt

- F01: ConvNeXt-Tiny `in12k_ft_in1k`, fine-tune, label smoothing 0,1, Hflip probability K=2, temperature fit trên val của mỗi seed.
- Seed 0/1/2. Test macro-F1 0,971512 ± 0,001189; top-1 0,977759 ± 0,000754; ECE 0,007030 ± 0,001104.
- Baseline: cùng ConvNeXt-Tiny, CE, 1-view. Macro-F1 0,968003 ± 0,000566; delta 0,003509.
- Fold 0 chính thức: train 10.501, val 3.501, test 3.507; giao tập rỗng. CSV nhãn gốc lấy từ [DeepWeeds của tác giả](https://github.com/AlexOlsen/DeepWeeds/tree/master/labels).
- Số liệu score/grade nguyên bản ở eval_out; phép chấm lại từ gói tại evidence/recomputed. `grade_I.json` cho **19/20 phần I**, không phải tổng điểm toàn rubric.

## Các tệp

`results.xlsx`: 7 sheet đúng GUIDE, số dạng 0–1, delta và mean/std qua seed có công thức. Summary xếp hạng lịch sử seed 0; không tự đổi hạng khi người đọc sửa số.  
`report.md`: thiết lập, bảng kết quả, phân tích ablation/calibration, confusion và tám ảnh lỗi.  
`curves/`: đủ ảnh cho 22 nhãn run B/T/F, EDA, ba confusion và các hình tổng hợp.  
`predictions/`: toàn bộ CSV đã lưu, gồm final/baseline test/val/uncal cho ba seed và vòng validation.  
`runs/`: config, history, summary và provenance; không chứa checkpoint.  
`code/`: code version 5, notebook source và script đo latency bổ sung.  
`labels/`: CSV fold 0 và tên lớp để đối chiếu khi chấm; không chứa toàn bộ dataset ảnh.  
`error_images/`: metadata tám lỗi; ảnh minh hoạ được ghép ở curves/error_examples.png, không kèm JPEG dataset nguồn.  
`evidence/`: bảng CSV, selection trước test, COMPLETE, kiểm tra và phép chấm lại. Đối chiếu yêu cầu đầy đủ tại [RUBRIC_CHECK.md](RUBRIC_CHECK.md).

Excel và báo cáo được hậu xử lý từ kết quả đã lưu sau khi Kaggle chạy xong; output Kaggle chứa bản thô. Code hậu xử lý là `build_report.py` và `build_workbook.mjs`. Có thể dựng lại báo cáo bằng `python code/build_report.py` với numpy/pandas/matplotlib/Pillow. Builder workbook dùng thư viện Artifact Tool trong môi trường Codex Spreadsheets và dữ liệu evidence/workbook-data.json; thư viện này không thuộc môi trường training Kaggle. Không có thay đổi model hoặc predictions trong hậu xử lý.

## Cách tính lại điểm (không chạy model)

Trong thư mục gói nộp, dùng Python với numpy và pandas:

```bash
python -X utf8 eval.py score --pred "predictions/F01_seed*_test.csv" --test-csv labels/test_subset0.csv --labels labels/labels.csv --tag F01 --out evidence/recomputed
python -X utf8 eval.py score --pred "predictions/T00_final_seed*_test.csv" --test-csv labels/test_subset0.csv --labels labels/labels.csv --tag T00_final --out evidence/recomputed
python -X utf8 eval.py grade --final "predictions/F01_seed*_test.csv" --baseline "predictions/T00_final_seed*_test.csv" --uncal "predictions/F01uncal_seed*_test.csv" --final-val "predictions/F01_seed*_val.csv" --test-csv labels/test_subset0.csv --val-csv labels/val_subset0.csv --labels labels/labels.csv --latency-p95-ms 14.51899485 --latency-method proper --out evidence/recomputed
```

Phép score/grade chỉ đọc dự đoán đã lưu. p95 dùng ở grade là **forward-only** của TTA trên Tesla T4, chưa tính preprocessing, truyền dữ liệu, gộp xác suất hoặc camera. Không xem điểm latency tự chấm là xác nhận end-to-end cho robot.

## Tái lập thí nghiệm từ đầu

Môi trường lần chạy: Python 3.13.15, PyTorch 2.11.0+cu128, timm 1.0.29, Tesla T4. `requirements.txt` pin torch/timm chính; phiên bản gói phụ không được lưu đầy đủ nên không khẳng định môi trường bit-for-bit. Các tag pretrained và mean/std thật được lưu ở mỗi config.

Trên một phiên Kaggle mới, gắn dataset JPEG DeepWeeds đầy đủ và CSV gốc. Notebook source trong code có thứ tự: setup → kiểm tra split → EDA/core tests → Config → overfit một batch → run_all. Bật GPU và Internet lấy pretrained weights. Có thể gắn output version 4 để tái sử dụng các backbone, hoặc bỏ input cũ để chạy backbone từ đầu. Script sẽ khóa selection trước test và giữ predictions khi orchestration tiếp tục trên cùng working directory.

Đây là hướng dẫn tái lập bằng một lần chạy độc lập, không phải bước cần làm thêm cho bài nộp hiện tại. Kết quả test hiện tại đã chốt: không dùng phân tích lỗi để đổi cấu hình rồi chạy test lại. Seed 0 của các run giống hệt được tái sử dụng, không tính thành thêm seed. Builder báo cáo giữ nguyên hình lỗi đã ghép nếu không có JPEG nguồn; không tạo hình trống thay bằng chứng thật.

## Latency bổ sung

Đo sau lần chạy chính, từ checkpoint gốc, trên tensor ngẫu nhiên 224 px: 5 backbone batch 1, 6 phương pháp batch 32, ba recipe T03/T05/T07 batch 1. Warmup 10, 100 lượt và đồng bộ CUDA; không tạo test loader. Số liệu nằm trong evidence/latency_supplement.csv và sheet Latency. T06_ls dùng phép đo I00 vì cùng checkpoint. Throughput = batch × 1000/p50. Native Excel chưa được mở để chạy công thức; công thức đã được tính/kiểm tra bằng Artifact Tool và kiểm tra giá trị cache trong XLSX.
