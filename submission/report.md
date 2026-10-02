# Lab 16 — GCP CPU LightGBM report

1. Tôi dùng Google Cloud Platform ở `us-central1` / `us-central1-a`, VM `e2-medium` (2 vCPU, 4 GB RAM), source commit `55539f67d7c78b43afe334a2ec3271c4bfdbbe2d`.
2. Dataset Credit Card Fraud có 284,807 dòng và 30 feature (`Class` là target); script dùng seed 42, chia train/validation/test lần lượt 182,276 / 45,569 / 56,962 dòng.
3. Đọc dữ liệu mất 2.8005 giây; training LightGBM mất 2.7012 giây và best iteration là 13.
4. Trên tập test, AUC-ROC là 0.963426, Accuracy 0.993417, F1 0.314442, Precision 0.191537 và Recall 0.877551.
5. Latency inference một dòng là 1.3464 ms; throughput batch 1,000 dòng là 485,233.38 dòng/giây. Warm-up được thực hiện ngoài phần đo; latency là trung bình 1,000 lần `predict_proba` một dòng.
6. Ảnh tài nguyên được chụp sau benchmark: CPU gần như nhàn rỗi, RAM dùng khoảng 529 MiB / 3.8 GiB; interface `ens4` nhận khoảng 256.9 MB và gửi khoảng 1.06 MB.
7. Billing Reports chụp ngày 02/10/2026 hiển thị ₫0 và trial credit còn ₫7,791,151; Billing chưa cập nhật dữ liệu dịch vụ tại thời điểm chụp, nên không suy ra chi phí thực tế bằng 0.
8. Tôi đã tải `benchmark.py` và `benchmark_result.json` về laptop trước khi chạy `terraform destroy`; Terraform state cục bộ sau dọn dẹp ghi nhận 0 resources, tương ứng `terraform state list` rỗng.
