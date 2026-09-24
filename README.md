# C2 - Domain Shift Radar 📡

> **Cảnh báo sớm suy giảm hiệu năng mô hình dưới tác động của dịch chuyển dữ liệu (Domain Shift) khi không có ground-truth labels.**

---

## 1. Giới thiệu & Bài toán Production (Context & Challenge)

Trong thực tế triển khai, mô hình học máy đang hoạt động ổn định trên tập nguồn (**Source Domain**). Khi tiếp nhận dữ liệu mới trong môi trường thực tế (từ các camera mới, điều kiện thời tiết, khung giờ khác, các thành phố hoặc sensor khác biệt), dữ liệu thường chưa có sẵn nhãn (unlabeled) hoặc nhãn về rất trễ. Điều này dẫn tới câu hỏi cốt lõi:

> **Câu hỏi Production:** *"Batch dữ liệu mới khác biệt so với training domain tới mức nào, và rủi ro sụt giảm hiệu năng (risk) của mô hình đã đủ lớn để cần can thiệp hay chưa?"*

Hệ thống **Domain Shift Radar** được thiết kế nhằm:
1. **Phát hiện dịch chuyển (Shift Detection)**: Tạo tín hiệu cảnh báo rủi ro (**Shift Score / Risk Signal**) theo thời gian thực hoặc theo từng batch.
2. **Khoanh vùng lát cắt (Slice Localization)**: Chỉ rõ lát cắt dữ liệu (slice / segment) nào đang bị shift mạnh nhất.
3. **Dự báo rủi ro thực tế (Risk-to-Drop Correlation)**: Chứng minh rằng tín hiệu shift có liên hệ chặt chẽ với sự sụt giảm hiệu năng thực tế (*performance degradation*) của downstream model.

---

## 2. Tiêu chuẩn & Yêu cầu tối thiểu (Minimum Requirements)

- **Đa miền đánh giá**: Tối thiểu **1 source domain** + ít nhất **3 target domains / shift levels** (ví dụ: CIFAR-10 vs CIFAR-10-C với các mức corruptions và severity khác nhau).
- **Đo lường downstream**: Có mô hình / task cụ thể để định lượng mức suy giảm hiệu năng (*Performance Drop*) trên tập evaluation.
- **Không chỉ đo khoảng cách hình thức**: Nghiêm cấm việc chỉ dừng lại ở việc chứng minh source và target "khác nhau" về mặt phân phối thuần túy.
- **Phân tích sai số thực tế**: Phải phát hiện và báo cáo ít nhất **1 trường hợp False Alarm** (báo động giả khi performance không tụt) hoặc **Miss** (bỏ sót khi performance tụt nghiêm trọng mà không phát hiện).

---

## 3. Không gian nghiên cứu (Research Space)

Hệ thống khai thác và so sánh các hướng tiếp cận chính:
- **Feature Statistics & Embeddings**: Giám sát phân phối không gian biểu diễn ẩn (latent features, representation drift).
- **Statistical Distances**: MMD (Maximum Mean Discrepancy), Wasserstein Distance, Two-Sample Classifier (C2ST), FID-like distance.
- **OOD Detection & Model Uncertainty**: Energy-based score, Mahalanobis distance, Softmax entropy, ATC (Average Thresholded Confidence).
- **VLM Semantics & Metadata**: Sử dụng mô hình nền tảng (Vision-Language Models) để trích xuất ngữ nghĩa và thuộc tính lát cắt.
- **Drift Detection & Streaming**: Change-Point Detection, Page-Hinkley, kiểm định trượt trên dòng dữ liệu liên tục.

---

## 4. Hệ thống Đánh giá (Evaluation Metrics)

### 🎯 Primary Metric
- **Spearman $\rho(\text{ShiftScore}, \text{PerformanceDrop})$**: Đo lường tương quan thứ hạng giữa *Shift Score* được tính toán không nhãn và mức sụt giảm hiệu năng thực tế (*Ground-truth Performance Drop*).

### 📊 Secondary Metrics
- **Shift-Detection AUROC**: Khả năng phân biệt giữa in-distribution vs shifted distributions.
- **False Alarm Rate (FAR)**: Tỷ lệ cảnh báo sai trên dữ liệu lành tính.
- **Slice Localization Accuracy**: Độ chính xác khi định vị phân vùng/nhóm dữ liệu chịu ảnh hưởng lớn nhất.
- **Score Calibration**: Mức độ hiệu chuẩn giữa giá trị Shift Score và phần trăm hiệu năng tụt.
- *(Streaming Bonus)*: Time-to-Detect (TTD), số false alarms/giờ, độ ổn định của tín hiệu cảnh báo.

---

## 5. Lưu ý & Bẫy Production (Pitfall Warning)

> ⚠️ **Bẫy thường gặp**: Một thước đo khoảng cách (distance metric) dù trông "rất khoa học" và có nền tảng toán học phức tạp, nhưng nếu **không có tương quan hay khả năng dự báo mức rủi ro sụt giảm của downstream model**, thì giá trị áp dụng trong môi trường production là cực kỳ thấp.
