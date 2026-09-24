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

- **Đa miền đánh giá**: Tối thiểu **1 source domain** + ít nhất **3 target domains / shift levels** (Gaussian Noise, Fog, Motion Blur, Contrast, Brightness, Pixelate với 5 cấp độ severity).
- **Đo lường downstream**: Có mô hình / task cụ thể để định lượng mức suy giảm hiệu năng (*Performance Drop*) trên tập evaluation.
- **Không chỉ đo khoảng cách hình thức**: Chứng minh tương quan thực tế với mức sụt giảm hiệu năng (Performance Drop), không chỉ dừng lại ở đo khoảng cách phân phối.
- **Phân tích sai số thực tế**: Phát hiện và báo cáo chi tiết **False Alarm** (báo động giả khi performance không tụt) hoặc **Miss** (bỏ sót khi performance tụt nghiêm trọng mà không phát hiện).

---

## 3. Không gian nghiên cứu & Phương pháp triển khai (Implemented Detectors)

Hệ thống tích hợp sẵn các thuật toán unsupervised shift & risk detection hàng đầu:
- **ATC (Average Thresholded Confidence - NeurIPS 2021)**: Tính toán ngưỡng tin cậy $t$ trên tập validation để ước lượng trực tiếp mức sụt giảm độ chính xác trên target batch.
- **Predictive Softmax Entropy**: Đo lường mức độ bất định (uncertainty) của mô hình trước dữ liệu mới.
- **Confidence Drop (tư tưởng NannyML CBPE)**: Theo dõi độ lệch mức độ tự tin lớn nhất (max-probability) so với phân phối chuẩn.
- **MMD (Maximum Mean Discrepancy)**: Kiểm định Two-Sample Test trên không gian biểu diễn đặc trưng ẩn (latent feature embeddings) với RBF kernel.
- **Slice Localization**: Tự động phân rã và xếp hạng lát cắt dữ liệu chịu ảnh hưởng rủi ro cao nhất.

---

## 4. Hệ thống Đánh giá (Evaluation Metrics)

### 🎯 Primary Metric
- **Spearman $\rho(\text{ShiftScore}, \text{PerformanceDrop})$**: Đo lường tương quan thứ hạng giữa *Shift Score* được tính toán không nhãn và mức sụt giảm hiệu năng thực tế (*Ground-truth Performance Drop*).

### 📊 Secondary Metrics
- **Shift-Detection AUROC**: Khả năng phân biệt giữa in-distribution vs shifted distributions.
- **False Alarm Rate (FAR)** & **Miss Rate**: Tỷ lệ cảnh báo sai trên dữ liệu lành tính và tỷ lệ bỏ sót rủi ro thật.
- **Slice Localization Ranking**: Độ chính xác khi định vị phân vùng/lát cắt dữ liệu chịu rủi ro cao nhất.

---

## 5. Cấu trúc Dự án (Repository Structure)

```text
C2-Domain-Shift-Radar/
├── .gitignore                   # Cấu hình loại bỏ dữ liệu nặng, model weights, venv
├── README.md                    # Tài liệu hướng dẫn & quy chuẩn bài toán
├── requirements.txt             # Danh sách thư viện phụ thuộc
├── run_radar.py                 # Pipeline chạy đánh giá End-to-End chính
├── src/
│   ├── data/
│   │   ├── corruptions.py       # Bộ sinh domain shifts (Noise, Fog, Blur, Brightness, Contrast...)
│   │   └── loader.py            # DataLoader CIFAR-10 chia tách Source Val & Test
│   ├── models/
│   │   └── model.py             # ResNet-18 Backbone & trích xuất latent features/logits
│   ├── radar/
│   │   ├── detectors.py         # Triển khai ATC, Entropy, ConfidenceDrop, MMD
│   │   └── slice_locator.py     # Module khoanh vùng lát cắt (Slice Localization)
│   ├── evaluation/
│   │   └── metrics.py           # Tính Spearman rho, AUROC, False Alarm & Misses
│   └── utils/
│       └── visualizer.py        # Vẽ biểu đồ tương quan Spearman & đường cong rủi ro
└── results/                     # Thư mục lưu biểu đồ và báo cáo kết quả tự động
```

---

## 6. Hướng dẫn Dành cho Team (Quickstart Guide)

### 🚀 Bước 1: Clone repository
```bash
git clone https://github.com/davidhevn/C2-Domain-Shift-Radar.git
cd C2-Domain-Shift-Radar
```

### 📦 Bước 2: Cài đặt môi trường
Khuyến nghị tạo môi trường ảo Python 3.10+:
```bash
python -m venv .venv
# Trên Windows:
.venv\Scripts\activate
# Trên Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### ⚡ Bước 3: Chạy thử nghiệm Radar Pipeline
Chạy pipeline đánh giá tự động trên toàn bộ các miền dịch chuyển và xuất báo cáo:
```bash
# Chạy đầy đủ:
python run_radar.py

# Hoặc chạy kiểm tra nhanh (chọn 200 mẫu mỗi batch):
python run_radar.py --samples_per_eval 200
```

Kết quả báo cáo bảng số liệu, phân tích False Alarm/Miss và biểu đồ Spearman $\rho$ sẽ tự động lưu vào thư mục `results/`.

---

## 7. Kết quả Fine-tune & Đánh giá Thực tế (Finetuning Results)

Mô hình hiện đã được fine-tune hoàn chỉnh trên tập CIFAR-10 trong 15 Epochs, với trọng số tối ưu nhất đạt mức **Clean Baseline Accuracy: 70.65%**.

Dưới đây là thống kê độ sụt giảm hiệu năng (Accuracy Drop) trực tiếp khi mô hình bị đánh giá trên 7 loại nhiễu vật lý (Severity = 3):

| Loại Nhiễu (Corruption) | Độ chính xác (Accuracy) | Độ sụt giảm (Drop) | Đánh giá Mức độ Ảnh hưởng |
| :--- | :---: | :---: | :--- |
| **Gaussian Noise** | 43.00% | `-27.65%` | Trung bình cao |
| **Brightness** | 56.95% | `-13.70%` | Thấp |
| **Contrast** | 39.20% | `-31.45%` | Cao |
| **Pixelate** | 42.70% | `-27.95%` | Trung bình cao |
| **Motion Blur** | 25.70% | `-44.95%` | **Cực kì nghiêm trọng (Nặng nhất)** |
| **Fog** | 38.40% | `-32.25%` | Cao |
| **Salt & Pepper** | 33.35% | `-37.30%` | Rất nghiêm trọng |

*Lưu ý: Các số liệu này đại diện cho Ground-Truth Risk của mô hình downstream khi chạy trên thực tế, phục vụ làm cột mốc để Calibration các chỉ số như Pixel/Feature Distance hay Predictive Entropy.*

---

## 8. Tài Nguyên (Dataset & Model Repository)

Dành cho các thành viên hoặc người dùng muốn tái tạo lại (reproduce) thử nghiệm trên máy cá nhân:

- **Dataset**: Dự án sử dụng bộ dữ liệu chuẩn **CIFAR-10** (phiên bản hình ảnh 32x32).
  - Tải tập dữ liệu gốc tại đây: [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
  - Giải nén và đặt các class folders của tập Test vào đường dẫn: `cifar10/test/`
  
- **Model Checkpoints**: 
  - Mã nguồn sử dụng kiến trúc **Custom CIFAR ResNet-18** (thiết kế riêng cho ảnh nhỏ 32x32, không dùng MaxPool ở đầu vào).
  - Trọng số mô hình (Weights) có thể tự động tạo ra bằng cách chạy script huấn luyện có sẵn: `python finetune_and_evaluate.py`
  - Link Repo chính thức của dự án: [https://github.com/davidhevn/C2-Domain-Shift-Radar](https://github.com/davidhevn/C2-Domain-Shift-Radar)
