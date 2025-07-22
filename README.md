
# 😷 Mask Detection App - Ứng dụng Nhận diện Đeo Khẩu Trang

Ứng dụng này giúp nhận diện việc đeo khẩu trang từ ảnh, video hoặc webcam theo thời gian thực. Giao diện xây dựng bằng Streamlit, mô hình học sâu sử dụng MobileNetV2 được huấn luyện để phân biệt giữa khuôn mặt có và không có khẩu trang.

---

## 🎯 Mục tiêu

- Tự động phát hiện người **không đeo khẩu trang**
- Hỗ trợ **ảnh, video** và **webcam thời gian thực**
- **Thống kê trực quan**: biểu đồ, phần trăm
- Giao diện hiện đại, dễ sử dụng

---

## 🧰 Công nghệ sử dụng

| Thành phần        | Công nghệ                             |
|-------------------|----------------------------------------|
| Ngôn ngữ          | Python 3.8+                            |
| Giao diện         | Streamlit                             |
| Nhận diện mặt     | Haar Cascade Classifier (OpenCV)       |
| Nhận diện khẩu trang | MobileNetV2 (Keras, TensorFlow)     |
| Trực quan hóa     | Streamlit Charts, Matplotlib, Pandas   |
| Truyền hình ảnh webcam | streamlit-webrtc                 |

---

## 📂 Cấu trúc thư mục

```
Mask_Face_Project/
│
├── app.py                          # Giao diện chính Streamlit
├── train_model.py                 # Huấn luyện mô hình MobileNetV2
├── main.py                        # Nhận diện đơn giản CLI
├── mask_detector.h5               # Mô hình đã huấn luyện
├── haarcascade_frontalface_default.xml  # Bộ nhận diện khuôn mặt
├── alert beep.wav                 # Âm thanh cảnh báo (tùy chọn)
├── dataset/                       # Dữ liệu train
│   ├── with_mask/
│   └── without_mask/
├── requirements.txt               # Thư viện cần thiết
└── README.md                      # Tài liệu mô tả
```

---

## 🚀 Cách sử dụng

### 1. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

Hoặc thủ công:

```bash
pip install streamlit opencv-python tensorflow pillow pandas streamlit-webrtc
```

> ⚠️ Khuyến nghị Python 3.8 hoặc 3.9 để đảm bảo tương thích TensorFlow 2.x

### 2. Chạy ứng dụng

```bash
streamlit run app.py
```

---

## ✨ Các tính năng chính

### 📷 Nhận diện từ ảnh và video
- Cho phép tải lên ảnh `.jpg`, `.png`, video `.mp4`, `.avi`
- Hiển thị kết quả ngay trên giao diện
- Cho phép **tải ảnh/video kết quả** về máy

### 🎥 Nhận diện từ Webcam
- Real-time nhận diện khuôn mặt và khẩu trang
- Vẽ khung màu theo kết quả (xanh/đỏ)
- Ghi log vào lịch sử nhận diện

### 📊 Thống kê & trực quan hóa
- Hiển thị **tổng số người đeo/không đeo khẩu trang**
- Tính **tỷ lệ phần trăm**
- Vẽ **biểu đồ cột**, **biểu đồ đường**
- Xuất thống kê ra **file CSV**
- Reset lịch sử với 1 nút bấm

### 🧠 Huấn luyện mô hình (tuỳ chọn)

```bash
python train_model.py
```

- Tự động tải mô hình MobileNetV2 gốc
- Train lại với ảnh trong thư mục `dataset/`

---

## 📌 Mẹo sử dụng

- Có thể **tải ảnh nhóm** để kiểm tra cùng lúc nhiều người
- Dùng camera **độ phân giải cao** để tăng độ chính xác
- Nên **huấn luyện lại mô hình** nếu bạn muốn mở rộng dữ liệu mới (ví dụ: khẩu trang vải, màu sắc lạ...)

---

## 📈 Một số cải tiến tương lai

- 🔊 Cảnh báo âm thanh khi phát hiện không đeo khẩu trang
- 👁️ Nhận diện khuôn mặt người dùng cụ thể (Face Recognition)
- 📧 Gửi email thông báo vi phạm
- 🕵️‍♂️ Hệ thống giám sát nhiều camera

---

## 👨‍💻 Tác giả

- **Họ tên:** Nguyễn Văn Nhân  
- **MSSV:** 2200002045
- **Môn học:** Đồ án chuyên ngành Khoa học Dữ liệu  
- **Trường:** Trường Đại học Nguyễn Tất Thành 
- **Năm thực hiện:** 2025