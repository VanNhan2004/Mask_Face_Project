# app.py (Streamlit version with tabs and enhanced visualization)
import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import datetime
import pandas as pd
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import tempfile
import time
import io

# ------------------ Khởi tạo ------------------
st.set_page_config(page_title="Nhận diện Khẩu Trang", layout="wide")
st.markdown("""
    <style>
    .stButton>button {
        background-color: #007BFF;
        color: white;
        border-radius: 8px;
        padding: 0.5em 1em;
        border: none;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

model = load_model("mask_detector.h5")
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

if "log" not in st.session_state:
    st.session_state.log = []
if "start_time" not in st.session_state:
    st.session_state.start_time = None

st.title("😷 Ứng dụng Nhận diện Đeo Khẩu Trang")
st.markdown(f"🕒 Thời gian hiện tại: **{datetime.datetime.now().strftime('%H:%M:%S')}**")

# ------------------ Hàm chính ------------------
def detect_and_predict_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    result_image = image.copy()
    summary = {"Mask": 0, "No Mask": 0}

    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)

        pred = model.predict(face)[0][0]
        label = "Mask" if pred < 0.5 else "No Mask"
        confidence = 1 - pred if pred < 0.5 else pred
        summary[label] += 1

        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        cv2.putText(result_image, f"{label} ({confidence:.2%})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)

    return result_image, summary

# ------------------ Tabs ------------------
tabs = st.tabs(["📷 Ảnh & Video", "🎥 Webcam", "📊 Thống kê"])

# ------------------ Tab 1: Ảnh & Video ------------------
with tabs[0]:
    uploaded_file = st.file_uploader("📁 Tải ảnh hoặc video...", type=["jpg", "jpeg", "png", "mp4", "avi"])
    if uploaded_file:
        if uploaded_file.type.startswith("image"):
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.image(image, caption="Ảnh đã tải lên", width=600)

            if st.button("🚀 Bắt đầu nhận diện ảnh"):
                output, res = detect_and_predict_mask(image)
                st.image(output, caption="Kết quả", width=600)
                st.success(f"😷 Đeo khẩu trang: {res['Mask']}")
                st.error(f"❌ Không đeo: {res['No Mask']}")
                st.session_state.log.append({"timestamp": datetime.datetime.now(), "mask": res["Mask"], "no_mask": res["No Mask"]})

                result_pil = Image.fromarray(output)
                img_bytes = io.BytesIO()
                result_pil.save(img_bytes, format="JPEG")
                st.download_button("📥 Tải ảnh kết quả", img_bytes.getvalue(), file_name="result.jpg")

        elif uploaded_file.type.startswith("video"):
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()

            st.session_state.start_time = time.time()
            autostop = st.number_input("⏱️ Tự động dừng sau (giây)", min_value=0, value=0, step=1)

            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".avi").name
            out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

            while cap.isOpened():
                if autostop > 0 and time.time() - st.session_state.start_time > autostop:
                    break
                ret, frame = cap.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                output, res = detect_and_predict_mask(rgb)
                stframe.image(output, channels="RGB", width=720)
                st.session_state.log.append({"timestamp": datetime.datetime.now(), "mask": res["Mask"], "no_mask": res["No Mask"]})
                out.write(cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

            cap.release()
            out.release()
            with open(output_path, "rb") as f:
                st.download_button("📥 Tải video kết quả", f.read(), file_name="result_video.avi")

# ------------------ Tab 2: Webcam ------------------
with tabs[1]:
    class MaskDetector(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result, res = detect_and_predict_mask(rgb)
            st.session_state.log.append({"timestamp": datetime.datetime.now(), "mask": res["Mask"], "no_mask": res["No Mask"]})
            return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    webrtc_streamer(key="webcam", video_transformer_factory=MaskDetector)

# ------------------ Tab 3: Thống kê ------------------
with tabs[2]:
    st.subheader("📊 Lịch sử nhận diện")
    if st.session_state.log:
        df = pd.DataFrame(st.session_state.log)
        st.dataframe(df.tail(10))

        mask_total = df["mask"].sum()
        no_mask_total = df["no_mask"].sum()
        total = mask_total + no_mask_total

        if total > 0:
            st.markdown(f"### 📈 Tỷ lệ tổng thể")
            st.write(f"✅ Đeo khẩu trang: {mask_total} ({mask_total/total:.1%})")
            st.write(f"❌ Không đeo: {no_mask_total} ({no_mask_total/total:.1%})")

            col1, col2 = st.columns(2)
            with col1:
                st.bar_chart({"Mask": mask_total, "No Mask": no_mask_total})
            with col2:
                df_grouped = df.set_index("timestamp").resample("1min").sum(numeric_only=True)
                st.line_chart(df_grouped)

        if st.button("🔁 Reset thống kê"):
            st.session_state.log = []
            st.experimental_rerun()

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📄 Tải CSV", csv, "log_detection.csv", "text/csv")