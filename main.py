import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from playsound import playsound
import threading

# Load mô hình khẩu trang
model = load_model("mask_detector.h5")

# Mediapipe face detector
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Hàm cảnh báo bằng âm thanh
def play_alert_sound():
    playsound("alert beep.wav")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (800, 600))
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image_rgb)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x = int(bboxC.xmin * iw)
            y = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)

            # Cắt ảnh khuôn mặt
            face = frame[y:y+h, x:x+w]
            try:
                face_input = cv2.resize(face, (224, 224))
            except:
                continue  # bỏ qua nếu cắt ngoài ảnh

            face_input = img_to_array(face_input)
            face_input = preprocess_input(face_input)
            face_input = np.expand_dims(face_input, axis=0)

            (mask_score,) = model.predict(face_input, verbose=0)[0]
            label = "No Mask" if mask_score >= 0.5 else "Mask"
            color = (0, 0, 255) if label == "No Mask" else (0, 255, 0)

            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            if label == "No Mask":
                threading.Thread(target=play_alert_sound, daemon=True).start()

    cv2.imshow("Mask Detector - Press Q to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
