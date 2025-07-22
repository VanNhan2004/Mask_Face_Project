import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Cấu hình
DATASET_DIR = "dataset"
EPOCHS = 10
INIT_LR = 1e-4
BS = 32
IMAGE_SIZE = (224, 224)

print("[INFO] Loading images...")
data = []
labels = []

for label in ["with_mask", "without_mask"]:
    folder_path = os.path.join(DATASET_DIR, label)
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        image = load_img(image_path, target_size=IMAGE_SIZE)
        image = img_to_array(image)
        image = preprocess_input(image)

        data.append(image)
        labels.append(0 if label == "with_mask" else 1)

data = np.array(data, dtype="float32")
labels = np.array(labels)

# CHIA DỮ LIỆU THÀNH 3 PHẦN: train - val - test
(trainX, tempX, trainY, tempY) = train_test_split(data, labels, test_size=0.3, stratify=labels, random_state=42)
(valX, testX, valY, testY) = train_test_split(tempX, tempY, test_size=0.5, stratify=tempY, random_state=42)

# Augmentation
aug = ImageDataGenerator(
    zoom_range=0.3,
    brightness_range=[0.5, 1.5],
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

# MobileNetV2
baseModel = MobileNetV2(weights="imagenet", include_top=False,
                        input_tensor=Input(shape=(224, 224, 3)))

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten()(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(1, activation="sigmoid")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

# Freeze feature extractor
for layer in baseModel.layers:
    layer.trainable = False

# Biên dịch mô hình
model.compile(loss="binary_crossentropy",
              optimizer=Adam(learning_rate=INIT_LR),
              metrics=["accuracy"])

# Huấn luyện mô hình
print("[INFO] Training model...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(valX, valY),
    validation_steps=len(valX) // BS,
    epochs=EPOCHS
)

# Đánh giá trên tập kiểm định (test)
print("[INFO] Evaluating on test set...")
predIdxs = model.predict(testX, batch_size=BS)
predIdxs = (predIdxs > 0.5).astype("int32")

# In báo cáo phân loại và độ chính xác
print(classification_report(testY, predIdxs, target_names=["Mask", "No Mask"]))
acc = accuracy_score(testY, predIdxs)
print(f"[RESULT] Test Accuracy: {acc:.4f}")

# Lưu mô hình
model.save("mask_detector.h5")
print("[INFO] Model saved as mask_detector.h5")