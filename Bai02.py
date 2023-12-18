import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from keras.preprocessing import image as img_preprocess
from keras.models import load_model

# Đường dẫn đến file nhãn
LABELS_PATH = "Data/Chest X-Ray Images (Pneumonia).csv"
# Đường dẫn đến file mô hình
MODEL_PATH = "path/to/your/model.h5"

# Load file nhãn
labels_df = pd.read_csv(LABELS_PATH)
labels = labels_df["Label"].values.tolist()

# Load mô hình đã được huấn luyện
model = load_model(MODEL_PATH)

# Hàm để xử lý ảnh và dự đoán nhãn
def process_image(image):
    # Chuyển đổi ảnh thành mảng numpy
    img_array = img_preprocess.img_to_array(image)
    # Chuẩn hóa dữ liệu
    img_array = img_array / 255.0
    # Thêm một chiều để phù hợp với đầu vào của mô hình
    img_array = np.expand_dims(img_array, axis=0)
    # Dự đoán nhãn
    prediction = model.predict(img_array)
    # Lấy nhãn dự đoán cao nhất
    predicted_label = labels[np.argmax(prediction)]
    return predicted_label

# Giao diện ứng dụng
def main():
    st.title("Ứng dụng nhận diện bệnh nhiễm khuẩn phổi từ ảnh X-Quang")
    st.write("Upload một tấm ảnh X-Quang để nhận dự đoán")

    # Upload ảnh từ người dùng
    uploaded_image = st.file_uploader("Chọn tệp ảnh", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Đọc ảnh và hiển thị
        image = Image.open(uploaded_image)
        st.image(image, use_column_width=True)

        # Xử lý ảnh và dự đoán nhãn
        predicted_label = process_image(image)

        # Hiển thị nhãn dự đoán
        st.write("Nhãn dự đoán:", predicted_label)


if __name__ == "__main__":
    main()