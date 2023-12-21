import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.models import load_model

# Load pre-trained model
# https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
model = load_model('chest_xray_model')

# Define class labels
class_labels = ['NORMAL', 'PNEUMONIA']

# Function to preprocess image
def preprocess_image(image):
    img = image.resize((150, 150))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Streamlit app
def main():
    st.title('Pneumonia Detection from X-ray Images')
    st.text('Upload a chest X-ray image')

    # Upload image
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Preprocess and predict
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        predicted_class = class_labels[np.argmax(prediction)]

        # Display prediction
        st.subheader('Prediction:')
        st.write(f'The image is classified as {predicted_class}')

if __name__ == '__main__':
    main()