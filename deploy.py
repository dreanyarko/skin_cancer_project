import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import streamlit as st
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_v2_preprocess_input

model = tf.keras.models.load_model(
    "./model",
    custom_objects={'KerasLayer': hub.KerasLayer})

# load file
uploaded_file = st.file_uploader("Select an image", type="jpg")

labels = {0: 'Benign',
          1: 'Malignant'}

# Execute this code if an image is uploaded
if uploaded_file is not None:

    # Convert the uploaded image into an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_image, (224, 224))

    # Display the image
    st.image(opencv_image, channels="RGB")

    # Resize image to 224 x 224
    resized = mobilenet_v2_preprocess_input(resized)
    img_reshape = resized[np.newaxis, ...]

    # Check the prediction, either benign or malignant
    check_prediction = st.button("Check Prediction")
    if check_prediction:
        predicted_value = model.predict(img_reshape)
        predicted_label = int(np.round(predicted_value))
        st.title(f"PREDICTION: {predicted_value}")
        st.title(f"P. LABEL: {labels[predicted_label]}")
