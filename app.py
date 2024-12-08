import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os

# Load the trained model
model = tf.keras.models.load_model('final_v1.h5')

# Define allowed image types
ALLOWED_TYPES = ["png", "jpg", "jpeg"]

# Streamlit application
st.title("Emotion Detection from Image")

# Upload image section
st.header("Upload an image for emotion detection")
uploaded_file = st.file_uploader("Choose an image...", type=ALLOWED_TYPES)

if uploaded_file is not None:
    # Check file size (limit to 10MB for example)
    file_size = uploaded_file.size
    if file_size > 10 * 1024 * 1024:  # 10MB limit
        st.error("File is too large. Please upload an image smaller than 10MB.")
    else:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image
        img = image.convert('L')  # Convert to grayscale
        img = img.resize((64, 64))  # Resize to 64x64 pixels
        img_array = np.array(img) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension (64, 64, 1)
        img_array = np.expand_dims(img_array, axis=0)   # Add batch dimension (1, 64, 64, 1)

        # Make prediction
        if st.button('Predict Emotion'):
            predictions = model.predict(img_array)
            class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
            predicted_class = class_names[np.argmax(predictions)]

            # Display the prediction result
            st.write(f"Predicted Emotion: **{predicted_class}**")
else:
    st.warning("Please upload an image file.")
