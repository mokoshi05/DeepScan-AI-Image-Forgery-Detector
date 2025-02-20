import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from PIL import Image

# Set page config
st.set_page_config(
    page_title="Deepfake Detection",
    page_icon="🔍",
    layout="centered"
)

# Title and description
st.title("Deepfake Detection")
st.write("Upload an image to check if it's real or fake")

# Load the trained model
try:
    model = load_model('deepfake_detection_model.h5')
except:
    st.error("Error: Could not load model file. Please make sure 'deepfake_detection_model.h5' exists in the same directory.")
    st.stop()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    img = image.resize((96, 96))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    prediction = model.predict(img_array)
    result = prediction[0]
    
    # Display result
    st.write("## Analysis Result")
    if np.argmax(result) == 0:
        st.error("⚠️ This image appears to be FAKE")
        st.write(f"Confidence: {result[0]*100:.2f}%")
    else:
        st.success("✅ This image appears to be REAL")
        st.write(f"Confidence: {result[1]*100:.2f}%")

# Add instructions
st.markdown("""
### How to use:
1. Click 'Browse files' to upload an image
2. Wait for the analysis
3. View the result and confidence score

Note: For best results, use clear face images.""")

# Model Training Graph
st.title("Model Training Graph")
st.markdown("### Model Training accuracy: 95%")
st.image("Figure_2.png")
st.markdown("### Model Training Loss")
st.image("Figure_1.png")

# Footer section
st.markdown("""
---
**Contact Us:**
For more information and queries, please contact us at [contact@example.com](mailto:contact@example.com).

**Follow us on:**
[Twitter](https://twitter.com) | [LinkedIn](https://linkedin.com) | [Facebook](https://facebook.com)
""")
