import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import os

# Force CPU usage (optional; remove if you want GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

@st.cache_resource
def load_model():
    st.write("Loading model...")
    model = ResNet50(weights='imagenet')
    st.write("Model loaded!")
    return model

model = load_model()

st.title("AI Image Classifier with ResNet50")
st.write("Upload a photo of a natural object or scene, and the AI will classify it based on ImageNet categories.")

uploaded_file = st.file_uploader("Upload a natural image (e.g., animal, object, landscape)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)

    img_resized = img.resize((224, 224))
    x = image.img_to_array(img_resized)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    st.write("Running prediction...")
    preds = model.predict(x)
    st.write("Prediction complete!")

    decoded = decode_predictions(preds, top=3)[0]
    st.write("### Top Predictions:")
    for i, (imagenetID, label, prob) in enumerate(decoded):
        st.write(f"{i+1}. **{label}** with confidence {prob*100:.2f}%")
