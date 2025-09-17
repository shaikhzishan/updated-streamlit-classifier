import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --- Page config ---
st.set_page_config(page_title="Flower Classifier", layout="centered")

# --- Load the fine-tuned model ---
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("flowers_model.h5")
    return model

model = load_model()

# --- Class names ---
flower_classes = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

# --- Preprocess function ---
def preprocess_image(image: Image.Image, target_size=(224, 224)):
    img = image.convert("RGB").resize(target_size)
    arr = tf.keras.preprocessing.image.img_to_array(img)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return arr

# --- Prediction function ---
def predict(image: Image.Image):
    img_arr = preprocess_image(image)
    img_batch = np.expand_dims(img_arr, axis=0)
    preds = model.predict(img_batch)
    return preds[0]  # Array of probabilities for 5 classes

# --- Streamlit UI ---
st.title("🌸 Flower Classifier")
st.write(
    "Upload an image of a flower and get predictions from a MobileNetV2 model fine-tuned on flower species."
)

uploaded_file = st.file_uploader("Upload a flower image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Classifying..."):
        preds = predict(image)

    top_idx = np.argmax(preds)
    top_prob = preds[top_idx]

    st.markdown(f"### Prediction: **{flower_classes[top_idx].title()}**")
    st.write(f"Confidence: {top_prob:.2%}")

    # Show top-3 predictions
    st.markdown("### Top 3 Predictions:")
    top3_indices = preds.argsort()[-3:][::-1]
    for i in top3_indices:
        st.write(f"- {flower_classes[i].title()}: {preds[i]:.2%}")
