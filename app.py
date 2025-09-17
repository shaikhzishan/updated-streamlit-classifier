import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import os
import matplotlib.cm as cm

# --- Page config ---
st.set_page_config(page_title="Flower Classifier", layout="wide")

# --- Class Names for tf_flowers ---
CLASS_NAMES = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

# --- Load CSS if available ---
def load_css(file_name: str):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("flowers_model.h5")

def pil_to_model_array(image: Image.Image, target_size=(224, 224)):
    img = image.convert("RGB").resize(target_size)
    arr = tf.keras.preprocessing.image.img_to_array(img)
    return arr

def build_augmentation_pipeline():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.12),
        tf.keras.layers.RandomZoom(0.12),
        tf.keras.layers.RandomContrast(0.15),
    ])

def predict_with_augmentations(model, pil_image: Image.Image, num_augmentations: int = 3):
    base_arr = pil_to_model_array(pil_image)
    base_batch = np.expand_dims(base_arr, axis=0).astype(np.float32)
    base_batch_tf = tf.convert_to_tensor(base_batch)
    batch = tf.repeat(base_batch_tf, repeats=num_augmentations, axis=0)

    aug_pipeline = build_augmentation_pipeline()
    augmented_batch = aug_pipeline(batch, training=True)

    augmented_display = tf.clip_by_value(augmented_batch, 0.0, 255.0)
    augmented_display_uint8 = tf.cast(augmented_display, tf.uint8).numpy()

    processed = tf.keras.applications.mobilenet_v2.preprocess_input(augmented_batch.numpy())
    preds = model.predict(processed, verbose=0)

    averaged_probs = np.mean(preds, axis=0)
    return averaged_probs, augmented_display_uint8

def decode_top_k_custom(probs: np.ndarray, top_k: int = 3):
    indices = np.argsort(probs)[::-1][:top_k]
    return [(CLASS_NAMES[i], probs[i]) for i in indices]

# --- Grad-CAM ---
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="Conv_1", pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_gradcam(image, heatmap, alpha=0.4):
    img = np.array(image.resize((224, 224)))
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = Image.fromarray((jet_heatmap * 255).astype(np.uint8)).resize(
        (img.shape[1], img.shape[0])
    )
    jet_heatmap = np.array(jet_heatmap)

    superimposed_img = np.uint8(jet_heatmap * alpha + img)
    return Image.fromarray(superimposed_img)

# --- Main App ---
load_css("style.css")
model = load_model()

st.title("🌸 Flower Classifier with Augmentations & Grad-CAM")
st.write("Upload a flower image to classify it using your **custom-trained MobileNetV2** model. "
         "See augmented predictions and a Grad-CAM heatmap for model explainability.")

# Sidebar
with st.sidebar:
    st.header("Settings")
    num_augmentations = st.slider("Number of augmentations", 1, 8, value=3)
    show_augmented = st.checkbox("Show augmented images", value=True)
    show_original_prediction = st.checkbox("Show original image prediction", value=True)

uploaded_file = st.file_uploader("Upload a flower image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.subheader("Original Image")
    st.image(image, use_column_width=True)

    # Original image prediction
    if show_original_prediction:
        with st.spinner("Classifying original image..."):
            arr = pil_to_model_array(image)
            proc = tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(arr, axis=0))
            preds_orig = model.predict(proc, verbose=0)[0]
            top_preds = decode_top_k_custom(preds_orig)

        st.subheader("Top Predictions (Original Image)")
        for i, (label, score) in enumerate(top_preds):
            st.write(f"{i+1}. **{label.title()}** — {score:.2%}")

    # Augmented image prediction
    with st.spinner(f"Classifying with {num_augmentations} augmentations..."):
        averaged_probs, augmented_images = predict_with_augmentations(model, image, num_augmentations)
        decoded_avg = decode_top_k_custom(averaged_probs)

    st.subheader(f"Top Predictions (Averaged over {num_augmentations} augmentations)")
    for i, (label, score) in enumerate(decoded_avg):
        st.write(f"{i+1}. **{label.title()}** — {score:.2%}")

    # Show augmented images
    if show_augmented:
        st.subheader("Augmented Images Used")
        cols = st.columns(min(4, num_augmentations))
        for idx in range(num_augmentations):
            aug_pil = Image.fromarray(augmented_images[idx])
            cols[idx % len(cols)].image(aug_pil, use_column_width=True, caption=f"Aug #{idx+1}")

    # Grad-CAM
    st.subheader("Grad-CAM Heatmap")
    arr = pil_to_model_array(image)
    proc = tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(arr, axis=0))
    heatmap = make_gradcam_heatmap(proc, model, last_conv_layer_name="Conv_1")
    gradcam_img = overlay_gradcam(image, heatmap)
    st.image(gradcam_img, caption="Grad-CAM (Model Attention)", use_column_width=True)

st.markdown("---")
st.caption("Custom-trained MobileNetV2 | tf_flowers dataset | Streamlit + TensorFlow 🔍")
