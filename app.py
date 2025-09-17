import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import os
import matplotlib.cm as cm

# --- Page config ---
st.set_page_config(page_title="Universal Image Classifier", layout="wide")

# --- Helper functions ---

def load_css(file_name: str):
    """Load CSS file into Streamlit if exists (silent fallback)."""
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load MobileNetV2 model (pretrained on ImageNet)."""
    model = tf.keras.applications.MobileNetV2(weights="imagenet")
    return model

def pil_to_model_array(image: Image.Image, target_size=(224, 224)):
    """Convert PIL image to a float32 numpy array suitable for the model."""
    img = image.convert("RGB").resize(target_size)
    arr = tf.keras.preprocessing.image.img_to_array(img)  # float32
    return arr

def build_augmentation_pipeline():
    """Create augmentation pipeline using Keras preprocessing layers."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.12),
        tf.keras.layers.RandomZoom(0.12),
        tf.keras.layers.RandomContrast(0.15),
    ])

def predict_with_augmentations(model, pil_image: Image.Image, num_augmentations: int = 3):
    """Apply augmentations, predict with model, return averaged probs + augmented images."""
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

def decode_top_k_from_probs(probs: np.ndarray, top_k: int = 3):
    """Decode top-k predictions from probs array."""
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(
        np.expand_dims(probs, axis=0), top=top_k
    )[0]
    return decoded

# --- Grad-CAM functions ---
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="Conv_1", pred_index=None):
    """Generate Grad-CAM heatmap for an image and model."""
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
    """Overlay Grad-CAM heatmap on original image."""
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

# --- Main App UI & Logic ---

load_css("style.css")
model = load_model()

st.title("🖼️ Universal Image Classifier with Augmentation + Grad-CAM")
st.write(
    "Upload an image to get predictions from **MobileNetV2**. "
    "The app applies augmentations for robust classification and shows a **Grad-CAM heatmap** for explainability."
)

# Sidebar
with st.sidebar:
    st.header("Settings")
    num_augmentations = st.slider("Number of augmentations", 1, 8, value=3)
    show_augmented = st.checkbox("Show augmented images", value=True)
    show_original_prediction = st.checkbox("Show original image prediction", value=True)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.subheader("Original Image")
    st.image(image, use_column_width=True)

    # Original image prediction
    if show_original_prediction:
        with st.spinner("Classifying original image..."):
            arr = pil_to_model_array(image)
            proc = tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(arr, axis=0))
            preds_orig = model.predict(proc, verbose=0)
            decoded_orig = tf.keras.applications.mobilenet_v2.decode_predictions(preds_orig, top=3)[0]

        st.subheader("Top-3 Predictions (Original Image)")
        for i, (imagenet_id, label, score) in enumerate(decoded_orig):
            st.write(f"{i+1}. **{label.replace('_',' ').title()}** — {score:.2%}")

    # Augmentation-based prediction
    with st.spinner(f"Classifying with {num_augmentations} augmentations..."):
        averaged_probs, augmented_images = predict_with_augmentations(model, image, num_augmentations)
        decoded_avg = decode_top_k_from_probs(averaged_probs, top_k=3)

    st.subheader(f"Top-3 Predictions (Averaged over {num_augmentations} augmentations)")
    for i, (imagenet_id, label, score) in enumerate(decoded_avg):
        st.write(f"{i+1}. **{label.replace('_',' ').title()}** — {score:.2%}")

    # Show augmented images
    if show_augmented:
        st.subheader("Augmented Images Used for Prediction")
        cols = st.columns(min(4, num_augmentations))
        for idx in range(num_augmentations):
            aug_pil = Image.fromarray(augmented_images[idx])
            cols[idx % len(cols)].image(aug_pil, use_column_width=True, caption=f"Augmented #{idx+1}")

    # Grad-CAM on original
    st.subheader("Grad-CAM Heatmap (Model Attention)")
    arr = pil_to_model_array(image)
    proc = tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(arr, axis=0))
    heatmap = make_gradcam_heatmap(proc, model, last_conv_layer_name="Conv_1")
    gradcam_img = overlay_gradcam(image, heatmap)
    st.image(gradcam_img, caption="Grad-CAM Visualization", use_column_width=True)

st.markdown("---")
st.caption("Built with Streamlit + TensorFlow | Demonstrates data augmentation and model explainability (Grad-CAM).")
