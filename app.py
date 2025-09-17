import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import matplotlib.cm as cm
import cv2

# --- Page config ---
st.set_page_config(page_title="Flower Classifier with Grad-CAM", layout="wide")

# --- Helper functions ---
@st.cache_resource
def load_model():
    """
    Load your fine-tuned MobileNetV2 model trained on tf_flowers dataset.
    Make sure 'flowers_model.h5' is in the same directory or update path.
    """
    model = tf.keras.models.load_model("flowers_model.h5")
    return model

def pil_to_model_array(image: Image.Image, target_size=(224, 224)):
    """
    Convert PIL Image to a numpy array, resized and ready for model input.
    """
    img = image.convert("RGB").resize(target_size)
    arr = tf.keras.preprocessing.image.img_to_array(img)
    return arr

def build_augmentation_pipeline():
    """
    Build augmentation pipeline using Keras preprocessing layers.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.12),
        tf.keras.layers.RandomZoom(0.12),
        tf.keras.layers.RandomContrast(0.15),
    ])

def predict_with_augmentations(model, pil_image: Image.Image, num_augmentations: int = 3):
    """
    Apply augmentations to the image and average model predictions.
    Returns averaged probabilities and augmented images for display.
    """
    base_arr = pil_to_model_array(pil_image)
    base_batch = np.expand_dims(base_arr, axis=0).astype(np.float32)
    base_batch_tf = tf.convert_to_tensor(base_batch)
    batch = tf.repeat(base_batch_tf, repeats=num_augmentations, axis=0)

    aug_pipeline = build_augmentation_pipeline()
    augmented_batch = aug_pipeline(batch, training=True)

    # Clip values for displaying augmented images
    augmented_display = tf.clip_by_value(augmented_batch, 0.0, 255.0)
    augmented_display_uint8 = tf.cast(augmented_display, tf.uint8).numpy()

    # Preprocess for MobileNetV2
    processed = tf.keras.applications.mobilenet_v2.preprocess_input(augmented_batch.numpy())

    preds = model.predict(processed, verbose=0)
    averaged_probs = np.mean(preds, axis=0)

    return averaged_probs, augmented_display_uint8

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generates a Grad-CAM heatmap for a given image and model.
    Args:
        img_array: Preprocessed input image array of shape (1, H, W, C)
        model: The keras model
        last_conv_layer_name: Name of the last conv layer in the model
        pred_index: Index of the predicted class to generate heatmap for (optional)
    Returns:
        heatmap: 2D numpy array of shape (H, W) with values between 0 and 1
    """
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

    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
    heatmap = heatmap.numpy()
    return heatmap

def overlay_heatmap_on_image(img: Image.Image, heatmap: np.ndarray, alpha=0.4):
    """
    Overlays heatmap on PIL Image and returns combined PIL Image.
    """
    img = np.array(img.convert("RGB"))

    # Resize heatmap to image size
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Convert heatmap to RGB using colormap
    colormap = cm.get_cmap("jet")
    heatmap_colored = colormap(heatmap_resized)
    heatmap_colored = np.uint8(heatmap_colored[:, :, :3] * 255)

    # Combine heatmap with original image
    overlayed_img = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)

    return Image.fromarray(overlayed_img)

# --- Class names in correct order from tf_flowers ---
flower_classes = ['dandelion', 'daisy', 'tulips', 'sunflowers', 'roses']

# --- Main app UI and logic ---
st.title("🌸 Flower Classifier (tf_flowers) with Augmentation & Grad-CAM")
st.write("""
Upload a flower image, and get predictions from a fine-tuned MobileNetV2 model.
Uses augmentation and Grad-CAM for robust classification and interpretability.
""")

# Load model once
model = load_model()

# Last convolutional layer name for MobileNetV2 (adjust if needed)
last_conv_layer_name = "Conv_1"

# Sidebar options
with st.sidebar:
    st.header("Settings")
    num_augmentations = st.slider("Number of augmentations", 1, 8, value=3)
    show_augmented = st.checkbox("Show augmented images", value=True)
    show_original_prediction = st.checkbox("Show original image prediction", value=True)
    show_gradcam = st.checkbox("Show Grad-CAM heatmaps", value=True)

# Image uploader
uploaded_file = st.file_uploader("Choose a flower image...", type=["jpg", "jpeg", "png"])

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
            # Get top 3 indices sorted by probability
            top_indices = preds_orig.argsort()[-3:][::-1]

        st.subheader("Top-3 Predictions (Original Image)")
        for i in top_indices:
            st.write(f"**{flower_classes[i].title()}** — {preds_orig[i]:.2%}")

    # Show Grad-CAM on original image
    if show_gradcam and show_original_prediction:
        with st.spinner("Generating Grad-CAM for original image..."):
            top_pred_index = np.argmax(preds_orig)
            heatmap_orig = make_gradcam_heatmap(proc, model, last_conv_layer_name, pred_index=top_pred_index)
            overlayed_orig = overlay_heatmap_on_image(image, heatmap_orig)
        st.subheader("Grad-CAM on Original Image")
        st.image(overlayed_orig, use_column_width=True)

    # Augmentation-based prediction
    with st.spinner(f"Classifying with {num_augmentations} augmentations..."):
        averaged_probs, augmented_images = predict_with_augmentations(model, image, num_augmentations)
        top_indices_avg = averaged_probs.argsort()[-3:][::-1]

    st.subheader(f"Top-3 Predictions (Averaged over {num_augmentations} augmentations)")
    for i in top_indices_avg:
        st.write(f"**{flower_classes[i].title()}** — {averaged_probs[i]:.2%}")

    # Show augmented images used for prediction
    if show_augmented:
        st.subheader("Augmented Images Used for Prediction")
        cols = st.columns(min(4, num_augmentations))
        for idx in range(num_augmentations):
            aug_pil = Image.fromarray(augmented_images[idx])
            cols[idx % len(cols)].image(aug_pil, use_column_width=True, caption=f"Augmented #{idx+1}")

    # Show Grad-CAM for augmented images
    if show_gradcam and show_augmented:
        st.subheader("Grad-CAM on Augmented Images")
        overlayed_aug_imgs = []
        for i in range(num_augmentations):
            pred_index = np.argmax(augmented_preds[i])
            heatmap = make_gradcam_heatmap(processed[i:i+1], model, last_conv_layer_name, pred_index)

            # Use clipped uint
