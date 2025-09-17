import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.cm as cm

# --- Page config ---
st.set_page_config(page_title="Flower Classifier with Grad-CAM", layout="centered")

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

# --- Grad-CAM functions ---
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="Conv_1", pred_index=None):
    # Access the base MobileNetV2 inside the fine-tuned model
    base_model = model.layers[0]  # Usually first layer

    # Create a model mapping input to last conv layer and predictions
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [base_model.get_layer(last_conv_layer_name).output, model.output]
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

# --- Streamlit UI ---
st.title("🌸 Flower Classifier with Grad-CAM")
st.write(
    "Upload a flower image to get predictions and see the model's attention using Grad-CAM."
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

    # Generate Grad-CAM heatmap
    with st.spinner("Generating Grad-CAM..."):
        img_arr = preprocess_image(image)
        img_batch = np.expand_dims(img_arr, axis=0)
        heatmap = make_gradcam_heatmap(img_batch, model, last_conv_layer_name="Conv_1")
        gradcam_img = overlay_gradcam(image, heatmap)

    st.image(gradcam_img, caption="Grad-CAM Heatmap", use_column_width=True)
