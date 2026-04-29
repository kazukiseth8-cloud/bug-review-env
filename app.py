import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ── CONFIG ────────────────────────────────────────────────────────────────────
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
IMG_SIZE    = 224

DISPOSAL_TIPS = {
    'cardboard': ('♻️', 'Flatten it, keep it dry, place in recycling bin.'),
    'glass':     ('🫙', 'Rinse thoroughly. Place in glass recycling — do not mix with other recyclables.'),
    'metal':     ('🥫', 'Rinse cans and tins. Place in metal/mixed recycling bin.'),
    'paper':     ('📄', 'Keep dry. Remove plastic windows from envelopes before recycling.'),
    'plastic':   ('🧴', 'Check the recycling number on the bottom. Rinse before disposal.'),
    'trash':     ('🗑️',  'This item cannot be recycled. Dispose in general waste bin.')
}

# ── PAGE SETUP ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="♻️ Waste Classifier",
    page_icon="♻️",
    layout="centered"
)

st.title("♻️ Waste Classification System")
st.markdown("Upload an image of waste and the model will classify it into one of 6 categories.")
st.markdown("**Categories:** Cardboard · Glass · Metal · Paper · Plastic · Trash")
st.divider()

# ── MODEL LOADING ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    # ✅ Point this to wherever you downloaded waste_model_v2.h5
    return tf.keras.models.load_model("waste_model_v2.h5")

with st.spinner("Loading model..."):
    model = load_model()

# ── FILE UPLOAD ───────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    # ── PREPROCESSING & PREDICTION ────────────────────────────────────────────
    img_resized = image.resize((IMG_SIZE, IMG_SIZE))
    img_array  = np.array(img_resized) / 255.0
    img_array  = np.expand_dims(img_array, axis=0)

    with st.spinner("Classifying..."):
        predictions  = model.predict(img_array, verbose=0)[0]

    predicted_idx   = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence      = float(predictions[predicted_idx])
    emoji, tip      = DISPOSAL_TIPS[predicted_class]

    # ── RESULTS ───────────────────────────────────────────────────────────────
    with col2:
        st.markdown("### Result")

        # Confidence threshold check
        if confidence < 0.55:
            st.warning("⚠️ Low confidence — the model is unsure about this image.")
        else:
            st.success(f"{emoji} **{predicted_class.upper()}**")

        st.metric("Confidence", f"{confidence * 100:.1f}%")
        st.info(f"**Disposal Tip:** {tip}")

    st.divider()

    # ── PROBABILITY CHART ─────────────────────────────────────────────────────
    st.markdown("### Confidence Breakdown")
    prob_data = {name: float(prob) for name, prob in zip(CLASS_NAMES, predictions)}
    st.bar_chart(prob_data)