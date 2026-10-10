import json
import numpy as np
import tensorflow as tf
import streamlit as st

from PIL import Image

from src.config import MODEL_DIR, IMAGE_SIZE


# ==========================================
# PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌱",
    layout="centered"
)


# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        MODEL_DIR / "plant_disease_model.h5"
    )


model = load_model()


# ==========================================
# LOAD CLASS NAMES
# ==========================================

with open(
    MODEL_DIR / "class_names.json",
    "r"
) as file:

    class_names = json.load(file)


# ==========================================
# TITLE
# ==========================================

st.title("🌱 Plant Disease Detection")

st.write(
    "Upload an image of a tomato leaf "
    "to identify its condition."
)


# ==========================================
# IMAGE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload a tomato leaf image",
    type=["jpg", "jpeg", "png"]
)


# ==========================================
# PREDICTION
# ==========================================

if uploaded_file is not None:

    # Open image
    img = Image.open(
        uploaded_file
    ).convert("RGB")


    # Display uploaded image
    st.image(
        img,
        caption="Uploaded Leaf",
        width=400
    )


    # Resize image
    resized_img = img.resize(
        IMAGE_SIZE
    )


    # Convert to NumPy array
    img_array = np.array(
        resized_img
    )


    # Normalize pixel values
    img_array = img_array / 255.0


    # Add batch dimension
    img_array = np.expand_dims(
        img_array,
        axis=0
    )


    # Make prediction
    prediction = model.predict(
        img_array,
        verbose=0
    )


    # Find predicted class
    predicted_index = np.argmax(
        prediction[0]
    )


    predicted_class = (
        class_names[predicted_index]
    )


    # Calculate confidence
    confidence = (
        prediction[0][predicted_index]
        * 100
    )


    # ======================================
    # DISPLAY RESULT
    # ======================================

    st.success(
        f"Prediction: {predicted_class}"
    )

    st.info(
        f"Confidence: {confidence:.2f}%"
    )


    # ======================================
    # SHOW ALL CLASS PROBABILITIES
    # ======================================

    st.subheader("Prediction Probabilities")

    for i, class_name in enumerate(
        class_names
    ):

        probability = (
            prediction[0][i] * 100
        )

        st.write(
            f"{class_name}: "
            f"{probability:.2f}%"
        )

        st.progress(
            float(prediction[0][i])
        )


# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.caption(
    "CNN-based Plant Disease Detection "
    "| Trained on PlantVillage dataset"
)