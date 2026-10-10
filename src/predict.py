import sys
import json

import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing import image

from config import (
    MODEL_DIR,
    IMAGE_SIZE
)


# ==========================================
# CHECK IMAGE PATH
# ==========================================

if len(sys.argv) < 2:

    print("\nUsage:")

    print(
        "python src/predict.py "
        "test_images/leaf.jpg"
    )

    sys.exit()


IMAGE_PATH = sys.argv[1]


# ==========================================
# LOAD MODEL
# ==========================================

model = tf.keras.models.load_model(
    MODEL_DIR /
    "plant_disease_model.h5"
)


# ==========================================
# LOAD CLASS NAMES
# ==========================================

with open(
    MODEL_DIR /
    "class_names.json",
    "r"
) as file:

    class_names = json.load(file)


# ==========================================
# LOAD IMAGE
# ==========================================

img = image.load_img(
    IMAGE_PATH,
    target_size=IMAGE_SIZE
)


# Convert image to array

img_array = image.img_to_array(
    img
)


# Normalize pixels

img_array = img_array / 255.0


# Add batch dimension

img_array = np.expand_dims(
    img_array,
    axis=0
)


# ==========================================
# MAKE PREDICTION
# ==========================================

prediction = model.predict(
    img_array,
    verbose=0
)


predicted_index = np.argmax(
    prediction[0]
)


predicted_class = (
    class_names[predicted_index]
)


confidence = (
    prediction[0][predicted_index]
    * 100
)


# ==========================================
# DISPLAY RESULT
# ==========================================

print("\n================================")
print("     PLANT DISEASE DETECTION")
print("================================")

print(
    f"Prediction: {predicted_class}"
)

print(
    f"Confidence: {confidence:.2f}%"
)

print("================================\n")