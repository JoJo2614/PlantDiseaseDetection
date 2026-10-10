import json

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, models

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

from config import (
    TRAIN_DIR,
    VALIDATION_DIR,
    TEST_DIR,
    MODEL_DIR,
    RESULTS_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    EPOCHS
)


# ==========================================
# CREATE OUTPUT DIRECTORIES
# ==========================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================
# LOAD DATASET
# ==========================================

print("Loading training data...")

train_data = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)


print("Loading validation data...")

validation_data = tf.keras.utils.image_dataset_from_directory(
    VALIDATION_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


print("Loading test data...")

test_data = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ==========================================
# CLASS NAMES
# ==========================================

class_names = train_data.class_names

print("\nClasses:")

for i, name in enumerate(class_names):
    print(f"{i}: {name}")


# Save class names

with open(
    MODEL_DIR / "class_names.json",
    "w"
) as file:

    json.dump(
        class_names,
        file,
        indent=4
    )


# ==========================================
# NORMALIZE IMAGES
# ==========================================

normalization_layer = layers.Rescaling(
    1.0 / 255
)


train_data = train_data.map(
    lambda x, y:
    (normalization_layer(x), y)
)


validation_data = validation_data.map(
    lambda x, y:
    (normalization_layer(x), y)
)


test_data = test_data.map(
    lambda x, y:
    (normalization_layer(x), y)
)


# ==========================================
# CNN MODEL
# ==========================================

model = models.Sequential([

    layers.Input(
        shape=(
            IMAGE_SIZE[0],
            IMAGE_SIZE[1],
            3
        )
    ),

    # First convolution block
    layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),


    # Second convolution block
    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),


    # Third convolution block
    layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),


    # Classification layers
    layers.Flatten(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(0.5),

    layers.Dense(
        len(class_names),
        activation="softmax"
    )
])


# ==========================================
# COMPILE MODEL
# ==========================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)


# ==========================================
# SHOW MODEL
# ==========================================

print("\nCNN Model:")

model.summary()


# ==========================================
# TRAIN MODEL
# ==========================================

print("\n================================")
print("       STARTING TRAINING")
print("================================\n")


history = model.fit(

    train_data,

    validation_data=validation_data,

    epochs=EPOCHS
)


# ==========================================
# SAVE MODEL
# ==========================================

model_path = (
    MODEL_DIR /
    "plant_disease_model.h5"
)

model.save(model_path)

print(
    f"\nModel saved to: {model_path}"
)


# ==========================================
# TEST MODEL
# ==========================================

print("\nEvaluating model...")

test_loss, test_accuracy = model.evaluate(
    test_data
)


print(
    f"\nTest Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Test Loss: "
    f"{test_loss:.4f}"
)


# ==========================================
# ACCURACY GRAPH
# ==========================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title(
    "Training and Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "accuracy.png"
)

plt.close()


# ==========================================
# LOSS GRAPH
# ==========================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title(
    "Training and Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "loss.png"
)

plt.close()


# ==========================================
# PREDICTIONS
# ==========================================

true_labels = []

predicted_labels = []


for images, labels in test_data:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted = np.argmax(
        predictions,
        axis=1
    )

    true_labels.extend(
        labels.numpy()
    )

    predicted_labels.extend(
        predicted
    )


# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    true_labels,
    predicted_labels
)


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)


fig, ax = plt.subplots(
    figsize=(8, 8)
)


disp.plot(
    ax=ax,
    xticks_rotation=45
)


plt.title(
    "Plant Disease Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR /
    "confusion_matrix.png"
)

plt.close()


# ==========================================
# CLASSIFICATION REPORT
# ==========================================

report = classification_report(
    true_labels,
    predicted_labels,
    target_names=class_names
)


with open(
    RESULTS_DIR /
    "classification_report.txt",
    "w"
) as file:

    file.write(report)


# ==========================================
# FINISHED
# ==========================================

print("\n================================")
print("   TRAINING COMPLETED!")
print("================================")

print(
    "\nResults saved in the results folder."
)