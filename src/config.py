from pathlib import Path

# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

# PlantVillage dataset
# The class folders are inside:
# raw_data/PlantVillage/

RAW_DATA_DIR = BASE_DIR / "raw_data" / "PlantVillage"

DATASET_DIR = BASE_DIR / "dataset"

TRAIN_DIR = DATASET_DIR / "train"
VALIDATION_DIR = DATASET_DIR / "validation"
TEST_DIR = DATASET_DIR / "test"

MODEL_DIR = BASE_DIR / "model"
RESULTS_DIR = BASE_DIR / "results"

TEST_IMAGES_DIR = BASE_DIR / "test_images"


# ==========================================
# DATA SETTINGS
# ==========================================

IMAGE_SIZE = (128, 128)

BATCH_SIZE = 32

RANDOM_SEED = 42


# ==========================================
# DATA SPLIT
# ==========================================

TRAIN_RATIO = 0.70

VALIDATION_RATIO = 0.15

TEST_RATIO = 0.15


# ==========================================
# MODEL SETTINGS
# ==========================================

EPOCHS = 10

LEARNING_RATE = 0.001


# ==========================================
# SELECTED PLANT DISEASE CLASSES
# ==========================================

SELECTED_CLASSES = [
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_healthy"
]