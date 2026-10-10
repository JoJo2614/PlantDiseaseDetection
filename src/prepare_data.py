import shutil
import random

from config import (
    RAW_DATA_DIR,
    TRAIN_DIR,
    VALIDATION_DIR,
    TEST_DIR,
    SELECTED_CLASSES,
    TRAIN_RATIO,
    VALIDATION_RATIO,
    RANDOM_SEED
)


random.seed(RANDOM_SEED)


def create_directories():
    """Create train, validation and test directories."""

    for directory in [
        TRAIN_DIR,
        VALIDATION_DIR,
        TEST_DIR
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def split_class(class_name):
    """Split one class into train, validation and test."""

    source_dir = RAW_DATA_DIR / class_name

    if not source_dir.exists():
        print(f"ERROR: Could not find: {source_dir}")
        return

    images = [
        file
        for file in source_dir.iterdir()
        if file.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)

    validation_end = (
        train_end +
        int(total * VALIDATION_RATIO)
    )

    train_images = images[:train_end]

    validation_images = images[
        train_end:validation_end
    ]

    test_images = images[
        validation_end:
    ]

    destinations = {
        TRAIN_DIR: train_images,
        VALIDATION_DIR: validation_images,
        TEST_DIR: test_images
    }

    for destination, files in destinations.items():

        class_directory = destination / class_name

        class_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        for image in files:

            shutil.copy2(
                image,
                class_directory / image.name
            )

    print(
        f"{class_name}: "
        f"{len(train_images)} train | "
        f"{len(validation_images)} validation | "
        f"{len(test_images)} test"
    )


def main():

    print("================================")
    print("   PREPARING PLANT DATASET")
    print("================================\n")

    create_directories()

    for class_name in SELECTED_CLASSES:
        split_class(class_name)

    print("\nDataset preparation completed!")


if __name__ == "__main__":
    main()