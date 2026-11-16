
from pathlib import Path

DATA_DIR = Path("data/raw")


def inspect_dataset():
    print("=" * 60)
    print("TEALEAF AI - DATASET INSPECTION")
    print("=" * 60)

    if not DATA_DIR.exists():
        print("Dataset directory does not exist.")
        return

    classes = [
        item for item in DATA_DIR.iterdir()
        if item.is_dir()
    ]

    if not classes:
        print("No class folders found yet.")
        return

    total_images = 0
    extensions = {
        ".jpg", ".jpeg", ".png", ".bmp", ".webp"
    }

    for class_dir in sorted(classes):
        images = [
            file for file in class_dir.rglob("*")
            if file.is_file()
            and file.suffix.lower() in extensions
        ]

        count = len(images)
        total_images += count

        print(f"{class_dir.name}: {count} images")

    print("-" * 60)
    print(f"Number of classes: {len(classes)}")
    print(f"Total images: {total_images}")
    print("=" * 60)


if __name__ == "__main__":
    inspect_dataset()
