
from pathlib import Path
from collections import Counter
import yaml

DATASET_DIR = Path(
    r"C:\Users\FPT SHOP\Downloads\La_che_Thai_Nguyen.v2i.yolov8"
)

SPLITS = ["train", "valid", "test"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def validate_dataset():
    yaml_path = DATASET_DIR / "data.yaml"

    if not yaml_path.exists():
        print("ERROR: data.yaml not found.")
        return

    with open(yaml_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    num_classes = config["nc"]
    class_names = config["names"]

    print("=" * 65)
    print("TEALEAF AI - YOLO DATASET VALIDATION")
    print("=" * 65)

    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")

    for split in SPLITS:
        image_dir = DATASET_DIR / split / "images"
        label_dir = DATASET_DIR / split / "labels"

        if not image_dir.exists() or not label_dir.exists():
            print(f"\nERROR: Missing directories in {split}")
            continue

        images = {
            path.stem: path
            for path in image_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        }

        labels = {
            path.stem: path
            for path in label_dir.glob("*.txt")
            if path.is_file()
        }

        missing_labels = sorted(set(images) - set(labels))
        missing_images = sorted(set(labels) - set(images))

        class_counts = Counter()
        empty_files = []
        invalid_lines = []
        invalid_coordinates = []
        total_boxes = 0

        for label_path in labels.values():
            lines = label_path.read_text(
                encoding="utf-8"
            ).splitlines()

            if not any(line.strip() for line in lines):
                empty_files.append(label_path.name)
                continue

            for line_number, line in enumerate(lines, start=1):
                if not line.strip():
                    continue

                parts = line.split()

                if len(parts) != 5:
                    invalid_lines.append(
                        f"{label_path.name}:{line_number}"
                    )
                    continue

                try:
                    class_id = int(parts[0])
                    x, y, width, height = map(float, parts[1:])
                except ValueError:
                    invalid_lines.append(
                        f"{label_path.name}:{line_number}"
                    )
                    continue

                if not 0 <= class_id < num_classes:
                    invalid_lines.append(
                        f"{label_path.name}:{line_number} "
                        f"(invalid class ID {class_id})"
                    )
                    continue

                class_counts[class_id] += 1
                total_boxes += 1

                valid_coordinates = (
                    0 <= x <= 1
                    and 0 <= y <= 1
                    and 0 < width <= 1
                    and 0 < height <= 1
                    and x - width / 2 >= 0
                    and x + width / 2 <= 1
                    and y - height / 2 >= 0
                    and y + height / 2 <= 1
                )

                if not valid_coordinates:
                    invalid_coordinates.append(
                        f"{label_path.name}:{line_number}"
                    )

        print(f"\n{'-' * 65}")
        print(f"SPLIT: {split.upper()}")
        print(f"Images: {len(images)}")
        print(f"Label files: {len(labels)}")
        print(f"Total bounding boxes: {total_boxes}")

        print("\nClass distribution:")
        for class_id in range(num_classes):
            print(
                f"  {class_id} - {class_names[class_id]}: "
                f"{class_counts[class_id]}"
            )

        print("\nLabel quality:")
        print(f"  Images without labels: {len(missing_labels)}")
        print(f"  Labels without images: {len(missing_images)}")
        print(f"  Empty label files: {len(empty_files)}")
        print(f"  Invalid lines: {len(invalid_lines)}")
        print(
            f"  Invalid bounding boxes: "
            f"{len(invalid_coordinates)}"
        )

        if missing_labels:
            print("  Examples missing labels:", missing_labels[:5])

        if missing_images:
            print("  Examples missing images:", missing_images[:5])

        if invalid_lines:
            print("  Examples invalid lines:", invalid_lines[:5])

        if invalid_coordinates:
            print(
                "  Examples invalid boxes:",
                invalid_coordinates[:5]
            )

    print("\n" + "=" * 65)
    print("VALIDATION COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    validate_dataset()
