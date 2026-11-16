
from pathlib import Path
import math
import yaml

from PIL import Image, ImageDraw, ImageFont, ImageOps


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "La_che_Thai_Nguyen.v2i.yolov8"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
    / "yolo_label_preview"
)

SPLIT = "train"

IMAGE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
]

# Mỗi lớp chọn một ảnh đại diện.
SAMPLES_PER_CLASS = 1

# Màu bounding box cho từng lớp (RGB).
CLASS_COLORS = [
    (255, 80, 80),
    (255, 165, 0),
    (80, 180, 255),
    (50, 200, 100),
    (190, 100, 255),
    (255, 220, 50),
]


# ============================================================
# 2. LOAD CLASS NAMES
# ============================================================

def load_class_names():
    yaml_path = DATASET_DIR / "data.yaml"

    with open(yaml_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    names = data["names"]

    if isinstance(names, dict):
        names = [
            names[index]
            for index in sorted(names.keys(), key=int)
        ]

    print("Class names:")
    for class_id, class_name in enumerate(names):
        print(f"  {class_id}: {class_name}")

    return names


# ============================================================
# 3. LOAD FONT
# ============================================================

def load_font(size=20):
    font_path = Path("C:/Windows/Fonts/arial.ttf")

    if font_path.exists():
        return ImageFont.truetype(str(font_path), size)

    return ImageFont.load_default()


# ============================================================
# 4. FIND IMAGE CORRESPONDING TO LABEL
# ============================================================

def build_image_lookup(images_dir):
    image_lookup = {}

    for image_path in images_dir.iterdir():
        if (
            image_path.is_file()
            and image_path.suffix.lower() in IMAGE_EXTENSIONS
        ):
            image_lookup[image_path.stem] = image_path

    return image_lookup


# ============================================================
# 5. SELECT REPRESENTATIVE IMAGES
# ============================================================

def select_samples(class_names, labels_dir, image_lookup):
    selected = {
        class_id: []
        for class_id in range(len(class_names))
    }

    label_files = sorted(labels_dir.glob("*.txt"))

    for label_path in label_files:
        with open(label_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        for line in lines:
            parts = line.strip().split()

            if len(parts) != 5:
                continue

            class_id = int(float(parts[0]))

            if class_id not in selected:
                continue

            if len(selected[class_id]) >= SAMPLES_PER_CLASS:
                continue

            image_path = image_lookup.get(label_path.stem)

            if image_path is None:
                continue

            selected[class_id].append(
                (image_path, label_path)
            )

        if all(
            len(samples) >= SAMPLES_PER_CLASS
            for samples in selected.values()
        ):
            break

    return selected


# ============================================================
# 6. DRAW YOLO BOUNDING BOXES
# ============================================================

def draw_yolo_boxes(image, label_path, class_names):
    image = image.convert("RGB")

    image_width, image_height = image.size

    draw = ImageDraw.Draw(image)

    font = load_font(20)

    with open(label_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    for line in lines:
        parts = line.strip().split()

        if len(parts) != 5:
            continue

        class_id = int(float(parts[0]))

        x_center = float(parts[1])
        y_center = float(parts[2])
        box_width = float(parts[3])
        box_height = float(parts[4])

        # Chuyển tọa độ YOLO chuẩn hóa sang pixel.
        x_center *= image_width
        y_center *= image_height

        box_width *= image_width
        box_height *= image_height

        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)
        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)

        # Giới hạn tọa độ trong ảnh.
        x1 = max(0, min(x1, image_width - 1))
        y1 = max(0, min(y1, image_height - 1))
        x2 = max(0, min(x2, image_width - 1))
        y2 = max(0, min(y2, image_height - 1))

        color = CLASS_COLORS[class_id % len(CLASS_COLORS)]

        class_name = class_names[class_id]

        # Vẽ bounding box.
        draw.rectangle(
            [x1, y1, x2, y2],
            outline=color,
            width=4,
        )

        # Vẽ tên lớp phía trên bounding box.
        text = class_name

        text_bbox = draw.textbbox(
            (x1, y1),
            text,
            font=font,
        )

        text_height = text_bbox[3] - text_bbox[1]

        text_y = max(0, y1 - text_height - 10)

        draw.rectangle(
            [
                x1,
                text_y,
                text_bbox[2] + 8,
                text_y + text_height + 8,
            ],
            fill=color,
        )

        draw.text(
            (x1 + 4, text_y + 3),
            text,
            fill=(0, 0, 0),
            font=font,
        )

    return image


# ============================================================
# 7. CREATE CONTACT SHEET
# ============================================================

def create_contact_sheet(preview_items):
    columns = 3
    tile_width = 500
    tile_height = 430

    rows = math.ceil(len(preview_items) / columns)

    sheet = Image.new(
        "RGB",
        (columns * tile_width, rows * tile_height),
        (245, 245, 245),
    )

    draw = ImageDraw.Draw(sheet)

    title_font = load_font(22)
    info_font = load_font(15)

    for index, (class_name, image_path) in enumerate(preview_items):
        row = index // columns
        column = index % columns

        x = column * tile_width
        y = row * tile_height

        image = Image.open(image_path).convert("RGB")

        thumbnail = ImageOps.contain(
            image,
            (tile_width - 30, tile_height - 75),
        )

        image_x = x + (tile_width - thumbnail.width) // 2
        image_y = y + 50

        sheet.paste(thumbnail, (image_x, image_y))

        draw.text(
            (x + 12, y + 10),
            class_name,
            fill=(20, 20, 20),
            font=title_font,
        )

        draw.text(
            (x + 12, y + tile_height - 25),
            image_path.name,
            fill=(70, 70, 70),
            font=info_font,
        )

    return sheet


# ============================================================
# 8. MAIN
# ============================================================

def main():
    print("=" * 65)
    print("TEALEAF AI - YOLO BOUNDING BOX VISUALIZATION")
    print("=" * 65)

    images_dir = DATASET_DIR / SPLIT / "images"
    labels_dir = DATASET_DIR / SPLIT / "labels"

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET_DIR}"
        )

    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(
            "Images or labels directory does not exist."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    class_names = load_class_names()

    image_lookup = build_image_lookup(images_dir)

    selected = select_samples(
        class_names,
        labels_dir,
        image_lookup,
    )

    preview_items = []

    print("\nGenerating previews...\n")

    for class_id, samples in selected.items():
        class_name = class_names[class_id]

        if not samples:
            print(f"[WARNING] No sample found for {class_name}")
            continue

        for sample_index, (image_path, label_path) in enumerate(samples):
            image = Image.open(image_path)

            preview = draw_yolo_boxes(
                image,
                label_path,
                class_names,
            )

            output_name = (
                f"{class_id}_{class_name}_{sample_index + 1}.jpg"
            )

            output_path = OUTPUT_DIR / output_name

            preview.save(
                output_path,
                quality=95,
            )

            preview_items.append(
                (class_name, output_path)
            )

            print(f"[OK] {class_name}")
            print(f"     Image: {image_path.name}")
            print(f"     Label: {label_path.name}")
            print(f"     Saved: {output_path}")

    if not preview_items:
        raise RuntimeError(
            "No preview images were generated."
        )

    # Tạo ảnh tổng hợp.
    contact_sheet = create_contact_sheet(preview_items)

    contact_path = OUTPUT_DIR / "all_classes_preview.jpg"

    contact_sheet.save(
        contact_path,
        quality=95,
    )

    print("\n" + "=" * 65)
    print("VISUALIZATION COMPLETED")
    print("=" * 65)

    print(f"Number of previews: {len(preview_items)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Contact sheet: {contact_path}")


if __name__ == "__main__":
    main()
