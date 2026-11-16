
from pathlib import Path
from PIL import Image


SUPPORTED_FORMATS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


def get_image_files(directory):
    """
    Return all supported image files
    inside a directory and its subdirectories.
    """
    directory = Path(directory)

    if not directory.exists():
        return []

    return [
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_FORMATS
    ]


def inspect_image(image_path):
    """
    Inspect image properties and detect corrupted images.
    """
    image_path = Path(image_path)

    try:
        with Image.open(image_path) as image:
            return {
                "path": str(image_path),
                "format": image.format,
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
                "valid": True
            }

    except Exception as error:
        return {
            "path": str(image_path),
            "valid": False,
            "error": str(error)
        }


if __name__ == "__main__":
    print("TeaLeaf AI - Image Utilities")
    print("Supported formats:", SUPPORTED_FORMATS)
