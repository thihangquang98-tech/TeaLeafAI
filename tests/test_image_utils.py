
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.image_utils import (
    get_image_files,
    inspect_image
)


def test_image_utils():
    test_directory = PROJECT_ROOT / "data" / "raw"

    files = get_image_files(test_directory)

    print("Image utilities test")
    print("Found images:", len(files))

    if files:
        result = inspect_image(files[0])
        print("First image:", result)

    print("Test completed successfully.")


if __name__ == "__main__":
    test_image_utils()
