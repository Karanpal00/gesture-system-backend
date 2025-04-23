import cv2
import numpy as np
from pathlib import Path

def augment_image(input_path: str, output_dir: str) -> None:
    """
    Given a single image file, produce several augmented variants
    (flip, rotate, brightness, blur) and save them alongside the original.
    Raises descriptive exceptions on failure.
    """
    input_p = Path(input_path)
    out_dir = Path(output_dir)

    # 1️⃣ Validate input image
    if not input_p.exists():
        raise FileNotFoundError(f"[augment] Input image not found: {input_p}")
    if not input_p.is_file():
        raise ValueError(f"[augment] Input path is not a file: {input_p}")

    # 2️⃣ Ensure output directory exists
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise IOError(f"[augment] Could not create output directory {out_dir}: {e}")

    # 3️⃣ Load image
    img = cv2.imread(str(input_p))
    if img is None:
        raise ValueError(f"[augment] Failed to read image (cv2 returned None): {input_p}")

    base = input_p.stem

    # 4️⃣ Perform augmentations
    try:
        # Horizontal flip
        flip = cv2.flip(img, 1)
        cv2.imwrite(str(out_dir / f"{base}_flip.jpg"), flip)

        # Rotations
        rows, cols = img.shape[:2]
        for angle in (-15, 15):
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            rotated = cv2.warpAffine(img, M, (cols, rows))
            cv2.imwrite(str(out_dir / f"{base}_rot{angle}.jpg"), rotated)

        # Brightness / contrast adjustment
        bright = cv2.convertScaleAbs(img, alpha=1.1, beta=30)
        cv2.imwrite(str(out_dir / f"{base}_bright.jpg"), bright)

        # Gaussian blur
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        cv2.imwrite(str(out_dir / f"{base}_blur.jpg"), blur)

    except Exception as e:
        raise RuntimeError(f"[augment] Error during augmentation of {input_p.name}: {e}")
