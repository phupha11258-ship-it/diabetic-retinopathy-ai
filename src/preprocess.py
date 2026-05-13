from pathlib import Path
import cv2
import numpy as np
import pandas as pd


def crop_black_border(image, tolerance=7):
    """
    Crop black border around fundus image.
    image: BGR image from OpenCV
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray > tolerance

    if not mask.any():
        return image

    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    return image[y0:y1, x0:x1]


def ben_graham_preprocess(image, sigma=10):
    """
    Local Average Subtraction / Ben Graham Method.
    Formula idea:
    processed = 4 * image - 4 * gaussian_blur(image) + 128
    """
    image = crop_black_border(image)

    # Resize before enhancement for stable processing
    image = cv2.resize(image, (224, 224))

    gaussian = cv2.GaussianBlur(image, (0, 0), sigma)
    processed = cv2.addWeighted(image, 4, gaussian, -4, 128)

    return processed


def preprocess_image_path(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    processed = ben_graham_preprocess(image)
    return processed


def create_preview(split_csv="data/splits/train.csv", out_dir="outputs/preprocess_preview", n=10):
    split_csv = Path(split_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(split_csv).head(n)

    for i, row in df.iterrows():
        image_path = Path(row["image_path"])
        raw = cv2.imread(str(image_path))

        if raw is None:
            print(f"Skip unreadable image: {image_path}")
            continue

        raw_resized = cv2.resize(crop_black_border(raw), (224, 224))
        processed = ben_graham_preprocess(raw)

        combined = np.hstack([raw_resized, processed])

        out_path = out_dir / f"preview_{i}_label_{row['diagnosis']}.png"
        cv2.imwrite(str(out_path), combined)

        print(f"Saved: {out_path}")


if __name__ == "__main__":
    create_preview()