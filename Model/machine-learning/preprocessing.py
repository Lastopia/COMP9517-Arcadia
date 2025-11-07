import cv2
import numpy as np
import os, re
import matplotlib.pyplot as plt
import numpy as np

"""
Input (RGB image)
    ↓
Crop to proper size(Read labels)
    ↓
Color conversion (RGB → HSV)
    ↓
Contrast enhancement (CLAHE on V channel)
    ↓
Noise reduction (Gaussian blur σ=1)
    ↓
Optional: Unsharp mask (α≈0.5)
    ↓
Output (class_id, Enhanced RGB/HSV image)
"""

def convert_to_hsv(images):
    hsv = cv2.cvtColor(images, cv2.COLOR_RGB2HSV)
    return hsv

def apply_clahe_on_v(hsv_img, clip_limit = 2.0, tile_grid_size = (8, 8)):

    h, s, v = cv2.split(hsv_img)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    v_eq = clahe.apply(v)
    hsv_eq = cv2.merge((h, s, v_eq))
    return hsv_eq


def gaussian_blur(images, sigma = 1.0, ksize = 5) -> np.ndarray:

    blurred = cv2.GaussianBlur(images, (ksize, ksize), sigma)
    return blurred


def unsharp_mask(image, sigma = 1.0, alpha = 0.5, ksize = 5):

    blurred = cv2.GaussianBlur(image, (ksize, ksize), sigma)
    sharpened = cv2.addWeighted(image, 1 + alpha, blurred, -alpha, 0)
    return sharpened


def preprocess(image, apply_sharpen = True, output_rgb=False):

    hsv = convert_to_hsv(image)
    # hsv_smooth = apply_clahe_on_v(hsv)
    hsv_smooth = cv2.GaussianBlur(hsv, (3, 3), 0.3)

    # if apply_sharpen:
    #     hsv_smooth = unsharp_mask(hsv_smooth, sigma=1.0, alpha=0.7)

    if output_rgb:
        return cv2.cvtColor(hsv_smooth, cv2.COLOR_HSV2RGB)

    return hsv_smooth

"""
Arguments:
    <img_path>: str             -> Directory path containing input `.jpg` images.
    <label_path> : str          -> Directory path containing YOLO-format label text files.
    <apply_preprocess> : bool   -> optional, default=True If True, apply the `preprocess()` function to each cropped image.

Returns:
    list of tuple of (int, numpy.ndarray)
        A list of tuples where each tuple contains:
        - <class_id> : int              -> Object class label.
        - <crop_img> : numpy.ndarray    -> Cropped image region of the detected object.
"""
def read_images(img_path, label_path, apply_preprocess = True, apply_sharpen = True, output_rgb=False):
    images = []
    for filename in os.listdir(img_path):

        if not filename.lower().endswith(".jpg"):
            continue

        img = cv2.imread(f"{img_path}/{filename}", cv2.IMREAD_COLOR)
        h, w, c = img.shape

        if img is None:
            continue

        label_name = os.path.splitext(filename)[0]

        with open(f"{label_path}/{label_name}.txt", "r") as f:
            for lines in f:
                elements = lines.strip().split()
                if len(elements) != 5:
                    continue

                class_id, x_center, y_center, bond_w, bond_h = map(float, elements)

                x1 = max(int(x_center * w - (bond_w * w) / 2), 0)
                y1 = max(int(y_center * h - (bond_h * h) / 2), 0)
                x2 = min(int(x_center * w + (bond_w * w) / 2), w)
                y2 = min(int(y_center * h + (bond_h * h) / 2), h)

                crop_img = img[y1:y2, x1:x2]
                if apply_preprocess:
                    crop_img = preprocess(crop_img, apply_sharpen, output_rgb)

                images.append((int(class_id), crop_img))

    return images


if __name__ == "__main__":
    image_path = "../../Data/test/images"
    label_path = "../../Data/test/labels"
    processed_data = read_images(image_path, label_path, True)
    print(f"Processed {len(processed_data)} cropped and preprocessed images.")
