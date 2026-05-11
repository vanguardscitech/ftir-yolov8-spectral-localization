from pathlib import Path
import random
import shutil

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "FTIR_Dataset_Base"
DST = ROOT / "data" / "FTIR_Dataset_Augmented_1000"
MULTIPLIER = 5
SEED = 42


random.seed(SEED)
np.random.seed(SEED)


def add_gaussian_noise(img, sigma_range=(3, 15)):
    sigma = random.uniform(*sigma_range)
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def adjust_brightness_contrast(img, alpha_range=(0.85, 1.15), beta_range=(-18, 18)):
    alpha = random.uniform(*alpha_range)
    beta = random.uniform(*beta_range)
    return np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)


def apply_gaussian_blur(img):
    kernel = random.choice([3, 5])
    return cv2.GaussianBlur(img, (kernel, kernel), 0)


def simulate_baseline_drift(img, max_amplitude=8):
    h, w = img.shape[:2]
    amplitude = random.uniform(0, max_amplitude)
    freq = random.uniform(0.5, 2.0)
    phase = random.uniform(0, 2 * np.pi)
    drift = amplitude * np.sin(2 * np.pi * freq * np.arange(h) / h + phase)
    drift_map = np.tile(drift[:, None], (1, w)).astype(np.float32)
    if img.ndim == 3:
        drift_map = drift_map[:, :, None]
    return np.clip(img.astype(np.float32) + drift_map, 0, 255).astype(np.uint8)


def augment_image(img):
    aug = add_gaussian_noise(img)
    aug = adjust_brightness_contrast(aug)
    if random.random() < 0.6:
        aug = apply_gaussian_blur(aug)
    if random.random() < 0.7:
        aug = simulate_baseline_drift(aug)
    return aug


def process_split(split):
    src_img_dir = SRC / split / "images"
    src_lbl_dir = SRC / split / "labels"
    dst_img_dir = DST / split / "images"
    dst_lbl_dir = DST / split / "labels"
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(src_img_dir.glob("*.png"))
    written = 0

    for image_path in image_paths:
        img = cv2.imread(str(image_path))
        if img is None:
            raise RuntimeError(f"Could not read {image_path}")

        label_path = src_lbl_dir / f"{image_path.stem}.txt"
        for version in range(MULTIPLIER):
            tag = "orig" if version == 0 else f"aug{version}"
            out_img = img.copy() if version == 0 else augment_image(img)
            out_img_path = dst_img_dir / f"{image_path.stem}_{tag}.png"
            out_lbl_path = dst_lbl_dir / f"{image_path.stem}_{tag}.txt"
            cv2.imwrite(str(out_img_path), out_img)
            if label_path.exists():
                shutil.copy2(label_path, out_lbl_path)
            else:
                out_lbl_path.write_text("", encoding="utf-8")
            written += 1

    return written


def main():
    counts = {split: process_split(split) for split in ("train", "valid")}
    print(f"train={counts['train']}")
    print(f"valid={counts['valid']}")
    print(f"total={sum(counts.values())}")
    print(f"output={DST}")


if __name__ == "__main__":
    main()
