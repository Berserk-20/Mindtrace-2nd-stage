"""
setup_ckplus.py — CK+ Dataset Preparation Script
==================================================
Run this AFTER downloading and extracting the CK+ dataset from Kaggle.

Kaggle source : https://www.kaggle.com/datasets/shawon10/ckplus

The Kaggle version comes as a flat folder per emotion (no train/test split).
This script:
  1. Auto-detects the downloaded folder structure
  2. Normalises class names to match RAF-DB / FER2013
  3. Drops 'contempt' if present (RAF-DB & FER2013 don't have it)
  4. Does a stratified 80/20 train/test split
  5. Copies files into backend/dataset_ckplus/train/ and /test/

Usage
-----
  # After extracting the Kaggle zip, run:
  python setup_ckplus.py --source <path-to-extracted-ckplus-folder>

  # Example:
  python setup_ckplus.py --source C:/Users/sanka/Downloads/CK+48

  # Custom output dir:
  python setup_ckplus.py --source C:/Users/sanka/Downloads/CK+48 --output dataset_ckplus
"""

import os
import shutil
import random
import argparse


# ──────────────────────────────────────────────────────────────────────
# CLASS NAME NORMALISATION
# Maps various naming conventions found in CK+ downloads → standard name
# ──────────────────────────────────────────────────────────────────────
NAME_MAP = {
    "anger":    "angry",
    "angry":    "angry",
    "disgust":  "disgust",
    "fear":     "fear",
    "happy":    "happy",
    "happiness":"happy",
    "neutral":  "neutral",
    "sadness":  "sad",
    "sad":      "sad",
    "surprise": "surprise",
    # 'contempt' is intentionally excluded (not in RAF-DB / FER2013)
}

TARGET_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


# ──────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────
def find_class_folders(source_dir: str) -> dict:
    """
    Recursively walk source_dir to find emotion class folders.
    Handles arbitrarily nested zip extractions.
    Returns {normalised_class_name: [list_of_image_paths]}.
    """
    class_images = {}

    # Walk ALL subdirectories looking for folders whose names match emotions
    for root, dirs, files in os.walk(source_dir):
        folder_name = os.path.basename(root).lower().strip()
        norm_name   = NAME_MAP.get(folder_name)

        if norm_name is None:
            continue  # not an emotion folder, keep walking

        images = [
            os.path.join(root, f)
            for f in files
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]

        if not images:
            continue  # emotion-named folder but empty, keep walking

        if norm_name not in class_images:
            class_images[norm_name] = []
        class_images[norm_name].extend(images)

    return class_images


def stratified_split(images: list, test_ratio: float = 0.2, seed: int = 42):
    """Returns (train_images, test_images) with a fixed random seed."""
    random.seed(seed)
    shuffled = images.copy()
    random.shuffle(shuffled)
    split_at = max(1, int(len(shuffled) * (1 - test_ratio)))
    return shuffled[:split_at], shuffled[split_at:]


def copy_files(file_paths: list, dest_dir: str):
    """Copy a list of files into dest_dir, preserving original filenames."""
    os.makedirs(dest_dir, exist_ok=True)
    for src in file_paths:
        fname = os.path.basename(src)
        dst   = os.path.join(dest_dir, fname)
        # Avoid name collisions by prefixing with parent folder name
        if os.path.exists(dst):
            parent = os.path.basename(os.path.dirname(src))
            fname  = f"{parent}_{fname}"
            dst    = os.path.join(dest_dir, fname)
        shutil.copy2(src, dst)


# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="CK+ dataset train/test split preparation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source", required=True,
        help="Path to the extracted CK+ folder (e.g. C:/Downloads/CK+48)",
    )
    parser.add_argument(
        "--output", default="dataset_ckplus",
        help="Output directory (relative to backend/)",
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.2,
        help="Fraction of images to use for test split",
    )
    args = parser.parse_args()

    source = os.path.abspath(args.source)
    output = os.path.abspath(args.output)

    if not os.path.isdir(source):
        print(f"\n  ✗  Source folder not found: {source}")
        print("  Please check the path and try again.\n")
        return

    print(f"\n{'='*60}")
    print(f"  CK+ Dataset Setup")
    print(f"  Source : {source}")
    print(f"  Output : {output}")
    print(f"  Split  : {int((1-args.test_ratio)*100)}% train / "
          f"{int(args.test_ratio*100)}% test")
    print(f"{'='*60}\n")

    # ── Find emotion folders ───────────────────────────────────────
    class_images = find_class_folders(source)

    if not class_images:
        print("  ✗  No recognisable emotion folders found in source directory.")
        print("  Check that the extracted path is correct.\n")
        return

    # ── Report what was found ──────────────────────────────────────
    print("  Found classes:")
    total_images = 0
    for cls in sorted(class_images):
        n = len(class_images[cls])
        total_images += n
        print(f"    {cls:<12}: {n:>4} images")
    print(f"    {'TOTAL':<12}: {total_images:>4} images\n")

    # ── Check for missing classes ──────────────────────────────────
    missing = [c for c in TARGET_CLASSES if c not in class_images]
    if missing:
        print(f"  ⚠  Missing classes (will be skipped): {missing}\n")

    # ── Split and copy ─────────────────────────────────────────────
    print("  Splitting and copying files...")
    train_total, test_total = 0, 0

    for cls, images in class_images.items():
        train_imgs, test_imgs = stratified_split(images, args.test_ratio)

        copy_files(train_imgs, os.path.join(output, "train", cls))
        copy_files(test_imgs,  os.path.join(output, "test",  cls))

        print(f"    {cls:<12}: {len(train_imgs):>4} train | {len(test_imgs):>4} test")
        train_total += len(train_imgs)
        test_total  += len(test_imgs)

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  ✓  Done!")
    print(f"  Train : {train_total} images → {output}/train/")
    print(f"  Test  : {test_total}  images → {output}/test/")
    print(f"\n  You can now train with:")
    print(f"    python train_unified.py --model resnet18 --dataset ckplus")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
