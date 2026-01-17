#!/usr/bin/env python3
"""
Prepare Mercedes-Benz and Volkswagen training data for Kohya LoRA training.
Creates separate training folders for each car model with images and captions.
"""

import os
import shutil
from pathlib import Path

# Configuration
COMBINED_DIR = Path("./combined")
ENHANCED_DIR = Path("./processed/run_20260111_045656/02_enhanced")
OUTPUT_BASE = Path("./lora_training/mercedes_vw_test")

# Models to train
MODELS = [
    # Mercedes-Benz models
    "Mercedes-Benz 300-Class Convertible 1993",
    "Mercedes-Benz C-Class Sedan 2012",
    "Mercedes-Benz E-Class Sedan 2012",
    "Mercedes-Benz S-Class Sedan 2012",
    "Mercedes-Benz SL-Class Coupe 2009",
    "Mercedes-Benz Sprinter Van 2012",
    # Volkswagen models
    "Volkswagen Beetle Hatchback 2012",
    "Volkswagen Golf Hatchback 1991",
    "Volkswagen Golf Hatchback 2012",
]

def clean_model_name(model_name):
    """Convert model name to safe folder name."""
    # Remove spaces and special characters
    safe_name = model_name.replace(" ", "_").replace("-", "")
    return safe_name.lower()

def prepare_training_data():
    """Organize training data for Kohya LoRA training."""

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    for model in MODELS:
        print(f"\nPreparing: {model}")

        # Create model-specific folder
        model_folder = OUTPUT_BASE / clean_model_name(model)
        model_folder.mkdir(exist_ok=True)

        # Source directories
        image_source = COMBINED_DIR / model

        if not image_source.exists():
            print(f"  WARNING: Source not found: {image_source}")
            continue

        # Find all images in source
        image_files = list(image_source.glob("*.jpg")) + list(image_source.glob("*.png"))
        print(f"  Found {len(image_files)} images")

        copied = 0
        for img_file in image_files:
            # Get base filename without extension
            base_name = img_file.stem

            # Find corresponding caption in enhanced directory
            # Caption files are named like: Mercedes_Benz_C_Class_Sedan_2012_001234.txt
            caption_pattern = f"{base_name}.txt"

            # Search for caption file
            caption_files = list(ENHANCED_DIR.glob(f"*{base_name}*.txt"))

            if not caption_files:
                print(f"  WARNING: No caption found for {img_file.name}")
                continue

            caption_file = caption_files[0]

            # Copy image to training folder
            dest_img = model_folder / img_file.name
            shutil.copy2(img_file, dest_img)

            # Copy caption with same base name
            dest_caption = model_folder / f"{img_file.stem}.txt"
            shutil.copy2(caption_file, dest_caption)

            copied += 1

        print(f"  Copied {copied} image/caption pairs to {model_folder}")

if __name__ == "__main__":
    print("=" * 60)
    print("Preparing Mercedes-Benz and Volkswagen LoRA Training Data")
    print("=" * 60)
    prepare_training_data()
    print("\n" + "=" * 60)
    print("Training data preparation complete!")
    print(f"Output directory: {OUTPUT_BASE.absolute()}")
    print("=" * 60)
