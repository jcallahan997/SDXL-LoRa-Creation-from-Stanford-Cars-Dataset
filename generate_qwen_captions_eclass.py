#!/usr/bin/env python3
"""
Enhanced caption generator using Qwen2-VL via Ollama for Mercedes E-Class dataset.
Generates detailed, accurate captions for LoRA training with correct colors and positioning.

⚠️  IMPORTANT: Update DATASET_DIR below to match your local setup!
"""

import os
import subprocess
import json
import base64
from pathlib import Path
import time

# ⚠️  UPDATE THIS PATH: Location of your E-Class training images
# Example: "/Users/YOUR_USERNAME/path/to/lora_training_kohya/mercedesbenz_eclass/10_mercedesbenzeclasssedan2012"
DATASET_DIR = "/path/to/lora_training_kohya/mercedesbenz_eclass/10_mercedesbenzeclasssedan2012"
TRIGGER_WORD = "mercedesbenzeclasssedan2012"

# Qwen2-VL model via Ollama (stronger vision understanding than LLaVA)
MODEL_NAME = "qwen3-vl:8b"

# Prompt for caption generation
CAPTION_PROMPT = f"""Describe this Mercedes-Benz E-Class Sedan (2012 model) image for AI training.

Start your response with "{TRIGGER_WORD}, " then provide a detailed description including:
1. The EXACT color of the vehicle (be specific: black, white, silver, red, blue, gray, etc.)
2. The viewing angle (front view, rear view, side view, three-quarter view, front three-quarter view, rear three-quarter view)
3. The environment/setting (street, parking lot, outdoor, indoor, road, driveway, etc.)
4. Any distinctive features visible (grille, headlights, wheels, body lines, etc.)
5. Background elements if relevant (trees, buildings, other vehicles, etc.)

Be precise about colors and angles. Keep the description concise but accurate (1-2 sentences max).

Format: "{TRIGGER_WORD}, [your detailed description]"

Only provide the caption, no additional text."""


def encode_image_to_base64(image_path: str) -> str:
    """Encode image to base64 string for Ollama."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def generate_caption_for_image(image_path: str) -> str:
    """Generate an enhanced caption for a single image using LLaVA via Ollama."""

    # Encode the image
    image_base64 = encode_image_to_base64(image_path)

    # Prepare the request for Ollama
    payload = {
        "model": MODEL_NAME,
        "prompt": CAPTION_PROMPT,
        "images": [image_base64],
        "stream": False
    }

    # Call Ollama API
    result = subprocess.run(
        ["ollama", "run", MODEL_NAME],
        input=json.dumps(payload),
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise Exception(f"Ollama error: {result.stderr}")

    # Parse the response - try alternate approach with curl
    result = subprocess.run(
        ["curl", "-s", "http://localhost:11434/api/generate", "-d", json.dumps(payload)],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise Exception(f"API call failed: {result.stderr}")

    # Parse JSON response
    response = json.loads(result.stdout)
    caption = response.get("response", "").strip()

    # Ensure it starts with trigger word
    if not caption.lower().startswith(TRIGGER_WORD.lower()):
        caption = f"{TRIGGER_WORD}, {caption}"

    # Clean up any extra whitespace
    caption = " ".join(caption.split())

    return caption


def main():
    """Main function to process all images in the dataset."""

    print(f"Using Qwen2-VL model via Ollama: {MODEL_NAME}")
    print("This model provides superior color and detail accuracy for captions.\n")

    # Get all image files
    dataset_path = Path(DATASET_DIR)
    image_extensions = {".jpg", ".jpeg", ".png"}
    image_files = [f for f in dataset_path.iterdir()
                   if f.suffix.lower() in image_extensions]

    print(f"Found {len(image_files)} images in {DATASET_DIR}")
    print(f"Starting caption generation...\n")

    # Process each image
    processed = 0
    errors = 0

    for img_file in sorted(image_files):
        txt_file = img_file.with_suffix(".txt")

        try:
            print(f"Processing: {img_file.name}...", end=" ", flush=True)

            # Generate caption
            caption = generate_caption_for_image(str(img_file))

            # Write caption to text file (overwrite existing)
            with open(txt_file, "w", encoding="utf-8") as f:
                f.write(caption)

            print(f"✓")
            print(f"  Caption: {caption}")

            processed += 1

            # Small delay to be respectful to Ollama
            time.sleep(0.5)

        except Exception as e:
            print(f"✗ Error: {e}")
            errors += 1
            continue

    print(f"\n{'='*80}")
    print(f"Processing complete!")
    print(f"Successfully processed: {processed}/{len(image_files)}")
    print(f"Errors: {errors}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
