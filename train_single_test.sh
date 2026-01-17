#!/bin/bash

# Train Mercedes E-Class LoRA
# Hardware Requirements: M4 Max with 36GB+ RAM
# Training time: ~1-2 hours on M4 Max (64GB)

set -e

echo "========================================================"
echo "Training Mercedes-Benz E-Class Sedan 2012 LoRA"
echo "========================================================"
echo ""
echo "Hardware Requirements: Apple Silicon with 36GB+ RAM"
echo "Training time: ~1-2 hours on M4 Max"
echo ""
echo "Model: Mercedes-Benz E-Class Sedan 2012"
echo "Images: 87"
echo "Epochs: 15"
echo ""
echo "Started at: $(date)"
echo "========================================================"
echo ""

# ⚠️  UPDATE THESE PATHS TO MATCH YOUR LOCAL SETUP ⚠️
# Replace /path/to/ with your actual directory paths

KOHYA_DIR="/path/to/kohya"
SCRIPTS_DIR="$KOHYA_DIR/sd-scripts"
CONFIG_FILE="./test_eclass.toml"

# Activate Kohya virtual environment
echo "Activating Kohya environment..."
cd "$KOHYA_DIR"
source venv/bin/activate

# Train the LoRA
echo "Starting training with config: $CONFIG_FILE"
echo ""
python "$SCRIPTS_DIR/sdxl_train_network.py" \
    --config_file="$CONFIG_FILE"

echo ""
echo "========================================================"
echo "Training complete!"
echo "========================================================"
echo "Completed at: $(date)"
echo ""
echo "LoRA saved to the output_dir specified in your config file"
echo ""
echo "To test in ComfyUI:"
echo "  1. Copy the .safetensors file to ComfyUI/models/loras/"
echo "  2. Use trigger word: 'mercedesbenzeclasssedan2012'"
echo "  3. Example prompt: 'mercedesbenzeclasssedan2012, blue Mercedes E-Class, front view'"
echo ""
echo "========================================================"
