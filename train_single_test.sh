#!/bin/bash

# Train a single LoRA for testing (Mercedes E-Class)
# Takes ~1-2 hours on M4 Max

set -e

echo "========================================================"
echo "Training Test LoRA: Mercedes-Benz E-Class Sedan 2012"
echo "========================================================"
echo ""
echo "This will train ONE LoRA to test the setup."
echo "Training time: ~1-2 hours"
echo ""
echo "Model: Mercedes-Benz E-Class Sedan 2012"
echo "Images: 87"
echo "Epochs: 15"
echo ""
echo "Started at: $(date)"
echo "========================================================"
echo ""

# Paths
KOHYA_DIR="/Users/jamescallahan/Projects/stable-diffusion/kohya"
SCRIPTS_DIR="$KOHYA_DIR/sd-scripts"
CONFIG_FILE="/Users/jamescallahan/Projects/stable-diffusion/stanford dataset/lora_configs/mercedesbenz_eclass_sedan_2012.toml"

# Activate Kohya virtual environment
echo "Activating Kohya environment..."
cd "$KOHYA_DIR"
source venv/bin/activate

# Check SDXL model exists
if [ ! -f "/Users/jamescallahan/Projects/stable-diffusion/models/sd_xl_base_1.0.safetensors" ]; then
    echo "ERROR: SDXL base model not found!"
    echo "Expected: /Users/jamescallahan/Projects/stable-diffusion/models/sd_xl_base_1.0.safetensors"
    exit 1
fi

echo "SDXL model found ✓"
echo ""

# Train the LoRA
echo "Starting training..."
echo ""
python "$SCRIPTS_DIR/sdxl_train_network.py" \
    --config_file="$CONFIG_FILE"

echo ""
echo "========================================================"
echo "Training complete!"
echo "========================================================"
echo "Completed at: $(date)"
echo ""
echo "LoRA saved to:"
echo "  /Users/jamescallahan/Projects/stable-diffusion/stanford dataset/lora_outputs/mercedesbenz_eclass_sedan_2012/"
echo ""
echo "To test in ComfyUI:"
echo "  1. Copy the .safetensors file to ComfyUI/models/loras/"
echo "  2. Use trigger: 'mercedesbenzeclasssedan2012'"
echo "  3. Prompt: 'mercedesbenzeclasssedan2012, blue Mercedes E-Class, front view'"
echo ""
echo "If this works well, run ./train_all_loras.sh to train all 9 models"
echo "========================================================"
