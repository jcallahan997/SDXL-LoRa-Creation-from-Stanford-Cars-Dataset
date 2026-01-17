#!/bin/bash

# Train all Mercedes-Benz and Volkswagen LoRAs sequentially
# Each LoRA takes approximately 1-2 hours on M4 Max

set -e

echo "========================================================"
echo "Training Mercedes-Benz and Volkswagen LoRAs"
echo "========================================================"
echo ""
echo "This will train 9 LoRAs:"
echo "  - 6 Mercedes-Benz models"
echo "  - 3 Volkswagen models"
echo ""
echo "Estimated time: 9-18 hours total (1-2 hours each)"
echo ""
echo "========================================================"
echo ""

# Paths
KOHYA_DIR="/Users/jamescallahan/Projects/stable-diffusion/kohya"
SCRIPTS_DIR="$KOHYA_DIR/sd-scripts"
CONFIG_DIR="/Users/jamescallahan/Projects/stable-diffusion/stanford dataset/lora_configs"

# Activate Kohya virtual environment
echo "Activating Kohya environment..."
cd "$KOHYA_DIR"
source venv/bin/activate

# Models to train
models=(
    "mercedesbenz_300class_convertible_1993"
    "mercedesbenz_cclass_sedan_2012"
    "mercedesbenz_eclass_sedan_2012"
    "mercedesbenz_sclass_sedan_2012"
    "mercedesbenz_slclass_coupe_2009"
    "mercedesbenz_sprinter_van_2012"
    "volkswagen_beetle_hatchback_2012"
    "volkswagen_golf_hatchback_1991"
    "volkswagen_golf_hatchback_2012"
)

# Train each model
counter=1
total=${#models[@]}

for model in "${models[@]}"; do
    echo ""
    echo "========================================================"
    echo "Training LoRA $counter/$total: $model"
    echo "========================================================"
    echo "Started at: $(date)"
    echo ""

    # Train the LoRA
    python "$SCRIPTS_DIR/sdxl_train_network.py" \
        --config_file="$CONFIG_DIR/$model.toml"

    echo ""
    echo "Completed at: $(date)"
    echo "LoRA saved to: lora_outputs/$model/"
    echo ""

    ((counter++))
done

echo ""
echo "========================================================"
echo "All LoRAs trained successfully!"
echo "========================================================"
echo "Completed at: $(date)"
echo ""
echo "LoRAs saved in: /Users/jamescallahan/Projects/stable-diffusion/stanford dataset/lora_outputs/"
echo ""
echo "To use a LoRA in ComfyUI:"
echo "  1. Copy .safetensors file to ComfyUI/models/loras/"
echo "  2. Load in ComfyUI with LoRA Loader node"
echo "  3. Use trigger word in prompt (e.g., 'mercedesbenzeclasssedan2012')"
echo ""
echo "========================================================"
