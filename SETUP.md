# Setup Guide

Complete setup instructions for training the Mercedes E-Class SDXL LoRA model.

## Prerequisites

### Hardware Requirements (IMPORTANT)

**This setup is specifically designed for local training on Apple Silicon MacBooks:**

- **Apple Silicon Mac** (M1/M2/M3/M4 Max or Ultra required)
- **36GB+ RAM MINIMUM** (64GB recommended, especially for M4 Max)
- **50GB+ free disk space** (for models, training data, and outputs)
- **Python 3.10 or higher**
- **Git**

**Training Performance:**
- M4 Max (64GB): ~1-2 hours per LoRA
- M3 Max (36GB): ~2-3 hours per LoRA
- M1/M2 Pro: May need reduced batch sizes

**Note:** If you have less than 36GB RAM or want to use NVIDIA GPUs, you'll need to modify the training configs (batch size, resolution, etc.) in `test_eclass.toml`.

## Step-by-Step Setup

### 1. Install Kohya ss-scripts

Kohya ss-scripts is the training framework. Install it separately:

```bash
# Navigate to your projects directory
cd ~/Projects/stable-diffusion

# Clone Kohya repository
git clone https://github.com/kohya-ss/sd-scripts.git kohya
cd kohya

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For Apple Silicon, install PyTorch with MPS support:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install additional requirements
pip install -r requirements.txt
```

### 2. Download SDXL Base Model

Download the SDXL 1.0 base model:

```bash
# Create models directory
mkdir -p ~/Projects/stable-diffusion/models

# Download from Hugging Face
# Visit: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
# Download: sd_xl_base_1.0.safetensors (6.94 GB)
# Place in: ~/Projects/stable-diffusion/models/
```

Or use `wget`:

```bash
cd ~/Projects/stable-diffusion/models
wget https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors
```

### 3. Install Ollama and Qwen2-VL (for Caption Generation)

Ollama runs the Qwen2-VL vision model locally for generating high-quality image captions:

```bash
# Install Ollama
# Visit: https://ollama.ai and download for your platform

# Pull the Qwen2-VL model (8B parameters, ~5GB)
ollama pull qwen2-vl:7b

# Or use the 3B model for faster generation (less accurate)
ollama pull qwen3-vl:8b
```

Start Ollama server (required for caption generation):
```bash
ollama serve
```

### 4. Clone This Repository

```bash
cd ~/Projects/stable-diffusion
git clone <your-repo-url> "stanford dataset"
cd "stanford dataset"
```

### 5. Verify Directory Structure

Your directory structure should look like:

```
~/Projects/stable-diffusion/
├── kohya/                    # Kohya ss-scripts installation
│   ├── venv/
│   └── sd-scripts/
├── models/                   # SDXL and other models
│   └── sd_xl_base_1.0.safetensors
└── stanford dataset/         # This repository
    ├── prepare_training_data.py
    ├── create_training_configs.py
    ├── train_single_test.sh
    └── ...
```

### 6. Update File Paths

If your directory structure differs, update these files:

**In `train_single_test.sh`:**
```bash
KOHYA_DIR="/YOUR/PATH/TO/kohya"
```

**In `train_all_loras.sh`:**
```bash
KOHYA_DIR="/YOUR/PATH/TO/kohya"
CONFIG_DIR="/YOUR/PATH/TO/stanford dataset/lora_configs"
```

**In `create_training_configs.py`:**
```python
KOHYA_DIR = Path("/YOUR/PATH/TO/kohya").absolute()
SDXL_MODEL = KOHYA_DIR / "models" / "sd_xl_base_1.0.safetensors"
```

### 7. Generate Captions (Optional)

If you need to regenerate captions for E-Class images:

```bash
# Make sure Ollama is running
ollama serve

# Generate captions
python3 generate_qwen_captions_eclass.py
```

This creates detailed captions with accurate color detection using Qwen2-VL.

### 8. Train the E-Class LoRA

Run the training (approximately 1-2 hours on M4 Max):

```bash
./train_single_test.sh
```

This will:
- Train for 15 epochs
- Save checkpoints every 5 epochs
- Generate sample images during training
- Output to `lora_outputs/mercedesbenz_eclass_sedan_2012/`

Monitor progress: Training logs are saved to `lora_outputs/mercedesbenz_eclass_sedan_2012/logs/`.

### 9. Optional: Multi-Model Training

If you want to expand to other car models:

```bash
# Prepare data for all models
python3 prepare_training_data.py

# Generate configs
python3 create_training_configs.py

# Train all models (9-18 hours)
./train_all_loras.sh
```

Note: This requires caption generation and data preparation for each model.

## Troubleshooting

### "SDXL base model not found"

Make sure the model is at the expected path:
```bash
ls -lh ~/Projects/stable-diffusion/models/sd_xl_base_1.0.safetensors
```

### "Kohya not found" or import errors

Activate the Kohya virtual environment:
```bash
cd ~/Projects/stable-diffusion/kohya
source venv/bin/activate
```

Verify installation:
```bash
python -c "import torch; print(torch.__version__)"
```

### Out of Memory (OOM)

Edit the TOML config files and reduce:
- `train_batch_size = 1`
- `network_dim = 16`
- `max_bucket_reso = 768`

### MPS/Metal Issues (Apple Silicon)

If you get Metal errors, set environment variable:
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
./train_single_test.sh
```

Or edit the training script to include:
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python ...
```

### Permission Denied

Make scripts executable:
```bash
chmod +x train_single_test.sh train_all_loras.sh
```

## Hardware Recommendations

### Minimum Specs
- Apple M1 or NVIDIA GTX 1060 (6GB VRAM)
- 16GB RAM
- 50GB disk space
- Training time: 3-4 hours per LoRA

### Recommended Specs
- Apple M4 Max or NVIDIA RTX 4090
- 32GB+ RAM
- 100GB disk space (for all outputs)
- Training time: 1-2 hours per LoRA

## Next Steps

After training completes:

1. Find your LoRAs in `lora_outputs/<model>/`
2. Copy `.safetensors` files to ComfyUI or Automatic1111
3. Test generation with trigger words
4. Adjust training params if needed and retrain

See [README.md](README.md) for usage instructions.
