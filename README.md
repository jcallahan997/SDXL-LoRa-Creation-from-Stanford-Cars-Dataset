# Mercedes E-Class SDXL LoRA Training

Train a Stable Diffusion XL (SDXL) LoRA model for the Mercedes-Benz E-Class Sedan 2012 using the Kohya training framework.

## Overview

This project trains a high-quality SDXL LoRA model to generate photorealistic images of the **Mercedes-Benz E-Class Sedan (2012 model)** from the Stanford Cars Dataset.

**Key Features:**
- **Qwen2-VL Vision Model** for intelligent caption generation with accurate color detection
- **Kohya ss-scripts** for LoRA training, optimized for Apple Silicon (M4 Max)
- **87 training images** with detailed, AI-generated captions
- **Trigger word:** `mercedesbenzeclasssedan2012`

The trained LoRA can generate E-Class images with proper colors, angles, and styling consistent with the 2012 model year.

**Note:** While this repository contains scripts for multiple car models, the current implementation focuses on the Mercedes E-Class. The multi-model scripts (`create_training_configs.py`, `train_all_loras.sh`) are included for potential future expansion.

## Project Structure

```
.
├── generate_qwen_captions_eclass.py        # Generate captions using Qwen2-VL for E-Class
├── generate_remaining_captions_eclass.py   # Generate captions for uncaptioned E-Class images
├── train_single_test.sh                    # Train the E-Class LoRA
├── test_eclass.toml                        # Training config for E-Class
├── prepare_training_data.py                # [Multi-model] Organize training data
├── create_training_configs.py              # [Multi-model] Generate training configs
├── train_all_loras.sh                      # [Multi-model] Batch training script
├── merge_train_test.sh                     # Utility script
├── combined/                               # Source images from Stanford Cars Dataset
├── lora_training_kohya/mercedesbenz_eclass/  # E-Class training data with captions
├── lora_outputs/                           # Trained LoRA models output directory
└── lora_configs/                           # Generated training configuration files
```

**Active Scripts (E-Class):**
- Caption generation: `generate_qwen_captions_eclass.py`
- Training: `train_single_test.sh` with `test_eclass.toml`

**Multi-Model Scripts (For Future Expansion):**
- `prepare_training_data.py`, `create_training_configs.py`, `train_all_loras.sh`

## Prerequisites

1. **Kohya ss-scripts** installed at `/Users/jamescallahan/Projects/stable-diffusion/kohya`
2. **SDXL base model** (`sd_xl_base_1.0.safetensors`) in the models directory
3. **Python 3.10+** with virtual environment
4. **Apple Silicon Mac** (M1/M2/M3/M4 recommended) or CUDA GPU
5. **Ollama** with Qwen2-VL model (for caption generation)

## Installation

### 1. Clone this repository

```bash
git clone <your-repo-url>
cd stanford\ dataset
```

### 2. Install Kohya ss-scripts

Follow the [Kohya setup guide](https://github.com/kohya-ss/sd-scripts) to install Kohya in the expected location.

### 3. Download SDXL base model

Download `sd_xl_base_1.0.safetensors` from [Stability AI](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) and place it in:
```
/Users/jamescallahan/Projects/stable-diffusion/models/sd_xl_base_1.0.safetensors
```

## Usage

### Quick Start - Train Mercedes E-Class LoRA

Train the E-Class LoRA model (approximately 1-2 hours on M4 Max):

```bash
./train_single_test.sh
```

This trains the Mercedes E-Class Sedan 2012 LoRA using:
- **87 training images** with AI-generated captions
- **15 epochs** with saves every 5 epochs
- **1024x1024 resolution** with bucketing
- **Trigger word:** `mercedesbenzeclasssedan2012`

### Generate Captions with Qwen2-VL

High-quality captions are crucial for training effective LoRAs. This project uses Qwen2-VL (via Ollama) for superior color detection and spatial accuracy.

#### Setup Ollama and Qwen2-VL

```bash
# Install Ollama (if not already installed)
# Visit: https://ollama.ai

# Pull the Qwen2-VL model
ollama pull qwen2-vl:7b

# Start Ollama (if not running)
ollama serve
```

#### Generate Captions for E-Class Images

```bash
# Generate captions for all E-Class images
python3 generate_qwen_captions_eclass.py
```

This will:
- Process all images in the E-Class training directory
- Generate detailed captions including exact colors, viewing angles, and settings
- Save captions as `.txt` files next to each image
- Use the trigger word `mercedesbenzeclasssedan2012`

#### Generate Captions for Remaining Images

If you need to caption only images that don't have good captions yet:

```bash
# Skip images that already have detailed captions
python3 generate_remaining_captions_eclass.py
```

This is useful if:
- The caption generation was interrupted
- You added new images to the dataset
- You want to re-caption only specific images

**Caption Quality:**
The Qwen2-VL model provides:
- Accurate color detection (black, white, silver, red, blue, etc.)
- Precise viewing angles (front view, three-quarter view, etc.)
- Detailed environment descriptions (parking lot, street, indoor, etc.)
- Consistent trigger word usage for optimal LoRA training

### Advanced: Multi-Model Scripts (Future Expansion)

The repository includes scripts for training multiple car models, though currently only E-Class is implemented:

**Prepare Training Data for Multiple Models:**
```bash
python3 prepare_training_data.py
```

**Generate Configs for Multiple Models:**
```bash
python3 create_training_configs.py
```

**Train All Models Sequentially:**
```bash
./train_all_loras.sh  # Would train all 9 models (9-18 hours)
```

These scripts are configured for 9 car models (6 Mercedes, 3 Volkswagen) but require:
- Caption generation for each model
- Training data preparation
- Updated config files

For now, focus on the E-Class implementation using `train_single_test.sh`.

## Training Configuration

### Hardware Optimization

The training configs are optimized for Apple Silicon (M4 Max):

- **Batch size:** 2
- **Mixed precision:** Disabled (Metal doesn't support it well)
- **Gradient checkpointing:** Enabled (reduces VRAM usage)
- **Cache latents to disk:** Enabled (saves memory)
- **VAE batch size:** 1 (prevents OOM errors)

### Training Parameters

- **Network module:** LoRA
- **Network dim:** 32
- **Network alpha:** 16
- **Optimizer:** AdamW
- **Learning rate:** 0.00005
- **Max epochs:** 15
- **Resolution:** 1024x1024 (with bucketing)
- **Saves:** Every 5 epochs

### Bucket Settings

- **Enable bucket:** True
- **Min resolution:** 512px
- **Max resolution:** 1024px
- **Bucket step:** 64px

This allows training on images of varying aspect ratios without distortion.

## Using Trained LoRAs

### In ComfyUI (Recommended)

ComfyUI + SDXL + LoRA is an excellent combination for high-quality image generation with fine control over the generation process.

1. Copy the `.safetensors` file from `lora_outputs/mercedesbenz_eclass_sedan_2012/` to `ComfyUI/models/loras/`
2. In your ComfyUI workflow:
   - Load the SDXL base model (checkpoint loader)
   - Add a "Load LoRA" node
   - Select your E-Class LoRA file
   - Set LoRA strength (start with 0.8-1.0)
3. Use the trigger word `mercedesbenzeclasssedan2012` in your prompt
4. Generate images with your preferred sampler and settings

**Why ComfyUI?**
- Node-based workflow provides precise control
- Easy to adjust LoRA strength and blend multiple LoRAs
- Excellent SDXL support with optimizations
- Real-time preview and workflow saving

### Trigger Word

The E-Class LoRA uses the trigger word: **`mercedesbenzeclasssedan2012`**

Always include this trigger word in your prompts for best results.

### Example Prompts

```
mercedesbenzeclasssedan2012, blue Mercedes E-Class sedan, front view, professional photography
mercedesbenzeclasssedan2012, black luxury sedan on city street, sunset lighting
mercedesbenzeclasssedan2012, silver E-Class parked in modern garage, studio lighting
mercedesbenzeclasssedan2012, white Mercedes sedan, three-quarter front view, outdoor parking lot
mercedesbenzeclasssedan2012, red E-Class driving on highway, dynamic shot
```

## Monitoring Training

### TensorBoard

Training logs are saved to `lora_outputs/<model>/logs/`. View progress with:

```bash
tensorboard --logdir=lora_outputs/mercedesbenz_eclass_sedan_2012/logs/
```

### Sample Images

Sample images are generated every 5 epochs using the sample prompt defined in the config file.

## Troubleshooting

### Out of Memory (OOM)

- Reduce `train_batch_size` to 1
- Reduce `network_dim` to 16
- Enable `gradient_checkpointing = true`
- Reduce `max_bucket_reso` to 768

### Slow Training

- Increase `train_batch_size` if you have VRAM headroom
- Use `cache_latents_to_disk = false` if you have enough RAM
- Increase `max_data_loader_n_workers` to 4

### Poor Quality Results

- Increase `max_train_epochs` to 20-30
- Adjust `learning_rate` (try 0.0001 or 0.00001)
- Increase `network_dim` to 64 or 128
- Add more diverse training images

## File Paths

Update these paths in the scripts if your setup differs:

- **Kohya directory:** `/Users/jamescallahan/Projects/stable-diffusion/kohya`
- **SDXL model:** `/Users/jamescallahan/Projects/stable-diffusion/models/sd_xl_base_1.0.safetensors`
- **Training data:** `./lora_training_kohya/`
- **Output directory:** `./lora_outputs/`

## License

MIT License - See LICENSE file for details

## Acknowledgments

- [Kohya ss-scripts](https://github.com/kohya-ss/sd-scripts) - Training framework
- [Stability AI](https://stability.ai/) - SDXL base model
- [Stanford Cars Dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) - Training data source

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## Support

For issues with:
- **Kohya training:** Check [Kohya documentation](https://github.com/kohya-ss/sd-scripts)
- **SDXL model:** See [Stability AI docs](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- **This project:** Open a GitHub issue
