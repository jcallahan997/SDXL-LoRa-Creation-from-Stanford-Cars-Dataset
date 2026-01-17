# Mercedes E-Class LoRA Training

Train SDXL LoRA models for Mercedes-Benz and Volkswagen vehicles using the Kohya training framework.

## Overview

This project trains Stable Diffusion XL (SDXL) LoRA models to generate photorealistic images of specific car models. The training pipeline uses the Kohya ss-scripts framework optimized for Apple Silicon (M4 Max).

### Supported Models

**Mercedes-Benz:**
- 300-Class Convertible 1993
- C-Class Sedan 2012
- E-Class Sedan 2012
- S-Class Sedan 2012
- SL-Class Coupe 2009
- Sprinter Van 2012

**Volkswagen:**
- Beetle Hatchback 2012
- Golf Hatchback 1991
- Golf Hatchback 2012

## Project Structure

```
.
├── prepare_training_data.py    # Organizes images and captions for training
├── create_training_configs.py  # Generates TOML config files for each model
├── train_single_test.sh        # Train a single LoRA (E-Class test)
├── train_all_loras.sh          # Train all 9 LoRAs sequentially
├── test_eclass.toml            # Simple config for E-Class testing
├── combined/                   # Source images by car model
├── lora_training_kohya/        # Prepared training data with captions
├── lora_outputs/               # Trained LoRA models (generated)
└── lora_configs/               # Training configuration files (generated)
```

## Prerequisites

1. **Kohya ss-scripts** installed at `/Users/jamescallahan/Projects/stable-diffusion/kohya`
2. **SDXL base model** (`sd_xl_base_1.0.safetensors`) in the models directory
3. **Python 3.10+** with virtual environment
4. **Apple Silicon Mac** (M1/M2/M3/M4 recommended) or CUDA GPU

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

Test the setup with a single model (approximately 1-2 hours on M4 Max):

```bash
./train_single_test.sh
```

This will train a LoRA for the Mercedes E-Class Sedan 2012 using 87 training images.

### Prepare Training Data

If you have new images to train with:

```bash
python3 prepare_training_data.py
```

This script:
- Reads images from `combined/<model_name>/`
- Finds matching captions from enhanced caption directory
- Copies image/caption pairs to `lora_training_kohya/<model>/`

### Generate Training Configurations

Create TOML config files for Kohya training:

```bash
python3 create_training_configs.py
```

This generates optimized config files in `lora_configs/` for each vehicle model.

### Train All Models

Train all 9 LoRAs sequentially (estimated 9-18 hours total):

```bash
./train_all_loras.sh
```

Each LoRA takes 1-2 hours depending on hardware. The script will:
1. Activate the Kohya virtual environment
2. Train each model with its config file
3. Save outputs to `lora_outputs/<model>/`

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

### In ComfyUI

1. Copy the `.safetensors` file from `lora_outputs/<model>/` to `ComfyUI/models/loras/`
2. Add a "Load LoRA" node in your workflow
3. Use the trigger word in your prompt

### Trigger Words

Each model has a unique trigger word:

| Model | Trigger Word |
|-------|-------------|
| Mercedes E-Class 2012 | `mercedesbenzeclasssedan2012` |
| Mercedes C-Class 2012 | `mercedesbenzccl asssedan2012` |
| Mercedes S-Class 2012 | `mercedesbenzsclasssedan2012` |
| VW Beetle 2012 | `volkswagenbeetlehatchback2012` |
| VW Golf 1991 | `volkswagengolfhatchback1991` |

### Example Prompts

```
mercedesbenzeclasssedan2012, blue Mercedes E-Class sedan, front view, professional photography
mercedesbenzeclasssedan2012, black luxury sedan on city street, sunset lighting
mercedesbenzeclasssedan2012, parked in modern garage, studio lighting
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
