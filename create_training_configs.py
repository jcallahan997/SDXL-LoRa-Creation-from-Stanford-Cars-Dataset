#!/usr/bin/env python3
"""
Generate Kohya LoRA training configuration files for each car model.
Creates TOML config files optimized for SDXL LoRA training on M4 Max.
"""

import os
from pathlib import Path

# Base paths
TRAINING_DATA_BASE = Path("./lora_training/mercedes_vw_test").absolute()
OUTPUT_DIR = Path("./lora_outputs").absolute()
KOHYA_DIR = Path("/Users/jamescallahan/Projects/stable-diffusion/kohya").absolute()

# SDXL base model path (update this to your SDXL model location)
SDXL_MODEL = KOHYA_DIR / "models" / "sd_xl_base_1.0.safetensors"

# Training models
MODELS = {
    "mercedesbenz_300class_convertible_1993": {
        "trigger": "mercedesbenz300classconvertible1993",
        "name": "Mercedes-Benz 300-Class Convertible 1993"
    },
    "mercedesbenz_cclass_sedan_2012": {
        "trigger": "mercedesbenzccl asssedan2012",
        "name": "Mercedes-Benz C-Class Sedan 2012"
    },
    "mercedesbenz_eclass_sedan_2012": {
        "trigger": "mercedesbenzeclasssedan2012",
        "name": "Mercedes-Benz E-Class Sedan 2012"
    },
    "mercedesbenz_sclass_sedan_2012": {
        "trigger": "mercedesbenzsclasssedan2012",
        "name": "Mercedes-Benz S-Class Sedan 2012"
    },
    "mercedesbenz_slclass_coupe_2009": {
        "trigger": "mercedesbenzslclasscoupe2009",
        "name": "Mercedes-Benz SL-Class Coupe 2009"
    },
    "mercedesbenz_sprinter_van_2012": {
        "trigger": "mercedesbenzsprintervan2012",
        "name": "Mercedes-Benz Sprinter Van 2012"
    },
    "volkswagen_beetle_hatchback_2012": {
        "trigger": "volkswagenbeetlehatchback2012",
        "name": "Volkswagen Beetle Hatchback 2012"
    },
    "volkswagen_golf_hatchback_1991": {
        "trigger": "volkswagengolfhatchback1991",
        "name": "Volkswagen Golf Hatchback 1991"
    },
    "volkswagen_golf_hatchback_2012": {
        "trigger": "volkswagengolfhatchback2012",
        "name": "Volkswagen Golf Hatchback 2012"
    },
}

def create_config(model_folder, trigger, model_name):
    """Create a TOML config file for Kohya LoRA training."""

    config = f'''# LoRA training configuration for {model_name}
# Optimized for SDXL on M4 Max (Metal)

[general]
enable_bucket = true
bucket_reso_steps = 64
bucket_no_upscale = false
min_bucket_reso = 512
max_bucket_reso = 1024

[model_arguments]
pretrained_model_name_or_path = "{SDXL_MODEL}"
v2 = false
v_parameterization = false

[dataset_arguments]
resolution = 1024
cache_latents = true
cache_latents_to_disk = true

[[datasets]]
  [[datasets.subsets]]
    image_dir = "{TRAINING_DATA_BASE / model_folder}"
    num_repeats = 10
    caption_extension = ".txt"

[training_arguments]
output_dir = "{OUTPUT_DIR / model_folder}"
output_name = "{model_folder}"
save_precision = "fp16"
save_every_n_epochs = 5
max_train_epochs = 15
train_batch_size = 2
gradient_accumulation_steps = 1
max_data_loader_n_workers = 2

[optimizer_arguments]
optimizer_type = "AdamW8bit"
learning_rate = 0.0001
lr_scheduler = "cosine"
lr_warmup_steps = 100

[network_arguments]
network_module = "networks.lora"
network_dim = 32
network_alpha = 16

[logging_arguments]
log_with = "tensorboard"
logging_dir = "{OUTPUT_DIR / model_folder / 'logs'}"

[sample_arguments]
sample_every_n_epochs = 5
sample_sampler = "euler_a"
sample_prompts = "{trigger}, {model_name.split()[0].lower()} car, front view, professional photography"

[saving_arguments]
save_model_as = "safetensors"
save_state = false
'''

    return config

def main():
    print("=" * 60)
    print("Creating Kohya LoRA Training Configurations")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create config directory
    config_dir = Path("./lora_configs")
    config_dir.mkdir(exist_ok=True)

    for model_folder, info in MODELS.items():
        trigger = info["trigger"]
        name = info["name"]

        print(f"\nCreating config for: {name}")

        # Generate config content
        config_content = create_config(model_folder, trigger, name)

        # Write config file
        config_file = config_dir / f"{model_folder}.toml"
        with open(config_file, 'w') as f:
            f.write(config_content)

        print(f"  Config saved: {config_file}")

    print("\n" + "=" * 60)
    print("Configuration files created!")
    print(f"Configs directory: {config_dir.absolute()}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("\nTo train a LoRA, run:")
    print(f"cd {KOHYA_DIR}")
    print("source venv/bin/activate")
    print(f"python sd-scripts/sdxl_train_network.py --config_file={config_dir.absolute()}/[model].toml")
    print("=" * 60)

if __name__ == "__main__":
    main()
