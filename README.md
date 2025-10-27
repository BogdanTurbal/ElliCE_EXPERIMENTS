## Install required libraries
```bash
pip install -r requirements.txt
```

## Run experiments locally

### Using unified configuration files (recommended)
```bash
# Run all configurations
python main_exp_wandb_opt.py unified_config.yml -o output_dir --wandb_project project_name

# Run specific range of configurations
python main_exp_wandb_opt.py unified_config.yml -o output_dir --mode range --range 0:5

# Run specific dataset:model combination
python main_exp_wandb_opt.py unified_config.yml -o output_dir --mode specific --specific austrc:linear

# List all available configurations
python main_exp_wandb_opt.py unified_config.yml --list-configs
```

### Using individual configuration files
```bash
python main_exp_wandb_opt.py <config_path> -o <output_folder> --wandb_project <wandb_project>
```

**Logs/results:** `output_folder/`  
**Configs:** `configs/unified_config.yml` (non-continuous), `configs/unified_config_cont.yml` (continuous)
