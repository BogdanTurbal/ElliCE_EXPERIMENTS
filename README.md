[![Pypi](https://img.shields.io/pypi/v/nanogcg?color=blue)](https://pypi.org/project/ellice)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS_2025-Spotlight-red)](https://neurips.cc/virtual/2025/loc/san-diego/poster/118970)

# ElliCE: Efficient and Provably Robust Algorithmic Recourse via the Rashomon Sets
Full experimental code for the NeurIPS 2025 paper "ElliCE: Efficient and Provably Robust Algorithmic Recourse via the Rashomon Sets".
> 🚀 The lightweight ElliCE library is available here: [github.com/BogdanTurbal/ellice](https://github.com/BogdanTurbal/ellice)

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
