"""
Configuration Manager for Unified Multi-Method Robustness Evaluation
Handles execution modes: all, range, specific
"""

import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from copy import deepcopy


class ConfigManager:
    """
    Manages unified configuration execution with different modes:
    - all: Execute all dataset/model combinations
    - range: Execute a range of configurations (by index)
    - specific: Execute a specific dataset/model combination
    """
    
    def __init__(self, config_path: str):
        """Initialize with unified config file path."""
        self.config_path = Path(config_path)
        self.unified_config = self._load_unified_config()
        self.config_list = self._build_config_list()
    
    def _load_unified_config(self) -> Dict[str, Any]:
        """Load the unified configuration file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _build_config_list(self) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Build a list of all possible configurations.
        Returns: List of (dataset_name, model_type, config_dict) tuples
        """
        config_list = []
        global_defaults = self.unified_config.get('global_defaults', {})
        
        for dataset_name, dataset_config in self.unified_config['datasets'].items():
            for model_type, model_config in dataset_config['model_types'].items():
                # Merge global defaults with dataset-specific and model-specific configs
                config = deepcopy(global_defaults)
                config.update({k: v for k, v in dataset_config.items() if k != 'model_types'})
                config.update(model_config)
                
                # Ensure dataset name is set correctly
                config['dataset'] = dataset_name
                
                config_list.append((dataset_name, model_type, config))
        
        return config_list
    
    def get_config_count(self) -> int:
        """Get total number of configurations."""
        return len(self.config_list)
    
    def list_configs(self) -> List[str]:
        """List all available configurations in format 'dataset:model_type'."""
        return [f"{dataset}:{model_type}" for dataset, model_type, _ in self.config_list]
    
    def get_config_by_index(self, index: int) -> Tuple[str, str, Dict[str, Any]]:
        """Get configuration by index."""
        if 0 <= index < len(self.config_list):
            return self.config_list[index]
        raise IndexError(f"Index {index} out of range. Available indices: 0-{len(self.config_list)-1}")
    
    def get_config_by_name(self, dataset: str, model_type: str) -> Dict[str, Any]:
        """Get configuration by dataset and model type."""
        for d, m, config in self.config_list:
            if d == dataset and m == model_type:
                return config
        raise ValueError(f"Configuration '{dataset}:{model_type}' not found")
    
    def get_configs_by_range(self, start: int, end: int) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Get configurations by index range."""
        if start < 0 or end > len(self.config_list) or start >= end:
            raise ValueError(f"Invalid range {start}:{end}. Available indices: 0-{len(self.config_list)-1}")
        return self.config_list[start:end]
    
    def get_all_configs(self) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Get all configurations."""
        return self.config_list.copy()
    
    def save_config(self, config: Dict[str, Any], output_path: str) -> None:
        """Save a single configuration to a YAML file."""
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
    
    def print_config_summary(self) -> None:
        """Print a summary of all available configurations."""
        print(f"Unified Configuration Summary:")
        print(f"Total configurations: {len(self.config_list)}")
        print(f"Available configurations:")
        for i, (dataset, model_type, _) in enumerate(self.config_list):
            print(f"  {i:2d}: {dataset}:{model_type}")
    
    def execute_mode(self, mode: str, range_str: Optional[str] = None, 
                    specific_str: Optional[str] = None) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Execute based on mode and parameters.
        
        Args:
            mode: Execution mode ("all", "range", "specific")
            range_str: Range in format "start:end" (for range mode)
            specific_str: Specific config in format "dataset:model_type" (for specific mode)
        
        Returns:
            List of (dataset_name, model_type, config_dict) tuples to execute
        """
        if mode == "all":
            return self.get_all_configs()
        
        elif mode == "range":
            if not range_str:
                raise ValueError("Range mode requires --range parameter")
            try:
                start, end = map(int, range_str.split(':'))
                return self.get_configs_by_range(start, end)
            except ValueError:
                raise ValueError(f"Invalid range format '{range_str}'. Use 'start:end' format")
        
        elif mode == "specific":
            if not specific_str:
                raise ValueError("Specific mode requires --specific parameter")
            try:
                dataset, model_type = specific_str.split(':')
                config = self.get_config_by_name(dataset, model_type)
                return [(dataset, model_type, config)]
            except ValueError:
                raise ValueError(f"Invalid specific format '{specific_str}'. Use 'dataset:model_type' format")
        
        else:
            raise ValueError(f"Unknown mode '{mode}'. Available modes: all, range, specific")


def main():
    """Test the ConfigManager."""
    config_manager = ConfigManager("unified_config.yml")
    config_manager.print_config_summary()
    
    # Test different modes
    print("\nTesting execution modes:")
    
    # Test all mode
    all_configs = config_manager.execute_mode("all")
    print(f"All mode: {len(all_configs)} configurations")
    
    # Test range mode
    range_configs = config_manager.execute_mode("range", range_str="0:2")
    print(f"Range mode (0:2): {len(range_configs)} configurations")
    for dataset, model_type, _ in range_configs:
        print(f"  - {dataset}:{model_type}")
    
    # Test specific mode
    specific_configs = config_manager.execute_mode("specific", specific_str="banknote:linear")
    print(f"Specific mode (banknote:linear): {len(specific_configs)} configurations")
    for dataset, model_type, _ in specific_configs:
        print(f"  - {dataset}:{model_type}")


if __name__ == "__main__":
    main()
