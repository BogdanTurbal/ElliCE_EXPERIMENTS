"""
Configuration constants and model registry for the experiment.
"""

import re
from typing import Dict, List, Any, Optional

# Import method classes
from robustx.generators.CE_methods.KDTreeNNCE import KDTreeNNCE
from robustx.generators.CE_methods.Wachter import Wachter
from robustx.generators.robust_CE_methods.RNCE import RNCE
from robustx.generators.robust_CE_methods.STCE import TRex
from robustx.generators.robust_CE_methods.PROPLACE import PROPLACE
from robustx.generators.robust_CE_methods.LastLayerEllipsoidCE import (
    LastLayerEllipsoidCEOHC,
    LastLayerEllipsoidCEOHCNT,
    LastLayerEllipsoidCEOHCBall,
)
from robustx.generators.robust_CE_methods.LastLayerEllipsoidCENam import (
    LastLayerEllipsoidCEOHCNam,
    LastLayerEllipsoidCEOHCNTNam,
)
from robustx.generators.robust_CE_methods.TRexI import TRexI

BASE_EVALUATORS = [
    "Validity", 
    "Distance", 
    "DistanceM", 
    "LOF"
]


class ModelRegistry:
    """
    Dynamic model registry that parses model names to extract architecture information.
    """
    
    def __init__(self):
        # Default model configurations
        self._models = {
            "Linear": dict(hidden_dim=[], output_dim=1),
        }
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        Get model configuration by parsing the model name dynamically.
        
        Args:
            model_name: Model name like "MLP16x8", "MLP32x16x8", "Linear", etc.
            
        Returns:
            Dictionary with hidden_dim and output_dim
        """
        if model_name in self._models:
            return self._models[model_name].copy()
        
        # Parse MLP models dynamically
        if model_name.startswith("MLP"):
            hidden_dims = self._parse_mlp_architecture(model_name)
            return dict(hidden_dim=hidden_dims, output_dim=1)
        
        raise ValueError(f"Unknown model name: {model_name}")
    
    def _parse_mlp_architecture(self, model_name: str) -> List[int]:
        """
        Parse MLP architecture from model name.
        
        Examples:
            "MLP16x8" -> [16, 8]
            "MLP32x16x8" -> [32, 16, 8]
            "MLP128" -> [128]
        """
        # Extract numbers from the model name
        numbers = re.findall(r'\d+', model_name)
        
        if not numbers:
            raise ValueError(f"Could not parse architecture from model name: {model_name}")
        
        # Convert to integers
        hidden_dims = [int(num) for num in numbers]
        
        return hidden_dims
    
    def add_model(self, model_name: str, config: Dict[str, Any]):
        """Add a custom model configuration."""
        self._models[model_name] = config.copy()
    
    def list_models(self) -> List[str]:
        """List all available model names."""
        return list(self._models.keys())
    
    def get_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get all model configurations."""
        return {name: self.get_model_config(name) for name in self._models.keys()}


# Create global instance
MODEL_REGISTRY = ModelRegistry()

# Add legacy models for backward compatibility
legacy_models = {
    "MLP16x8": dict(hidden_dim=[16, 8], output_dim=1),
    "MLP16x32": dict(hidden_dim=[16, 32], output_dim=1),
    "MLP32x16": dict(hidden_dim=[32, 16], output_dim=1),
    "MLP16x16": dict(hidden_dim=[16, 16], output_dim=1),
    "MLP16x16x16": dict(hidden_dim=[16, 16, 16], output_dim=1),
    "MLP32x32": dict(hidden_dim=[32, 32], output_dim=1),
    "MLP16": dict(hidden_dim=[16], output_dim=1),
    "MLP64x64": dict(hidden_dim=[64, 64], output_dim=1),
    "MLP128": dict(hidden_dim=[128], output_dim=1),
    "MLP64": dict(hidden_dim=[64], output_dim=1),
    "MLP32": dict(hidden_dim=[32], output_dim=1),
    "NAM": dict(hidden_dim=[],           # unused for NAM, kept for schema stability
            output_dim=1,            # override in cfg["model"] for multiclass
            activation="relu",        # or "relu"
            num_basis_functions=8,
            dropout = 0.1,
            feature_dropout = 0.0,
            l2_regularization = 0.001,
            lr = 0.01,
            num_epochs = 100,
            early_stopping_patience = 20,
    )
}

for name, config in legacy_models.items():
    MODEL_REGISTRY.add_model(name, config)


# =============================================================================
#  Base evaluators and robustness metric constants & visuals
# =============================================================================
BASE_EVALUATORS = ["Validity", "Distance", "DistanceM", "LOF"]

METRIC_SHORT = {
    "Retrain-Robustness-RFA": "Retrain",
    "AWP-Robustness-RFA": "AWP",
    "ROB-Robustness-RFA": "ROB",
}

# Method colors and styles for cross-method comparison
METHOD_COLORS = {
    "RNCE": "#e31a1c",
    "ROAR": "#e31a1c",
    "STCE": "#33a02c", 
    "LastLayerEllipsoidCEOH": "#1f78b4",
    "LastLayerEllipsoidCEOHC": "#6a3d9a",
    "LastLayerEllipsoidCEOHCNT": "#6a3d9a",
    "PROPLACE": "#ff7f00",
    "LastLayerEllipsoidCECuttingPlane": "#b15928",
    "TRexI": "#3180f7",
    # NAM-specific methods
    "LastLayerEllipsoidCEOHCNam": "#a6cee3",
    "LastLayerEllipsoidCEOHCNTNam": "#b2df8a",
}

METHOD_MARKERS = {
    "RNCE": "o",
    "ROAR": "o",
    "STCE": "s", 
    "LastLayerEllipsoidCEOH": "^",
    "LastLayerEllipsoidCEOHC": "d",
    "LastLayerEllipsoidCEOHCNT": "d",
    "PROPLACE": "*",
    "LastLayerEllipsoidCECuttingPlane": "x",
    "TRexI": "P",
    # NAM-specific methods
    "LastLayerEllipsoidCEOHCNam": "v",
    "LastLayerEllipsoidCEOHCNTNam": "<",
}

METHOD_LINESTYLES = {
    "RNCE": "-",
    "ROAR": "-",
    "STCE": "--", 
    "LastLayerEllipsoidCEOH": "-.",
    "LastLayerEllipsoidCEOHC": ":",
    "LastLayerEllipsoidCEOHCNT": ":",
    "PROPLACE": "-",
    "LastLayerEllipsoidCECuttingPlane": "--",
    "TRexI": "-",
    # NAM-specific methods
    "LastLayerEllipsoidCEOHCNam": "-.",
    "LastLayerEllipsoidCEOHCNTNam": ":",
}

# =============================================================================
#  Method class mapping
# =============================================================================
METHOD_CLASSES = {
    "RNCE": RNCE,
    "STCE": TRex,
    "LastLayerEllipsoidCEOHC": LastLayerEllipsoidCEOHC,
    "KDTreeNNCE": KDTreeNNCE,
    "Wachter": Wachter,
    "LastLayerEllipsoidCEOHCNT": LastLayerEllipsoidCEOHCNT,
    "PROPLACE": PROPLACE,
    "LastLayerEllipsoidCEOHCBall": LastLayerEllipsoidCEOHCBall,
    "TRexI": TRexI,
    # NAM-specific methods
    "LastLayerEllipsoidCEOHCNam": LastLayerEllipsoidCEOHCNam,
    "LastLayerEllipsoidCEOHCNTNam": LastLayerEllipsoidCEOHCNTNam,
}
