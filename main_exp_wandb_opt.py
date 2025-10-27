import os
import sys
import time
import math
import json
import random
import argparse
import logging
from pathlib import Path
from copy import deepcopy
from typing import Dict, List, Tuple, Union, Optional, Any, Set

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid tkinter issues
import matplotlib.pyplot as plt
from scipy import stats, interpolate
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import data_processing.data_loading as dl
import plots.plotting_utils as pu
from plots.plotting_post_utils import Plotting

import yaml
import wandb
from tqdm.auto import tqdm
from configs.config_manager import ConfigManager

# Standard libs used later for GCP instance identification
import socket
import uuid

# ─── RobustX ────────────────────────────────────────────────────────────────
from robustx.lib.models.pytorch_models.SimpleNNModel import (
    SimpleNNModel,
    remove_dropout,
)
from robustx.lib.models.pytorch_models.NamAdapter import (
    NamAdapter,
    FinalLayerOnly,
)
from robustx.lib.tasks.ClassificationTask import ClassificationTask
from robustx.datasets.custom_datasets.CsvDatasetLoader import CsvDatasetLoader
from robustx.lib.DefaultBenchmark import default_benchmark

from robustx.generators.CE_methods.KDTreeNNCE import KDTreeNNCE
from robustx.generators.CE_methods.Wachter import Wachter
from robustx.generators.robust_CE_methods.RNCE import RNCE, AmbiguousRNCE
from robustx.generators.robust_CE_methods.STCE import TRex
from robustx.generators.robust_CE_methods.PROPLACE import PROPLACE

from robustx.robustness_evaluations.VaRRobustnessEvaluator import (
    VaRRobustnessEvaluatorGlobal,
)

from robustx.evaluations.SparsityEvaluator import SparsityEvaluator
from robustx.evaluations.LOFEvaluator import LOFEvaluator


from rashomon_set.diverse_mod_gen import (
    generate_awp_models,
    generate_awp_models_parallel,
    generate_random_retrained_models,
    generate_random_retrained_models_parallel,
    generate_rashomon_models_with_auto_dropout,
    generate_dropout_rashomon_models_parallel,
    estimate_optimal_dropout_params_parallel
)

# from robustx.generators.robust_CE_methods.LastLayerEllipsoidCE import (
#     LastLayerEllipsoidCEOH,
#     LastLayerEllipsoidCEOHC,
#     LastLayerEllipsoidCEOHCNT,
#     LastLayerEllipsoidCECuttingPlane
# )
from rashomon_set.elipsoid_evaluator import AnalyticalEllipsoidEvaluator
from robustx.generators.robust_CE_methods.TRexI import TRexI

# ─── Torch / devic
DEVICE = torch.device("cpu")

# =============================================================================
#  Evaluation functions for metrics computation
# =============================================================================
def evaluate_model_metrics(model, X, y):
    """
    Evaluate model on given data and return loss, accuracy, and f1 score.
    
    Args:
        model: Trained model
        X: Input features
        y: True labels
        
    Returns:
        dict: Dictionary containing loss, accuracy, and f1_score
    """
    # Compute loss
    loss = model.compute_loss(X.values if hasattr(X, 'values') else X, y.values if hasattr(y, 'values') else y)
    
    # Get predictions
    with torch.no_grad():
        if hasattr(X, 'values'):
            X_tensor = torch.tensor(X.values, dtype=torch.float32)
        else:
            X_tensor = torch.tensor(X, dtype=torch.float32)
        
        predictions = model.predict(X_tensor)
        
        # Convert to numpy for sklearn metrics
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.numpy()
        
        # Ensure predictions is a numpy array
        if hasattr(predictions, 'values'):
            predictions = predictions.values
        predictions = np.array(predictions)
        
        # Handle binary vs multiclass
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            # Multiclass case - take argmax
            y_pred = np.argmax(predictions, axis=1)
        else:
            # Binary case - threshold at 0.5
            y_pred = (predictions > 0.5).astype(int).flatten()
    
    # Convert y to numpy if needed
    if hasattr(y, 'values'):
        y_true = y.values
    else:
        y_true = y
    
    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # For f1_score, handle binary vs multiclass
    if len(np.unique(y_true)) == 2:
        f1 = f1_score(y_true, y_pred, average='binary')
    else:
        f1 = f1_score(y_true, y_pred, average='weighted')
    
    return {
        'loss': float(loss),
        'accuracy': float(accuracy),
        'f1_score': float(f1)
    }

# =============================================================================
#  Argument parsing & logging
# =============================================================================
parser = argparse.ArgumentParser(
    description="Multi-method robustness evaluation pipeline - optimized"
)
parser.add_argument("config", type=str, help="Path to YAML config (or unified_config.yml)")
parser.add_argument(
    "-o", "--output", type=str, required=True, help="Folder for all outputs"
)
parser.add_argument("--wandb_project", type=str, default="robustness-evaluation", help="W&B project")
parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity")
parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name")
parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
parser.add_argument("--wandb_offline", action="store_true", help="Enable W&B offline mode")

# Unified config execution modes
parser.add_argument("--mode", type=str, default="all", 
                   choices=["all", "range", "specific"],
                   help="Execution mode: all (all configs), range (by index), specific (by name)")
parser.add_argument("--range", type=str, help="Range in format 'start:end' (for range mode)")
parser.add_argument("--specific", type=str, help="Specific config in format 'dataset:model_type' (for specific mode)")
parser.add_argument("--list-configs", action="store_true", help="List all available configurations and exit")
parser.add_argument("--data_shift", action="store_true", default=False, help="Enable data shift experiment mode (splits each CV fold into 2 parts)")
parser.add_argument("--fold", type=int, help="Run only the specified fold (1-based indexing)")
parser.add_argument("--only_train", action="store_true", default=False, help="Only train base model and evaluate metrics, skip counterfactual generation")
parser.add_argument("--save_metrics", action="store_true", default=False, help="Save train/test/valid metrics (loss, accuracy, f1) to JSON files")
parser.add_argument("--sensitivity_analysis", action="store_true", default=False, help="Enable sensitivity analysis mode with specific epsilon")
parser.add_argument("--eps", type=float, default=0.04, help="Specific epsilon value for sensitivity analysis (default: 0.04)")

# GCP-specific arguments
parser.add_argument("--gcp", action="store_true", default=False, help="Enable GCP mode for distributed execution")
parser.add_argument("--gcs_bucket", type=str, help="GCS bucket for storing results (required in GCP mode)")
parser.add_argument("--job_id", type=str, help="Unique job ID for this GCP run (required in GCP mode)")
parser.add_argument("--instance_id", type=str, help="Instance ID for this worker (auto-generated if not provided)")
parser.add_argument("--num_instances", type=int, default=20, help="Total number of instances in this GCP job")
parser.add_argument("--experiments_per_instance", type=int, default=1, help="Number of experiments to run per instance")
parser.add_argument("--folds_per_cpu", type=int, default=1, help="Number of folds to run per CPU core (adaptive parallelization)")

args = parser.parse_args()

# GCP validation and setup
if args.gcp:
    if not args.gcs_bucket:
        raise ValueError("--gcs_bucket is required when --gcp is enabled")
    if not args.job_id:
        raise ValueError("--job_id is required when --gcp is enabled")
    
    # Generate instance ID if not provided
    if not args.instance_id:
        args.instance_id = f"{socket.gethostname()}_{uuid.uuid4().hex[:8]}"
    
    print(f"🚀 GCP Mode Enabled")
    print(f"   Job ID: {args.job_id}")
    print(f"   Instance ID: {args.instance_id}")
    print(f"   GCS Bucket: {args.gcs_bucket}")
    print(f"   Total Instances: {args.num_instances}")
    print(f"   Experiments per Instance: {args.experiments_per_instance}")
    print(f"   Folds per CPU: {args.folds_per_cpu}")

out_dir = Path(args.output).expanduser().resolve()
vis_dir = out_dir / "visuals"
fold_dir = out_dir / "fold_results"
for p in (out_dir, vis_dir, fold_dir):
    p.mkdir(parents=True, exist_ok=True)

# Enhanced logging setup for GCP
log_handlers = [logging.FileHandler(out_dir / "run.log"), logging.StreamHandler(sys.stdout)]

# Add GCP-specific log file if in GCP mode
if args.gcp:
    gcp_log_file = out_dir / f"gcp_{args.instance_id}.log"
    log_handlers.append(logging.FileHandler(gcp_log_file))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s | %(message)s",
    handlers=log_handlers,
)
logger = logging.getLogger("robust-multi")

# GCP-specific functions
if args.gcp:
    from google.cloud import storage
    def upload_to_gcs(bucket_name: str, gcs_path: str, local_file: Path):
        """Upload a file to GCS bucket."""
        try:
            client = storage.Client()
            bucket = client.get_bucket(bucket_name)
            blob_path = f"{gcs_path}/{local_file.name}" if gcs_path else local_file.name
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(str(local_file))
            logger.info(f"✅ Uploaded {local_file} to gs://{bucket_name}/{blob_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to upload {local_file} to GCS: {e}")
            return False
    
    def upload_directory_to_gcs(bucket_name: str, gcs_path: str, local_dir: Path):
        """Upload entire directory to GCS bucket."""
        try:
            client = storage.Client()
            bucket = client.get_bucket(bucket_name)
            
            uploaded_count = 0
            for file_path in local_dir.rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(local_dir)
                    blob_path = f"{gcs_path}/{relative_path}" if gcs_path else str(relative_path)
                    blob = bucket.blob(blob_path)
                    blob.upload_from_filename(str(file_path))
                    uploaded_count += 1
            
            logger.info(f"✅ Uploaded {uploaded_count} files from {local_dir} to gs://{bucket_name}/{gcs_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to upload directory {local_dir} to GCS: {e}")
            return False

# =============================================================================
#  Single wandb initialization (handles both unified and single config modes)
# =============================================================================
if not args.no_wandb:
    # Create unique wandb directory per process to avoid conflicts
    wandb_dir = out_dir / f"wandb_{os.getpid()}_{int(time.time())}"
    
    # Determine run name and tags based on execution mode and GCP
    gcp_suffix = f"_gcp_{args.instance_id}" if args.gcp else ""
    
    if hasattr(args, 'mode') and args.mode in ['all', 'range', 'specific']:
        run_name = f"unified_config_{args.mode}_{time.strftime('%Y%m%d_%H%M%S')}{gcp_suffix}"
        tags = ["unified_config", args.mode]
        config = {"mode": args.mode}
    else:
        # For single config mode, we'll update the name and config later
        run_name = f"single_config_{time.strftime('%Y%m%d_%H%M%S')}{gcp_suffix}"
        tags = ["single_config"]
        config = {}
    
    # Add GCP-specific config
    if args.gcp:
        config.update({
            "gcp_mode": True,
            "job_id": args.job_id,
            "instance_id": args.instance_id,
            "num_instances": args.num_instances,
            "experiments_per_instance": args.experiments_per_instance,
            "folds_per_cpu": args.folds_per_cpu
        })
        tags.append("gcp")
    
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config=config,
        tags=tags,
        mode="offline" if args.wandb_offline else "online",
        dir=wandb_dir
    )

# =============================================================================
#  Reproducibility
# =============================================================================
def set_seeds(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seeds(42)

# Import configuration constants
from configs.config import BASE_EVALUATORS, MODEL_REGISTRY, METRIC_SHORT, METHOD_COLORS, METHOD_MARKERS, METHOD_LINESTYLES, METHOD_CLASSES
# =============================================================================
#  Utility: build ε grid
# =============================================================================
def build_eps_grid(cfg) -> List[float]:
    eps_cfg = cfg
    if eps_cfg["mode"] == "values":
        return eps_cfg["values"]
    start, stop, step = eps_cfg["start"], eps_cfg["stop"], eps_cfg["step"]
    return [round(start + i * step, 10) for i in range(int((stop - start) / step) + 1)]

# =============================================================================
#  OPTIMIZED: Evaluate at given ε with multiple model groups (SINGLE CALL)
# =============================================================================
def evaluate_with_eps_optimized(
    eps: float,
    task: ClassificationTask,
    all_ensembles: Dict[str, Dict[str, Any]],  # Now grouped by target_eps
    method_name: str,
    trex_threshold: float,
    cfg: Dict,
) -> Tuple[Dict[str, Dict[str, float]], float]:
    """
    Optimized evaluation that uses pre-generated model groups for all target epsilons.
    Generates counterfactuals ONCE and evaluates against all model groups.
    Returns results grouped by target_eps.
    """
    base_evaluators = BASE_EVALUATORS#cfg["base_evaluators"]#["Validity", "Distance", "DistanceM", "LOF"]
    
    # Get method-specific config
    method_cfg = cfg["methods"][method_name]
    
    # Set method-specific parameters on the task
    if method_name == "RNCE":
        task.delta = eps
    
    # Set other method-specific parameters from config
    params = method_cfg.get("params")
    if params is not None:
        for param_name, param_value in params.items():
            setattr(task, param_name, param_value)
    
    # Prepare all evaluators for all target_eps at once
    all_evaluators = base_evaluators.copy()
    target_eps_list = list(all_ensembles.keys())
    
    # Create robustness evaluators for each target_eps - only for available model types
    for target_eps_str in target_eps_list:
        for metric in pu.ROB_METRICS:
            # Skip metrics for which we don't have models in sensitivity analysis
            if "AWP" in metric and len(all_ensembles[target_eps_str].get("awp_models", [])) == 0:
                continue  # Skip AWP metrics if no AWP models
            if "ROB" in metric and len(all_ensembles[target_eps_str].get("diff_ensamble", [])) == 0:
                continue  # Skip ROB metrics if no dropout models
            
            # Use prefixes that match the ensemble naming scheme
            if "Retrain" in metric:
                prefix = f"Ret-{target_eps_str}"
            elif "AWP" in metric:
                prefix = f"AWP-{target_eps_str}"
            elif "ROB" in metric:
                prefix = f"ROB-{target_eps_str}"
            elif "ELIPSOID" in metric:
                prefix = f"ELIPSOID-{target_eps_str}"
            else:
                prefix = f"{metric}-{target_eps_str}"
            
            all_evaluators.append((VaRRobustnessEvaluatorGlobal, f"{prefix}-Robustness-RFA"))
    
    # Flatten all model groups into a single parameter dict
    all_model_params = {}
    
    for target_eps_str, ensembles in all_ensembles.items():
        for ensemble_type, models in ensembles.items():
            if models:  # Only add non-empty ensembles
                if ensemble_type == "models":
                    # Maps to "Ret" prefix for retrained models
                    all_model_params[f"models_{target_eps_str}"] = models
                elif ensemble_type == "awp_models" and not (hasattr(args, 'data_shift') and args.data_shift):
                    # Maps to "AWP" prefix for AWP models (skip in data_shift mode)
                    all_model_params[f"awp_models_{target_eps_str}"] = models
                elif ensemble_type == "diff_ensamble" and not (hasattr(args, 'data_shift') and args.data_shift):
                    # Maps to "ROB" prefix for ROB models (skip in data_shift mode)
                    all_model_params[f"diff_ensamble_{target_eps_str}"] = models
                elif ensemble_type.startswith("elips_ensamble"):
                    # Maps to "ELIPSOID" prefix for ellipsoid models
                    all_model_params[f"elips_ensamble_{target_eps_str}"] = models
    
    # Single call to default_benchmark with all models and evaluators
    raw_results = default_benchmark(
        task,
        [method_name],
        all_evaluators,
        **all_model_params,
        eps=eps,
        delta=eps,
        bias_delta=eps,
        delta_last=eps,
        delta_last_bias=eps,
        trex_threshold=trex_threshold,
        trex_k=cfg.get("trex_k", 1),
        trex_K=cfg.get("trex_K", 1),
        # ElliCE-specific parameters
        ellice_lr=cfg.get("ellice_lr", 0.05),
        ellice_robust_coef=cfg.get("ellice_robust_coef", 1.0),
        ellice_sparsity_coef=cfg.get("ellice_sparsity_coef", 0.0),
        ellice_proximity_coef=cfg.get("ellice_proximity_coef", 0.1),
        ellice_max_iterations=cfg.get("ellice_max_iterations", 400),
        ellice_opt=cfg.get("ellice_opt", "adam"),
        PROPLACE_k=cfg.get("PROPLACE_k", 10),
        # Method-specific parameters
        #**params if params else {},
    )[0]
    
    # Extract benchmark time (index 1 in raw_results)
    benchmark_time = raw_results[1]
    
    # Parse results and group by target_eps
    results_by_target_eps = {}
    
    # Create dynamic mapping based on evaluator names
    evaluator_mapping = {}
    
    for i, evaluator in enumerate(['Time'] + all_evaluators):
        if isinstance(evaluator, tuple):
            evaluator_name = evaluator[1]  # Get the name from tuple
        else:
            evaluator_name = evaluator
        evaluator_mapping[evaluator_name] = i + 1  # +1 because index 0 is method name
    #print("Time", raw_results[evaluator_mapping.get("Time", 1)])
    base_results = {
        "Time": raw_results[evaluator_mapping.get("Time", 1)],
        # "Validity": raw_results[evaluator_mapping.get("Validity", 2)],
        # "Distance": raw_results[evaluator_mapping.get("Distance", 3)],
        # "DistanceM": raw_results[evaluator_mapping.get("DistanceM", 4)],
        # "LOF": raw_results[evaluator_mapping.get("LOF", 5)],
    }
    for i in range(2, 2 + len(base_evaluators)):
        base_results[base_evaluators[i - 2]] = raw_results[evaluator_mapping.get(base_evaluators[i - 2], i)]
    
    # All target_eps get the same basic metrics (Time, Validity, Distance)
    for target_eps_str in target_eps_list:
        results_by_target_eps[target_eps_str] = base_results.copy()
    
    # Parse robustness metrics by target_eps using dynamic mapping
    for target_eps_str in target_eps_list:
        for metric in pu.ROB_METRICS:
            # In data_shift mode, skip AWP and ROB metrics since we don't generate those models
            if hasattr(args, 'data_shift') and args.data_shift:
                if "AWP" in metric or "ROB" in metric:
                    results_by_target_eps[target_eps_str][metric] = 0.0  # Set to 0 for skipped metrics
                    continue
            
            # Create the expected evaluator name pattern
            if "Retrain" in metric:
                expected_pattern = f"Ret-{target_eps_str}-Robustness-RFA"
            elif "AWP" in metric:
                expected_pattern = f"AWP-{target_eps_str}-Robustness-RFA"
            elif "ROB" in metric:
                expected_pattern = f"ROB-{target_eps_str}-Robustness-RFA"
            else:
                expected_pattern = f"{metric}-{target_eps_str}"
            
            # Find the evaluator name that matches this pattern
            evaluator_name = None
            for evaluator in all_evaluators:
                if isinstance(evaluator, tuple):
                    eval_name = evaluator[1]
                    if eval_name == expected_pattern:
                        evaluator_name = eval_name
                        break
                else:
                    if evaluator == expected_pattern:
                        evaluator_name = evaluator
                        break
            
            if evaluator_name and evaluator_name in evaluator_mapping:
                results_by_target_eps[target_eps_str][metric] = raw_results[evaluator_mapping[evaluator_name]]
            else:
                # If evaluator not found, set to 0.0
                results_by_target_eps[target_eps_str][metric] = 0.0
    
    return results_by_target_eps, benchmark_time

# =============================================================================
#  Helpers – temporary CSVs for CsvDatasetLoader
# =============================================================================
# Create unique tmp_cv directory per process to avoid conflicts when running multiple scripts in parallel
TMP_DIR_NAME = f"tmp_cv_{os.getpid()}_{int(time.time())}"
TMP = Path(TMP_DIR_NAME);TMP.mkdir(exist_ok=True)

def _df_to_loader(df: pd.DataFrame, split: str, fold: int, tag: str):
    # Ensure TMP directory exists (it should already be created above)
    TMP.mkdir(exist_ok=True)
    p = TMP / f"f{fold}_{tag}_{split}.csv"
    df.to_csv(p, index=False)
    return CsvDatasetLoader(csv=p, target_column="target")


def run_sensitivity_analysis_enhanced(cfg: Dict, fold: int, base_model, ds_trn, ds_val, ds_tst, vis_dir: Path, fold_dir: Path, target_eps: float) -> Dict:
    """
    Enhanced sensitivity analysis with:
    - Each worker gets 4 CPUs and runs 5 experiments
    - Evaluates both retrain (at target_eps) AND ellipsoid evaluators
    - Ellipsoid evaluator runs for epsilon range 0.01 to 0.1 (without stopping)
    - Records robustness history for both retrain AND ellipsoid evaluator
    """
    print(f"\n──── Running Enhanced Sensitivity Analysis for ε = {target_eps:.4f} ────")
    
    model_tag = cfg["model"]["cls"]
    base_kwargs = {**MODEL_REGISTRY.get_model_config(model_tag), **cfg["model"]}
    
    # Generate model ensembles for the target epsilon
    print(f"\n  ➤ Generating model ensembles for Target ε = {target_eps:.4f}")
    
    # Calculate Rashomon bound for the target_eps
    max_loss = base_model.compute_loss(ds_trn.X.values, ds_trn.y.values) + target_eps
    print(f"    Max allowed loss (Rashomon bound): {max_loss:.4f}")
    
    # Generate retrained models with 4 CPUs per worker
    mlp_l2_reg = cfg.get("mlp_l2_reg", 0.001)
    filter_loss_l2 = cfg.get("filter_loss_l2", 0.001)
    
    # Use 4 CPUs per worker for retrained models
    n_jobs = 4
    print(f"    Using parallel retrained models with {n_jobs} CPUs per worker")
    retrained = generate_random_retrained_models_parallel(
        ds_trn.X, ds_trn.y, ds_val.X, ds_val.y,
        base_kwargs["hidden_dim"], cfg["n_retrained"], cfg["epochs"], max_loss=max_loss,
        mlp_l2_reg=mlp_l2_reg, filter_loss_l2=filter_loss_l2, n_jobs=n_jobs
    )
    
    print(f"    Generated: {len(retrained)} retrained models")
    
    # Skip AWP and dropout models for sensitivity analysis - only use retrained and ellipsoid
    print("    Skipping AWP and dropout models for sensitivity analysis")
    awp = []  # Empty AWP models
    models_dif = []  # Empty dropout models
    
    # Store ensembles (only retrained models)
    target_eps_str = str(target_eps)
    all_ensembles = {
        target_eps_str: {
            "models": retrained,
            "awp_models": awp,
            "diff_ensamble": models_dif,
        }
    }
    
    print(f"    Using: {len(retrained)} retrained models only (AWP and dropout skipped)")
    
    # Initialize result structure with robustness history tracking
    sensitivity_results = {}
    robustness_history = {
        'retrain': {},
        'ellipsoid': {}
    }
    
    for method_name in cfg["methods"]:
        sensitivity_results[method_name] = {}
        sensitivity_results[method_name][target_eps_str] = {m: {} for m in pu.ROB_METRICS}
        robustness_history['retrain'][method_name] = []
        robustness_history['ellipsoid'][method_name] = []
    
    # Add ellipsoid evaluators for ElliCE methods
    print("\n──── Setting up ellipsoid evaluators ────")
    for method_name in cfg["methods"]:
        if "Ellipsoid" in method_name:
            print(f"  Setting up ellipsoid evaluator for {method_name}")
            try:
                if model_tag == "NAM":
                    task_ellipsoid = ClassificationTask(base_model.remove_dropout(), ds_trn, ds_val, ds_trn)
                else:
                    task_ellipsoid = ClassificationTask(remove_dropout(base_model), ds_trn, ds_val, ds_trn)
                
                task_ellipsoid.eps = target_eps
                task_ellipsoid.ellipsoid_iters = cfg.get("ellipsoid_iters", 0)
                task_ellipsoid.reg_coef = cfg.get("reg_coef", 0.001)
                
                print(f"    Task setup: eps={task_ellipsoid.eps}, ellipsoid_iters={task_ellipsoid.ellipsoid_iters}, reg_coef={task_ellipsoid.reg_coef}")
                
                # Check if method is available in METHOD_CLASSES
                if method_name not in METHOD_CLASSES:
                    print(f"    WARNING: Method {method_name} not found in METHOD_CLASSES. Available methods: {list(METHOD_CLASSES.keys())}")
                    continue
                
                ellipsoid = METHOD_CLASSES[method_name](task=task_ellipsoid, device=DEVICE)
                ellipsoid_eval = [AnalyticalEllipsoidEvaluator(ellipsoid)]
                all_ensembles[target_eps_str][f"elips_ensamble_{target_eps_str}"] = ellipsoid_eval
                print(f"    ✅ Created ellipsoid evaluator for {method_name}")
                
            except Exception as e:
                print(f"    ❌ Failed to create ellipsoid evaluator for {method_name}: {e}")
                print(f"    Continuing without ellipsoid evaluator for {method_name}")
                # Continue without this ellipsoid evaluator

    # Evaluate methods with BOTH retrain and ellipsoid evaluators for epsilon range 0.01 to 0.1
    print("\n──── Evaluating methods with retrain and ellipsoid evaluators for ε range 0.01-0.1 ────")
    
    # Create epsilon range for BOTH evaluators (0.01 to 0.1)
    eps_range = [round(0.01 + i * 0.01, 3) for i in range(10)]  # 0.01, 0.02, ..., 0.10
    print(f"    Both evaluators will run for epsilon range: {eps_range}")
    
    for method_name in cfg["methods"]:
        print(f"\n  ── {method_name} ──")
        
        # Create base task
        if model_tag == "NAM":
            task = ClassificationTask(base_model.remove_dropout(), ds_trn, ds_val, ds_trn)
        else:
            task = ClassificationTask(remove_dropout(base_model), ds_trn, ds_val, ds_trn)
        task.ellipsoid_iters = cfg.get("ellipsoid_iters", 0)
        task.reg_coef = cfg.get("reg_coef", 0.001)
        
        print(f"    Available ensembles: {list(all_ensembles[target_eps_str].keys())}")
        
        # Evaluate with BOTH evaluators for each epsilon in range 0.01 to 0.1
        for eps in eps_range:
            print(f"    📊 Evaluating at ε = {eps:.3f}")
            task.eps = eps
            task.threshold = eps
            
            # 1. Evaluate with RETRAIN evaluator
            print(f"      🔄 Retrain evaluator at ε = {eps:.3f}")
            try:
                # Create retrain-specific ensembles for this epsilon
                retrain_ensembles = {
                    str(eps): {
                        "models": retrained  # Use retrained models
                    }
                }
                
                retrain_results, retrain_time = evaluate_with_eps_optimized(
                    eps, task, retrain_ensembles, method_name,
                    trex_threshold=eps, cfg=cfg
                )
                
                # Store retrain results
                if str(eps) in retrain_results:
                    for metric in retrain_results[str(eps)]:
                        if "Retrain" in metric:
                            robustness_value = retrain_results[str(eps)][metric]
                            robustness_history['retrain'][method_name].append({
                                'eps': eps,
                                'robustness': robustness_value,
                                'validity': retrain_results[str(eps)].get("Validity", 0),
                                'distance': retrain_results[str(eps)].get("Distance", 0)
                            })
                            
                            print(f"        Retrain: robustness={robustness_value:.4f}")
                
            except Exception as e:
                print(f"        ❌ Retrain evaluation failed for {method_name} at ε={eps:.3f}: {e}")
            
            # 2. Evaluate with ELLIPSOID evaluator
            if f"elips_ensamble_{target_eps_str}" in all_ensembles[target_eps_str]:
                print(f"      🔄 Ellipsoid evaluator at ε = {eps:.3f}")
                try:
                    # Create ellipsoid-specific ensembles for this epsilon
                    ellipsoid_ensembles = {
                        str(eps): {
                            "elips_ensamble": all_ensembles[target_eps_str][f"elips_ensamble_{target_eps_str}"]
                        }
                    }
                    
                    ellipsoid_results, ellipsoid_time = evaluate_with_eps_optimized(
                        eps, task, ellipsoid_ensembles, method_name,
                        trex_threshold=eps, cfg=cfg
                    )
                    
                    # Store ellipsoid results
                    if str(eps) in ellipsoid_results:
                        for metric in ellipsoid_results[str(eps)]:
                            if "ELIPSOID" in metric:
                                robustness_value = ellipsoid_results[str(eps)][metric]
                                robustness_history['ellipsoid'][method_name].append({
                                    'eps': eps,
                                    'robustness': robustness_value,
                                    'validity': ellipsoid_results[str(eps)].get("Validity", 0),
                                    'distance': ellipsoid_results[str(eps)].get("Distance", 0)
                                })
                                
                                print(f"        Ellipsoid: robustness={robustness_value:.4f}")
                    
                except Exception as e:
                    print(f"        ❌ Ellipsoid evaluation failed for {method_name} at ε={eps:.3f}: {e}")
        
        print(f"    ✅ Evaluation completed for {method_name} across ε range {eps_range}")
    
    # Add metadata and robustness history
    sensitivity_results['metadata'] = {
        'target_eps': target_eps,
        'fold': fold,
        'dataset': cfg["dataset"],
        'model_type': cfg["model"]["cls"],
        'max_loss': max_loss,
        'n_retrained': len(retrained),
        'n_awp': 0,  # Skipped for sensitivity analysis
        'n_dropout': 0,  # Skipped for sensitivity analysis
        'eps_range': eps_range,
        'n_workers': 5,  # 5 experiments per worker
        'cpus_per_worker': 4
    }
    
    sensitivity_results['robustness_history'] = robustness_history
    
    # Save sensitivity results
    sensitivity_file = fold_dir / f"sensitivity_analysis_enhanced_fold_{fold}_eps_{target_eps:.4f}.json"
    with sensitivity_file.open("w") as fp:
        json.dump(sensitivity_results, fp, indent=2)
    
    print(f"\n  ↳ Enhanced sensitivity analysis completed for fold {fold}")
    print(f"  ↳ Results saved to {sensitivity_file}")
    print(f"  ↳ Robustness history recorded for retrain and ellipsoid evaluators across ε range {eps_range[0]:.3f}-{eps_range[-1]:.3f}")
    
    # Upload to GCS if in GCP mode
    if args.gcp:
        gcs_path = f"{args.job_id}/instance_{args.instance_id}/fold_{fold}"
        upload_to_gcs(args.gcs_bucket, gcs_path, sensitivity_file)
        logger.info(f"📤 Fold {fold} enhanced sensitivity results uploaded to gs://{args.gcs_bucket}/{gcs_path}")
    
    # Log to wandb if enabled
    if not args.no_wandb:
        wandb.log({
            "sensitivity_analysis_mode": True,
            "target_eps": target_eps,
            "fold": fold,
            "dataset": cfg["dataset"],
            "model_type": cfg["model"]["cls"],
            "eps_range": eps_range,
            "n_workers": 5,
            "cpus_per_worker": 4
        })
    
    return sensitivity_results


    """
    Run sensitivity analysis for a specific epsilon value.
    This function evaluates robustness against Rashomon sets without hyperparameter tuning.
    """
    print(f"\n──── Running Sensitivity Analysis for ε = {target_eps:.4f} ────")
    
    model_tag = cfg["model"]["cls"]
    base_kwargs = {**MODEL_REGISTRY.get_model_config(model_tag), **cfg["model"]}
    
    # Generate model ensembles for the target epsilon
    print(f"\n  ➤ Generating model ensembles for Target ε = {target_eps:.4f}")
    
    # Calculate Rashomon bound for the target_eps
    max_loss = base_model.compute_loss(ds_trn.X.values, ds_trn.y.values) + target_eps
    print(f"    Max allowed loss (Rashomon bound): {max_loss:.4f}")
    
    # Generate retrained models
    mlp_l2_reg = cfg.get("mlp_l2_reg", 0.001)
    filter_loss_l2 = cfg.get("filter_loss_l2", 0.001)
    
    cpu_parallel = cfg.get('cpu_parallel', False)
    available_cpus = os.cpu_count()
    
    if cpu_parallel and available_cpus > 1:
        print(f"    Using parallel retrained models with {available_cpus} CPUs")
        retrained = generate_random_retrained_models_parallel(
            ds_trn.X, ds_trn.y, ds_val.X, ds_val.y,
            base_kwargs["hidden_dim"], cfg["n_retrained"], cfg["epochs"], max_loss=max_loss,
            mlp_l2_reg=mlp_l2_reg, filter_loss_l2=filter_loss_l2, n_jobs=available_cpus
        )
    else:
        print(f"    Using sequential retrained models")
        retrained = generate_random_retrained_models(
            ds_trn.X, ds_trn.y, ds_val.X, ds_val.y,
            base_kwargs["hidden_dim"], cfg["n_retrained"], cfg["epochs"], max_loss=max_loss,
            mlp_l2_reg=mlp_l2_reg, filter_loss_l2=filter_loss_l2
        )
    
    print(f"    Generated: {len(retrained)} retrained models")
    
    # Skip AWP and dropout models for sensitivity analysis - only use retrained and ellipsoid
    print("    Skipping AWP and dropout models for sensitivity analysis")
    awp = []  # Empty AWP models
    models_dif = []  # Empty dropout models
    
    # Store ensembles (only retrained models)
    target_eps_str = str(target_eps)
    all_ensembles = {
        target_eps_str: {
            "models": retrained,
            "awp_models": awp,
            "diff_ensamble": models_dif,
        }
    }
    
    print(f"    Using: {len(retrained)} retrained models only (AWP and dropout skipped)")
    
    # Initialize result structure
    sensitivity_results = {}
    for method_name in cfg["methods"]:
        sensitivity_results[method_name] = {}
        sensitivity_results[method_name][target_eps_str] = {m: {} for m in pu.ROB_METRICS}
    
    # Add ellipsoid evaluators for ElliCE methods
    print("\n──── Setting up ellipsoid evaluators ────")
    for method_name in cfg["methods"]:
        if "Ellipsoid" in method_name:
            print(f"  Setting up ellipsoid evaluator for {method_name}")
            try:
                if model_tag == "NAM":
                    task_ellipsoid = ClassificationTask(base_model.remove_dropout(), ds_trn, ds_val, ds_trn)
                else:
                    task_ellipsoid = ClassificationTask(remove_dropout(base_model), ds_trn, ds_val, ds_trn)
                
                task_ellipsoid.eps = target_eps
                task_ellipsoid.ellipsoid_iters = cfg.get("ellipsoid_iters", 0)
                task_ellipsoid.reg_coef = cfg.get("reg_coef", 0.001)
                
                print(f"    Task setup: eps={task_ellipsoid.eps}, ellipsoid_iters={task_ellipsoid.ellipsoid_iters}, reg_coef={task_ellipsoid.reg_coef}")
                
                # Check if method is available in METHOD_CLASSES
                if method_name not in METHOD_CLASSES:
                    print(f"    WARNING: Method {method_name} not found in METHOD_CLASSES. Available methods: {list(METHOD_CLASSES.keys())}")
                    continue
                
                ellipsoid = METHOD_CLASSES[method_name](task=task_ellipsoid, device=DEVICE)
                ellipsoid_eval = [AnalyticalEllipsoidEvaluator(ellipsoid)]
                all_ensembles[target_eps_str][f"elips_ensamble_{target_eps_str}"] = ellipsoid_eval
                print(f"    ✅ Created ellipsoid evaluator for {method_name}")
                
            except Exception as e:
                print(f"    ❌ Failed to create ellipsoid evaluator for {method_name}: {e}")
                print(f"    Continuing without ellipsoid evaluator for {method_name}")
                # Continue without this ellipsoid evaluator

    # Evaluate methods with the specific epsilon
    print("\n──── Evaluating methods with specific epsilon ────")
    for method_name in cfg["methods"]:
        print(f"\n  ── {method_name} ──")
        
        # Create base task
        if model_tag == "NAM":
            task = ClassificationTask(base_model.remove_dropout(), ds_trn, ds_val, ds_trn)
        else:
            task = ClassificationTask(remove_dropout(base_model), ds_trn, ds_val, ds_trn)
        task.ellipsoid_iters = cfg.get("ellipsoid_iters", 0)
        task.reg_coef = cfg.get("reg_coef", 0.001)
        
        # Evaluate with the specific epsilon
        task.eps = target_eps
        task.threshold = target_eps
        
        print(f"    Task parameters: eps={task.eps}, threshold={task.threshold}")
        print(f"    Available ensembles: {list(all_ensembles[target_eps_str].keys())}")
        
        try:
            results_by_target_eps, benchmark_time = evaluate_with_eps_optimized(
                target_eps, task, all_ensembles, method_name,
                trex_threshold=target_eps, cfg=cfg
            )
            print(f"    ✅ Evaluation completed successfully for {method_name}")
        except Exception as e:
            print(f"    ❌ Evaluation failed for {method_name}: {e}")
            print(f"    Skipping {method_name}")
            continue
        
        # Store results
        for metric in pu.ROB_METRICS:
            if metric in results_by_target_eps[target_eps_str]:
                sensitivity_results[method_name][target_eps_str][metric] = {
                    'robustness': results_by_target_eps[target_eps_str][metric],
                    'validity': results_by_target_eps[target_eps_str]["Validity"],
                    'distance': results_by_target_eps[target_eps_str]["Distance"],
                    'distance_m': results_by_target_eps[target_eps_str]["DistanceM"],
                    'lof': results_by_target_eps[target_eps_str]["LOF"],
                    'benchmark_time': benchmark_time
                }
                
                print(f"    {METRIC_SHORT[metric]}: robustness={results_by_target_eps[target_eps_str][metric]:.4f}, "
                      f"validity={results_by_target_eps[target_eps_str]['Validity']:.4f}")
    
    # Add metadata
    sensitivity_results['metadata'] = {
        'target_eps': target_eps,
        'fold': fold,
        'dataset': cfg["dataset"],
        'model_type': cfg["model"]["cls"],
        'max_loss': max_loss,
        'n_retrained': len(retrained),
        'n_awp': 0,  # Skipped for sensitivity analysis
        'n_dropout': 0  # Skipped for sensitivity analysis
    }
    
    # Save sensitivity results
    sensitivity_file = fold_dir / f"sensitivity_analysis_fold_{fold}_eps_{target_eps:.4f}.json"
    with sensitivity_file.open("w") as fp:
        json.dump(sensitivity_results, fp, indent=2)
    
    print(f"\n  ↳ Sensitivity analysis completed for fold {fold}")
    print(f"  ↳ Results saved to {sensitivity_file}")
    
    # Upload to GCS if in GCP mode
    if args.gcp:
        gcs_path = f"{args.job_id}/instance_{args.instance_id}/fold_{fold}"
        upload_to_gcs(args.gcs_bucket, gcs_path, sensitivity_file)
        logger.info(f"📤 Fold {fold} sensitivity results uploaded to gs://{args.gcs_bucket}/{gcs_path}")
    
    # Log to wandb if enabled
    if not args.no_wandb:
        wandb.log({
            "sensitivity_analysis_mode": True,
            "target_eps": target_eps,
            "fold": fold,
            "dataset": cfg["dataset"],
            "model_type": cfg["model"]["cls"]
        })
    
    return sensitivity_results


def run_fold(cfg: Dict, fold: int, idx_trainval, idx_test, X, y, vis_dir: Optional[Path] = None, fold_dir: Optional[Path] = None, data_shift: bool = False) -> Dict:
    # Use provided directories or fall back to global ones
    if vis_dir is None:
        vis_dir = globals()['vis_dir']
    if fold_dir is None:
        fold_dir = globals()['fold_dir']
    
    print(f"\n──────── Fold {fold} ────────")
    in_dim = X.shape[1]
    
    # Initialize timing tracking
    fold_timing = {
        'total_fold_time': 0,
        'method_timings': {},
        'epsilon_timings': {},
        'hyperparameter_tuning_times': {},
        'individual_benchmark_times': {}
    }
    fold_start_time = time.time()
    total_benchmark_time = 0.0
    model_tag = cfg["model"]["cls"]
    base_kwargs = {**MODEL_REGISTRY.get_model_config(model_tag), **cfg["model"]}

    if data_shift:
        # Data shift mode: split each part (train/val/test) into 2 parts with stratification
        print(f"  Data shift mode: splitting each part into 2 stratified parts")
        
        # Split train/val data into two parts
        X_trv, y_trv = X[idx_trainval], y[idx_trainval]
        X_tst, y_tst = X[idx_test], y[idx_test]
        
        # Split train/val into two parts (50-50 split)
        X_trv_1, X_trv_2, y_trv_1, y_trv_2 = train_test_split(
            X_trv, y_trv, test_size=0.5, stratify=y_trv, random_state=fold
        )
        
        # Split test into two parts (50-50 split)
        X_tst_1, X_tst_2, y_tst_1, y_tst_2 = train_test_split(
            X_tst, y_tst, test_size=0.5, stratify=y_tst, random_state=fold
        )
        
        # Use first part for main model training and counterfactual generation
        X_trn, X_val, y_trn, y_val = train_test_split(
            X_trv_1, y_trv_1, test_size=cfg["valid_split"], stratify=y_trv_1, random_state=fold
        )
        X_tst = X_tst_1
        y_tst = y_tst_1
        
        # Use second part for robustness evaluation models
        X_trn_eval, X_val_eval, y_trn_eval, y_val_eval = train_test_split(
            X_trv_2, y_trv_2, test_size=cfg["valid_split"], stratify=y_trv_2, random_state=fold
        )
        X_tst_eval = X_tst_2
        y_tst_eval = y_tst_2
        
        # Scale both parts
        X_trn, X_val, X_tst = dl.scale_fold_selective(X_trn, X_val, X_tst)
        X_trn_eval, X_val_eval, X_tst_eval = dl.scale_fold_selective(X_trn_eval, X_val_eval, X_tst_eval)
        
        print(f"  Main part shapes - Train: {X_trn.shape}, Val: {X_val.shape}, Test: {X_tst.shape}")
        print(f"  Eval part shapes - Train: {X_trn_eval.shape}, Val: {X_val_eval.shape}, Test: {X_tst_eval.shape}")
        
    else:
        # Normal mode: standard train/val/test split
        X_trv, y_trv = X[idx_trainval], y[idx_trainval]
        X_tst, y_tst = X[idx_test], y[idx_test]
        X_trn, X_val, y_trn, y_val = train_test_split(
            X_trv, y_trv, test_size=cfg["valid_split"], stratify=y_trv, random_state=fold
        )
        X_trn, X_val, X_tst = dl.scale_fold_selective(X_trn, X_val, X_tst)
        
        # For consistency, set eval parts to None in normal mode
        X_trn_eval = X_val_eval = X_tst_eval = None
        y_trn_eval = y_val_eval = y_tst_eval = None
    


    cols = [f"f{i}" for i in range(in_dim)]
    mk_df = lambda a, b: pd.DataFrame(a, columns=cols).assign(target=b)

    # Data loaders for consistent use
    ds_trn = _df_to_loader(mk_df(X_trn, y_trn), "train", fold, "all")
    ds_val = _df_to_loader(mk_df(X_val, y_val), "val", fold, "all")
    ds_tst = _df_to_loader(mk_df(X_tst, y_tst), "test", fold, "all")
    
    # Data loaders for evaluation part (only in data_shift mode)
    if data_shift:
        ds_trn_eval = _df_to_loader(mk_df(X_trn_eval, y_trn_eval), "train_eval", fold, "all")
        ds_val_eval = _df_to_loader(mk_df(X_val_eval, y_val_eval), "val_eval", fold, "all")
        ds_tst_eval = _df_to_loader(mk_df(X_tst_eval, y_tst_eval), "test_eval", fold, "all")
    else:
        ds_trn_eval = ds_val_eval = ds_tst_eval = None

    # Train base models once
    print("\n──── Training base models ────")
    
    dropout = cfg.get("dropout", 0.0)
    mlp_l2_reg = cfg.get("mlp_l2_reg", 0.001)
    filter_loss_l2 = cfg.get("filter_loss_l2", 0.001)
    # Instantiate base model
    if model_tag == "NAM":
        # Remove seed from base_kwargs to avoid conflict
        nam_kwargs = {k: v for k, v in base_kwargs.items() if k != 'seed'}
        base = NamAdapter(input_dim=in_dim, **nam_kwargs, seed=42 + fold, early_stopping=True, mlp_l2_reg=mlp_l2_reg)
    else:
        base = SimpleNNModel(in_dim, **base_kwargs, seed=42 + fold, early_stopping=True, dropout=dropout, mlp_l2_reg=mlp_l2_reg)
    
    base.train(ds_trn.X, ds_trn.y, ds_val.X, ds_val.y, epochs=cfg["epochs"], to_print=True)
    
    # Log initial model metrics
    print("\n──── Evaluating initial base model ────")
    train_metrics = evaluate_model_metrics(base, ds_trn.X, ds_trn.y)
    val_metrics = evaluate_model_metrics(base, ds_val.X, ds_val.y)
    test_metrics = evaluate_model_metrics(base, ds_tst.X, ds_tst.y)
    
    print(f"  ↳ Train - Loss: {train_metrics['loss']:.4f}, Accuracy: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1_score']:.4f}")
    print(f"  ↳ Val   - Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1_score']:.4f}")
    print(f"  ↳ Test  - Loss: {test_metrics['loss']:.4f}, Accuracy: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1_score']:.4f}")
    
    # Prepare metrics payload
    initial_metrics = {
        'fold': fold,
        'dataset': cfg["dataset"],
        'model_type': cfg["model"]["cls"],
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'config': {
            'epochs': cfg["epochs"],
            'hidden_dim': base_kwargs.get("hidden_dim"),
            'dropout': dropout,
            'valid_split': cfg["valid_split"]
        }
    }

    # Save metrics to JSON if --save_metrics is enabled (always, not only in only_train)
    if args.save_metrics:
        metrics_file = fold_dir / f"initial_model_metrics_fold_{fold}.json"
        with metrics_file.open("w") as f:
            json.dump(initial_metrics, f, indent=2)
        print(f"  ↳ Saved metrics to {metrics_file}")

    # Log to wandb if enabled
    if not args.no_wandb:
        wandb.log({
            f"initial_model/train_loss": train_metrics['loss'],
            f"initial_model/train_accuracy": train_metrics['accuracy'],
            f"initial_model/train_f1": train_metrics['f1_score'],
            f"initial_model/val_loss": val_metrics['loss'],
            f"initial_model/val_accuracy": val_metrics['accuracy'],
            f"initial_model/val_f1": val_metrics['f1_score'],
            f"initial_model/test_loss": test_metrics['loss'],
            f"initial_model/test_accuracy": test_metrics['accuracy'],
            f"initial_model/test_f1": test_metrics['f1_score']
        })
    
    desired_loss = base.compute_loss(ds_trn.X.values, ds_trn.y.values)
    print(f"  ↳ Base model (main) loss: {desired_loss:.4f}")
    
    # If --only_train is specified, evaluate metrics and return early
    if args.only_train:
        print("\n──── Only Train Mode: Evaluating base model metrics ────")
        
        # Metrics already collected and saved above when --save_metrics
        
        # Upload individual metrics file to GCS if in GCP mode
        if args.gcp and args.save_metrics:
            metrics_file = fold_dir / f"initial_model_metrics_fold_{fold}.json"
            if metrics_file.exists():
                gcs_path = f"{args.job_id}/instance_{args.instance_id}/fold_{fold}"
                upload_to_gcs(args.gcs_bucket, gcs_path, metrics_file)
                logger.info(f"📤 Fold {fold} metrics uploaded to gs://{args.gcs_bucket}/{gcs_path}")
        
        # Log to wandb if enabled
        if not args.no_wandb:
            wandb.log({
                "only_train_mode": True,
                "fold": fold,
                "dataset": cfg["dataset"],
                "model_type": cfg["model"]["cls"]
            })
        
        print(f"  ↳ Only train mode completed for fold {fold}")
        return initial_metrics
    
    # If --sensitivity_analysis is specified, run enhanced sensitivity analysis mode
    if args.sensitivity_analysis:
        print(f"\n──── Enhanced Sensitivity Analysis Mode: Using ε = {args.eps:.4f} ────")
        return run_sensitivity_analysis_enhanced(cfg, fold, base, ds_trn, ds_val, ds_tst, vis_dir, fold_dir, args.eps)
    
    # In data_shift mode, train a separate model on the evaluation part
    if data_shift:
        print("\n──── Training evaluation model ────")
        if model_tag == "NAM":
            # Remove seed from base_kwargs to avoid conflict
            nam_kwargs = {k: v for k, v in base_kwargs.items() if k != 'seed'}
            base_eval = NamAdapter(input_dim=in_dim, **nam_kwargs, seed=42 + fold, early_stopping=True, mlp_l2_reg=mlp_l2_reg)
        else:
            base_eval = SimpleNNModel(in_dim, **base_kwargs, seed=42 + fold, early_stopping=True, dropout=dropout, mlp_l2_reg=mlp_l2_reg)
        base_eval.train(ds_trn_eval.X, ds_trn_eval.y, ds_val_eval.X, ds_val_eval.y, epochs=cfg["epochs"], to_print=True)
        
        eval_loss = base_eval.compute_loss(ds_trn_eval.X.values, ds_trn_eval.y.values)
        print(f"  ↳ Base model (eval) loss: {eval_loss:.4f}")
    else:
        base_eval = base  # Use same model in normal mode

    print("\n──── Generating model ensembles for all target epsilons ────")
    all_ensembles = {}

    # Choose which model and data to use for ensemble generation
    if data_shift:
        # In data_shift mode, use evaluation model and data for ensemble generation
        ensemble_base = base_eval
        ensemble_ds_trn = ds_trn_eval
        ensemble_ds_val = ds_val_eval
        print("  Using evaluation model and data for ensemble generation")
    else:
        # In normal mode, use main model and data
        ensemble_base = base
        ensemble_ds_trn = ds_trn
        ensemble_ds_val = ds_val
        print("  Using main model and data for ensemble generation")

    # Get the largest target_eps
    largest_target_eps = max(cfg["target_epsilons"])
    # Calculate Rashomon bound for the largest target_eps
    max_loss_largest = ensemble_base.compute_loss(ensemble_ds_trn.X.values, ensemble_ds_trn.y.values) + largest_target_eps
    print(f"\n  ➤ Generating retrained models once for largest Target ε = {largest_target_eps:.4f}")
    print(f"    Max allowed loss (Rashomon bound) for largest ε: {max_loss_largest:.4f}")

    # Generate retrained models once with the largest max_loss
    # Use parallel retrained models if cpu_parallel is enabled in config
    cpu_parallel = cfg.get('cpu_parallel', False)
    
    # Detect available CPUs - adapt for both SLURM and GCP
    if 'SLURM_CPUS_PER_TASK' in os.environ:
        available_cpus = int(os.environ['SLURM_CPUS_PER_TASK'])
    elif args.gcp:
        # For GCP, use the folds_per_cpu parameter to determine parallelization
        available_cpus = min(os.cpu_count(), args.folds_per_cpu)
    else:
        available_cpus = os.cpu_count()
    
    if cpu_parallel and available_cpus > 1:
        print(f"    Using parallel retrained models with {available_cpus} CPUs (cpu_parallel=True)")
        all_retrained = generate_random_retrained_models_parallel(
            ensemble_ds_trn.X, ensemble_ds_trn.y, ensemble_ds_val.X, ensemble_ds_val.y,
            base_kwargs["hidden_dim"], cfg["n_retrained"], cfg["epochs"], max_loss=max_loss_largest,
            mlp_l2_reg=mlp_l2_reg, filter_loss_l2=filter_loss_l2, n_jobs=available_cpus
        )
    else:
        print(f"    Using sequential retrained models (cpu_parallel=False or single CPU)")
        all_retrained = generate_random_retrained_models(
            ensemble_ds_trn.X, ensemble_ds_trn.y, ensemble_ds_val.X, ensemble_ds_val.y,
            base_kwargs["hidden_dim"], cfg["n_retrained"], cfg["epochs"], max_loss=max_loss_largest,
            mlp_l2_reg=mlp_l2_reg, filter_loss_l2=filter_loss_l2
        )
    print(f"    Generated: {len(all_retrained)} retrained models for largest epsilon")

    for target_eps in cfg["target_epsilons"]:
        print(f"\n  ➤ Target ε = {target_eps:.4f}")
        
        # Calculate Rashomon bound for this target_eps
        max_loss = ensemble_base.compute_loss(ensemble_ds_trn.X.values, ensemble_ds_trn.y.values) + target_eps
        print(f"    Max allowed loss (Rashomon bound): {max_loss:.4f}")
        
        # Filter retrained models based on the current max_loss
        retrained = []
        for model in all_retrained:
            model_loss = model.compute_loss(ensemble_ds_trn.X.values, ensemble_ds_trn.y.values)
            if model_loss <= max_loss:
                retrained.append(model)
        
        print(f"    Filtered to {len(retrained)} retrained models for current epsilon")
        
        # Generate other model types based on data_shift mode
        if data_shift:
            # In data_shift mode, skip AWP and dropout models, only use retrained models
            awp = []  # No AWP models in data_shift mode
            models_dif = []  # No dropout models in data_shift mode
            print("    Skipping AWP and dropout models in data_shift mode")
        else:
            # In normal mode, generate all model types
            # Use parallel AWP if cpu_parallel is enabled in config
            cpu_parallel = cfg.get('cpu_parallel', False)
            
            # Detect available CPUs - adapt for both SLURM and GCP
            if 'SLURM_CPUS_PER_TASK' in os.environ:
                available_cpus = int(os.environ['SLURM_CPUS_PER_TASK'])
            elif args.gcp:
                # For GCP, use the folds_per_cpu parameter to determine parallelization
                available_cpus = min(os.cpu_count(), args.folds_per_cpu)
            else:
                available_cpus = os.cpu_count()
            
            if cpu_parallel and available_cpus > 1:
                print(f"    Using parallel AWP with {available_cpus} CPUs (cpu_parallel=True)")
                awp = generate_awp_models_parallel(
                    ensemble_base, ensemble_ds_trn.X.values, ensemble_ds_trn.y.values,
                    X_val=ensemble_ds_trn.X.values, y_val=ensemble_ds_trn.y.values,
                    max_loss=max_loss, n_models=cfg["n_awp"], reg_coef=0.001,
                    retrain_epochs=400, mlp_l2_reg=mlp_l2_reg, filter_loss_l2=filter_loss_l2, n_jobs=available_cpus
                )
            else:
                print(f"    Using sequential AWP (cpu_parallel=False or single CPU)")
                awp = generate_awp_models(
                    ensemble_base, ensemble_ds_trn.X.values, ensemble_ds_trn.y.values,
                    max_loss=max_loss, n_models=cfg["n_awp"], reg_coef=0.001,
                    X_val=ensemble_ds_trn.X.values, y_val=ensemble_ds_trn.y.values, retrain_epochs=400,
                    mlp_l2_reg=mlp_l2_reg, filter_loss_l2=filter_loss_l2
                )
            
            models_dif, _ = generate_rashomon_models_with_auto_dropout(
                base_model=ensemble_base,
                X_train=ensemble_ds_trn.X.values,
                y_train=ensemble_ds_trn.y.values,
                config=cfg,
                max_loss=max_loss,
                dropout_type='gaussian',
                filter_loss_l2=filter_loss_l2
            )
        
        # Store ensembles grouped by target_eps
        all_ensembles[str(target_eps)] = {
            "models": retrained,
            "awp_models": awp,
            "diff_ensamble": models_dif,
        }
        
        print(f"    Using: {len(retrained)} retrained, {len(awp)} AWP, {len(models_dif)} Rashomon models")
    # Initialize result structure
    fold_res: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}
    for method_name in cfg["methods"]:
        fold_res[method_name] = {}
        for target_eps in cfg["target_epsilons"]:
            fold_res[method_name][str(target_eps)] = {m: {} for m in pu.ROB_METRICS}
            
            
    max_sup = cfg.get("max_data_support")            # None ⇒ no limit
    if max_sup is not None and max_sup < len(X_trn):
        idx_sup = np.random.choice(len(X_trn), max_sup, replace=False)
        ds_data_sup = _df_to_loader(
            mk_df(X_trn[idx_sup], y_trn[idx_sup]), "train", fold, "sup"
        )
    else:
        ds_data_sup = ds_trn                           # keep full training set

    # Test-support sample ---------------------------------------------------------
    max_tst = cfg.get("max_test")                    # None ⇒ no limit
    if max_tst is not None and max_tst < len(X_tst):
        idx_tst_sub = np.random.choice(len(X_tst), max_tst, replace=False)
        ds_test_sub = _df_to_loader(
            mk_df(X_tst[idx_tst_sub], y_tst[idx_tst_sub]), "test", fold, "sub"
        )
    else:
        ds_test_sub = ds_tst    
        
    max_val = cfg.get("max_val")                    # None ⇒ no limit
    if max_val is not None and max_val < len(X_val):
        idx_val_sub = np.random.choice(len(X_val), max_val, replace=False)
        ds_tval_sub = _df_to_loader(
            mk_df(X_val[idx_val_sub], y_tst[idx_val_sub]), "val", fold, "sub"
        )
    else:
        ds_tval_sub = ds_val    

    # OPTIMIZED: Evaluate all methods using pre-generated ensembles
    print("\n──── Evaluating methods ────")
    for method_name in cfg["methods"]:
        print(f"\n  ── {method_name} ──")
        method_cfg = cfg["methods"][method_name]
        eps_grid = build_eps_grid(method_cfg["search"]["mhp"])
        
        # Initialize timing for this method
        method_start_time = time.time()
        method_benchmark_time = 0.0
        fold_timing['method_timings'][method_name] = {
            'hyperparameter_tuning_time': 0,
            'epsilon_step_times': {},
            'per_epsilon_times': {},
            'benchmark_time': 0.0
        }
        
        # Create base task
        if model_tag == "NAM":
            task = ClassificationTask(base.remove_dropout(), ds_trn, ds_tval_sub, ds_data_sup)  # Use validation set
        else:
            task = ClassificationTask(remove_dropout(base), ds_trn, ds_tval_sub, ds_data_sup)  # Use validation set
        task.ellipsoid_iters = cfg.get("ellipsoid_iters", 0)
        task.reg_coef = cfg.get("reg_coef", 0.001)
        
        # Add ellipsoid evaluators if needed
        if "Ellipsoid" in method_name:
            for target_eps_str, ensembles in all_ensembles.items():
                if data_shift:
                    # In data_shift mode, create ellipsoid using evaluation model and data
                    if model_tag == "NAM":
                        task_ellipsoid = ClassificationTask(base_eval.remove_dropout(), ds_trn_eval, ds_val_eval, ds_trn_eval)
                    else:
                        task_ellipsoid = ClassificationTask(remove_dropout(base_eval), ds_trn_eval, ds_val_eval, ds_trn_eval)
                else:
                    # In normal mode, use main model and data
                    if model_tag == "NAM":
                        task_ellipsoid = ClassificationTask(base.remove_dropout(), ds_trn, ds_val, ds_trn)
                    else:
                        task_ellipsoid = ClassificationTask(remove_dropout(base), ds_trn, ds_val, ds_trn)
                
                task_ellipsoid.eps = float(target_eps_str)
                task_ellipsoid.ellipsoid_iters = cfg.get("ellipsoid_iters", 0)
                task_ellipsoid.reg_coef = cfg.get("reg_coef", 0.001)
                ellipsoid = METHOD_CLASSES[method_name](task=task_ellipsoid, device=DEVICE)
                ellipsoid_eval = [AnalyticalEllipsoidEvaluator(ellipsoid)]
                ensembles[f"elips_ensamble_{target_eps_str}"] = ellipsoid_eval
        
        # Tune hyperparameters using all model groups simultaneously
        all_validation_results = {str(target_eps): {} for target_eps in cfg["target_epsilons"]}
        
        # Start hyperparameter tuning timing
        hyperparameter_tuning_start = time.time()
        print(f"    Tuning on validation set with {len(eps_grid)} test eps values...")
        
        # Track validity and robustness history for early stopping
        validity_history = {str(target_eps): [] for target_eps in cfg["target_epsilons"]}
        robustness_history = {str(target_eps): {metric: [] for metric in pu.ROB_METRICS} for target_eps in cfg["target_epsilons"]}
        max_validity_achieved = {str(target_eps): 0.0 for target_eps in cfg["target_epsilons"]}  # Track max validity for new criterion
        early_stopping_enabled = cfg.get("early_stopping", True)  # Enable by default
        print(f"    Early stopping: {'enabled' if early_stopping_enabled else 'disabled'}")
        
        for test_eps in eps_grid:
            # Start timing for this epsilon step
            #epsilon_step_start = time.time()
            task.eps = test_eps
            task.threshold = test_eps
            
            # Evaluate on all target_eps groups at once
            results_by_target_eps, benchmark_time = evaluate_with_eps_optimized(
                test_eps, task, all_ensembles, method_name,
                trex_threshold=test_eps, cfg=cfg
            )
            
            # Accumulate benchmark time
            method_benchmark_time += benchmark_time
            total_benchmark_time += benchmark_time
            
            # Check for early stopping: validity drop OR all robustness metrics achieve 1.0
            should_stop = False
            stop_reason = ""
            validity_summary = []
            robustness_summary = []
            
            for target_eps_str, res in results_by_target_eps.items():
                # Track validity
                current_validity = res["Validity"]
                validity_history[target_eps_str].append(current_validity)
                # Update max validity achieved for new early stopping criterion
                max_validity_achieved[target_eps_str] = max(max_validity_achieved[target_eps_str], current_validity)
                validity_summary.append(f"ε={target_eps_str}:{current_validity:.3f}")
                
                # Track robustness metrics
                current_robustness_values = []
                for metric in pu.ROB_METRICS:
                    current_robustness = res[metric]
                    robustness_history[target_eps_str][metric].append(current_robustness)
                    current_robustness_values.append(current_robustness)
                
                robustness_summary.append(f"ε={target_eps_str}:{current_robustness_values}")
                
                # Check if we had positive validity before and now have validity=0 (only if early stopping is enabled)
                if early_stopping_enabled and len(validity_history[target_eps_str]) >= 2:
                    prev_validity = validity_history[target_eps_str][-2]
                    if prev_validity > 0.0 and current_validity == 0.0:
                        print(f"    Early stopping: Validity dropped from {prev_validity:.3f} to 0.0 for target_eps={target_eps_str}")
                        should_stop = True
                        stop_reason = f"validity drop for target_eps={target_eps_str}"
                        break
                
                # Check if validity was > 0.9 at some point and now < 0.5 (only if early stopping is enabled)
                if early_stopping_enabled and max_validity_achieved[target_eps_str] > 0.9 and current_validity < 0.5:
                    print(f"    Early stopping: Validity dropped from max {max_validity_achieved[target_eps_str]:.3f} to {current_validity:.3f} (< 0.5) for target_eps={target_eps_str}")
                    should_stop = True
                    stop_reason = f"validity dropped below 0.5 after achieving > 0.9 for target_eps={target_eps_str}"
                    break
            
            # Check if all robustness metrics achieve 1.0 for all target_eps (only if early stopping is enabled)
            if early_stopping_enabled and not should_stop:
                all_robustness_achieved = True
                for target_eps_str in cfg["target_epsilons"]:
                    target_eps_str = str(target_eps_str)
                    for metric in pu.ROB_METRICS:
                        if len(robustness_history[target_eps_str][metric]) > 0:
                            latest_robustness = robustness_history[target_eps_str][metric][-1]
                            if latest_robustness < 1.0:
                                all_robustness_achieved = False
                                break
                    if not all_robustness_achieved:
                        break
                
                if all_robustness_achieved:
                    print(f"    Early stopping: All robustness metrics achieved 1.0 for all target_eps")
                    should_stop = True
                    stop_reason = "all robustness metrics achieved 1.0"
            
            # Log validity progression (only every few steps to avoid spam)
            if test_eps in eps_grid[::max(1, len(eps_grid)//10)]:  # Log every 10% of steps
                print(f"    test_eps={test_eps:.4f}: {' | '.join(validity_summary)}")
                if robustness_summary:  # Also log robustness if available
                    print(f"    robustness: {' | '.join(robustness_summary)}")
            
            # Always store results for this epsilon step before potentially stopping early
            for target_eps_str, res in results_by_target_eps.items():
                # Store all metrics including Validity
                all_validation_results[target_eps_str].setdefault("Validity", {})[str(test_eps)] = res["Validity"]
                for metric in pu.ROB_METRICS:
                    all_validation_results[target_eps_str].setdefault(metric, {})[str(test_eps)] = (res[metric], res["Distance"], res["DistanceM"], res["LOF"])
            
            # Record timing for this epsilon step
            #epsilon_step_time = time.time() - epsilon_step_start
            fold_timing['method_timings'][method_name]['epsilon_step_times'][str(test_eps)] = benchmark_time
            fold_timing['method_timings'][method_name]['per_epsilon_times'][str(test_eps)] = benchmark_time
            
            if should_stop:
                print(f"    Stopping hyperparameter tuning early due to {stop_reason}")
                break
        
        # Record total hyperparameter tuning time (sum of benchmark times during tuning)
        hyperparameter_tuning_time = method_benchmark_time
        fold_timing['method_timings'][method_name]['hyperparameter_tuning_time'] = hyperparameter_tuning_time
        fold_timing['hyperparameter_tuning_times'][method_name] = hyperparameter_tuning_time
                    
        # OPTIMIZED: Process results for each target_eps with shared test evaluation
        for target_eps in cfg["target_epsilons"]:
            target_eps_str = str(target_eps)
            validation_results = all_validation_results[target_eps_str]
            
            # Plot validation curves
            pu.plot_fold_validation_curves(validation_results, target_eps, fold, cfg["dataset"], model_tag, method_name, vis_dir, use_wandb=(not args.no_wandb))
            
            # Step 1: Find optimal test_eps for each metric
            metric_to_optimal_eps = {}
            metric_to_validation_info = {}
            
            for metric in pu.ROB_METRICS:
                if metric not in validation_results:
                    continue
                    
                # Find test_eps that maximizes validity first, then robustness
                test_options = []
                for test_eps_str, (robustness, l1_dist, l2_dist, lof) in validation_results[metric].items():
                    validity = validation_results["Validity"].get(test_eps_str, 0)
                    test_options.append((test_eps_str, validity, robustness, l1_dist, l2_dist, lof))

                # Sort by validity (descending), then by robustness (descending)
                best_option = max(test_options, key=lambda x: (x[1], x[2]))
                best_test_eps = best_option[0]
                
                metric_to_optimal_eps[metric] = float(best_test_eps)
                metric_to_validation_info[metric] = {
                    'test_eps_str': best_test_eps,
                    'validity': best_option[1],
                    'robustness': best_option[2],
                    'l1_dist': best_option[3],
                    'l2_dist': best_option[4],
                    'lof': best_option[5]
                }
                
                print(f"    {METRIC_SHORT[metric]} (ε={target_eps:.4f}): Optimal test_eps={best_test_eps}, "
                      f"validity={best_option[1]:.4f}, robustness={best_option[2]:.4f}")
            
            # Step 2: Get unique test_eps values and create mapping
            unique_test_eps = list(set(metric_to_optimal_eps.values()))
            # Fallback: if nothing selected (e.g., due to very-early stop), pick latest evaluated eps
            if not unique_test_eps:
                validity_map = validation_results.get("Validity", {})
                if validity_map:
                    latest_eps_str = list(validity_map.keys())[-1]
                    for metric in pu.ROB_METRICS:
                        if metric in validation_results and latest_eps_str in validation_results[metric]:
                            metric_to_optimal_eps[metric] = float(latest_eps_str)
                    unique_test_eps = list(set(metric_to_optimal_eps.values()))
                    
            eps_to_metrics = {}
            for metric, test_eps in metric_to_optimal_eps.items():
                if test_eps not in eps_to_metrics:
                    eps_to_metrics[test_eps] = []
                eps_to_metrics[test_eps].append(metric)
            
            print(f"    Unique test_eps values: {unique_test_eps}")
            print(f"    Mapping: {eps_to_metrics}")
            
            # Step 3: Evaluate once per unique test_eps 
            test_results_cache = {}
            
            for test_eps in unique_test_eps:
                print(f"    Evaluating test_eps={test_eps} for metrics: {eps_to_metrics[test_eps]}")
                
                # Setup test task
                if model_tag == "NAM":
                    task_test = ClassificationTask(base.remove_dropout(), ds_trn, ds_test_sub, ds_data_sup)
                else:
                    task_test = ClassificationTask(remove_dropout(base), ds_trn, ds_test_sub, ds_data_sup)
                task_test.eps = test_eps
                task_test.threshold = test_eps
                task_test.ellipsoid_iters = cfg.get("ellipsoid_iters", 0)
                task_test.reg_coef = cfg.get("reg_coef", 0.001)
                
                # Create test ensembles using the models for this target_eps
                test_ensembles = {target_eps_str: all_ensembles[target_eps_str]}
                
                # Evaluate once
                test_results, test_benchmark_time = evaluate_with_eps_optimized(
                    test_eps, task_test, test_ensembles, method_name,
                    trex_threshold=test_eps, cfg=cfg
                )
                
                # Accumulate benchmark time
                method_benchmark_time += test_benchmark_time
                total_benchmark_time += test_benchmark_time
                
                # Cache results
                test_results_cache[test_eps] = test_results[target_eps_str]
            
            # Step 4: Store results for each metric using cached evaluations
            for metric in pu.ROB_METRICS:
                if metric not in metric_to_optimal_eps:
                    continue
                    
                optimal_test_eps = metric_to_optimal_eps[metric]
                test_met = test_results_cache[optimal_test_eps]
                #validation_info = metric_to_validation_info[metric]
                
                # Store results
                fold_res[method_name][target_eps_str][metric] = {
                    'optimal_test_eps': optimal_test_eps,
                    'optimal_robustness': test_met[metric],
                     'optimal_validity': test_met["Validity"],
                    'optimal_distance': (test_met["Distance"], test_met["DistanceM"], test_met["LOF"]),
                    'validation_curve': validation_results[metric],
                    'validation_curve_with_validity': {
                        eps_str: {
                            'robustness': robustness,
                            'distance': distance,
                            'distance_m': distance_m,
                            'lof': lof,
                            'validity': validation_results["Validity"].get(eps_str, 0)
                        }
                        for eps_str, (robustness, distance, distance_m, lof) in validation_results[metric].items()
                    }
                }
                
                print(f"    {METRIC_SHORT[metric]} (ε={target_eps:.4f}): Test robustness={test_met[metric]:.4f}, "
                      f"distance=({test_met['Distance']:.4f}, {test_met['DistanceM']:.4f}, {test_met['LOF']:.4f})")
        
        # Record benchmark time
        fold_timing['method_timings'][method_name]['benchmark_time'] = method_benchmark_time
        
        # Log timing information
        print(f"    Method {method_name} completed")
        print(f"    Hyperparameter tuning: {hyperparameter_tuning_time:.2f}s")
        epsilon_times = list(fold_timing['method_timings'][method_name]['per_epsilon_times'].values())
        if epsilon_times:
            print(f"    Average epsilon step time: {np.mean(epsilon_times):.3f}s ± {np.std(epsilon_times):.3f}s")
            print(f"    Epsilon step times: {len(epsilon_times)} steps")

    # Record total fold time (sum of individual benchmark times)
    fold_timing['total_fold_time'] = total_benchmark_time
    fold_timing['individual_benchmark_times'] = {method: timing['benchmark_time'] for method, timing in fold_timing['method_timings'].items()}
    print(f"\nFold {fold} completed in {fold_timing['total_fold_time']:.2f}s (sum of benchmark times)")
    print(f"Wall clock time: {time.time() - fold_start_time:.2f}s")
    
    # Add timing information to fold results
    fold_res['timing'] = fold_timing

    # Save fold results
    fold_results_file = fold_dir / f"results_fold_{fold}.json"
    with fold_results_file.open("w") as fp:
        json.dump(fold_res, fp, indent=2)
    
    # Upload to GCS if in GCP mode
    if args.gcp:
        gcs_path = f"{args.job_id}/instance_{args.instance_id}/fold_{fold}"
        upload_to_gcs(args.gcs_bucket, gcs_path, fold_results_file)
        
        # Upload individual metrics file if it exists
        if args.save_metrics:
            metrics_file = fold_dir / f"initial_model_metrics_fold_{fold}.json"
            if metrics_file.exists():
                upload_to_gcs(args.gcs_bucket, gcs_path, metrics_file)
                logger.info(f"📤 Fold {fold} metrics uploaded to gs://{args.gcs_bucket}/{gcs_path}")
        
        # Upload validation plots if they exist
        for plot_file in vis_dir.glob("*.png"):
            upload_to_gcs(args.gcs_bucket, f"{gcs_path}/visuals", plot_file)
        
        logger.info(f"📤 Fold {fold} results uploaded to gs://{args.gcs_bucket}/{gcs_path}")
    
    return fold_res

# =============================================================================
#  Full run (unchanged from previous version)
# =============================================================================
def run_all(cfg: Dict, output_dir: Optional[Path] = None, vis_dir: Optional[Path] = None, fold_dir: Optional[Path] = None):
    # Use provided directories or fall back to global ones
    if output_dir is None:
        output_dir = globals()['out_dir']
    if vis_dir is None:
        vis_dir = globals()['vis_dir']
    if fold_dir is None:
        fold_dir = globals()['fold_dir']
    
    # Note: wandb.init() is now handled in the main entry point to avoid duplication

    X, y = dl.load_dataset(cfg["dataset"])
    splits = list(StratifiedKFold(cfg["cv_folds"], shuffle=True, random_state=42).split(X, y))

    all_folds: List[Tuple[int, dict]] = []
    
    # If --fold is specified, run only that fold
    if args.fold is not None:
        if args.fold < 1 or args.fold > cfg["cv_folds"]:
            raise ValueError(f"Fold {args.fold} is out of range. Must be between 1 and {cfg['cv_folds']}")
        fold_idx = args.fold - 1  # Convert to 0-based indexing
        idx_trv, idx_tst = splits[fold_idx]
        res = run_fold(cfg, args.fold, idx_trv, idx_tst, X, y, vis_dir, fold_dir, data_shift=args.data_shift)
        all_folds.append((args.fold, res))
        print(f"Running only fold {args.fold}")
    else:
        # Run all folds
        for fold, (idx_trv, idx_tst) in enumerate(splits, 1):
            res = run_fold(cfg, fold, idx_trv, idx_tst, X, y, vis_dir, fold_dir, data_shift=args.data_shift)
            all_folds.append((fold, res))

    print("\n==== Aggregating results and creating comparison plots ====")
    
    # Check if we're in only_train mode
    if args.only_train:
        print("Only train mode: Aggregating initial model metrics")
        
        # Aggregate initial model metrics across folds
        aggregated_metrics = {
            'dataset': cfg["dataset"],
            'model_type': cfg["model"]["cls"],
            'folds': [],
            'summary': {
                'train': {'loss': [], 'accuracy': [], 'f1_score': []},
                'val': {'loss': [], 'accuracy': [], 'f1_score': []},
                'test': {'loss': [], 'accuracy': [], 'f1_score': []}
            }
        }
        
        for fold, data in all_folds:
            if isinstance(data, dict) and 'train_metrics' in data:
                aggregated_metrics['folds'].append(data)
                
                # Add to summary
                for split in ['train', 'val', 'test']:
                    split_key = f'{split}_metrics'
                    if split_key in data:
                        for metric in ['loss', 'accuracy', 'f1_score']:
                            aggregated_metrics['summary'][split][metric].append(data[split_key][metric])
        
        # Calculate mean and std for summary
        for split in ['train', 'val', 'test']:
            for metric in ['loss', 'accuracy', 'f1_score']:
                values = aggregated_metrics['summary'][split][metric]
                if values:
                    aggregated_metrics['summary'][split][f'{metric}_mean'] = float(np.mean(values))
                    aggregated_metrics['summary'][split][f'{metric}_std'] = float(np.std(values))
        
        # Save aggregated metrics
        aggregated_file = out_dir / "aggregated_initial_model_metrics.json"
        with aggregated_file.open("w") as f:
            json.dump(aggregated_metrics, f, indent=2)
        print(f"Saved aggregated initial model metrics to {aggregated_file}")
        
        # Upload to GCS if in GCP mode
        if args.gcp:
            gcs_path = f"{args.job_id}/instance_{args.instance_id}"
            upload_to_gcs(args.gcs_bucket, gcs_path, aggregated_file)
            logger.info(f"📤 Aggregated metrics uploaded to gs://{args.gcs_bucket}/{gcs_path}")
        
        # Log summary to wandb if enabled
        if not args.no_wandb:
            for split in ['train', 'val', 'test']:
                for metric in ['loss', 'accuracy', 'f1_score']:
                    mean_key = f"aggregated/{split}_{metric}_mean"
                    std_key = f"aggregated/{split}_{metric}_std"
                    if f'{metric}_mean' in aggregated_metrics['summary'][split]:
                        wandb.log({
                            mean_key: aggregated_metrics['summary'][split][f'{metric}_mean'],
                            std_key: aggregated_metrics['summary'][split][f'{metric}_std']
                        })
        
        print("Only train mode completed successfully")
        return
    
    # Check if we're in sensitivity analysis mode
    if args.sensitivity_analysis:
        print(f"Sensitivity analysis mode: Aggregating results for ε = {args.eps:.4f}")
        
        # Aggregate sensitivity analysis results across folds
        aggregated_sensitivity = {
            'target_eps': args.eps,
            'dataset': cfg["dataset"],
            'model_type': cfg["model"]["cls"],
            'folds': [],
            'summary': {}
        }
        
        # Collect results from all folds
        for fold, data in all_folds:
            if isinstance(data, dict) and 'metadata' in data:
                aggregated_sensitivity['folds'].append(data)
        
        # Calculate summary statistics across folds
        target_eps_str = str(args.eps)
        for method_name in cfg["methods"]:
            if method_name in aggregated_sensitivity['folds'][0]:
                aggregated_sensitivity['summary'][method_name] = {}
                for metric in pu.ROB_METRICS:
                    if metric in aggregated_sensitivity['folds'][0][method_name][target_eps_str]:
                        robustness_values = []
                        validity_values = []
                        distance_values = []
                        
                        for fold_data in aggregated_sensitivity['folds']:
                            if (method_name in fold_data and 
                                target_eps_str in fold_data[method_name] and 
                                metric in fold_data[method_name][target_eps_str]):
                                
                                metric_data = fold_data[method_name][target_eps_str][metric]
                                robustness_values.append(metric_data['robustness'])
                                validity_values.append(metric_data['validity'])
                                distance_values.append(metric_data['distance'])
                        
                        if robustness_values:
                            aggregated_sensitivity['summary'][method_name][metric] = {
                                'robustness_mean': float(np.mean(robustness_values)),
                                'robustness_std': float(np.std(robustness_values)),
                                'validity_mean': float(np.mean(validity_values)),
                                'validity_std': float(np.std(validity_values)),
                                'distance_mean': float(np.mean(distance_values)),
                                'distance_std': float(np.std(distance_values))
                            }
        
        # Save aggregated sensitivity results
        aggregated_file = out_dir / f"aggregated_sensitivity_analysis_eps_{args.eps:.4f}.json"
        with aggregated_file.open("w") as f:
            json.dump(aggregated_sensitivity, f, indent=2)
        print(f"Saved aggregated sensitivity analysis results to {aggregated_file}")
        
        # Upload to GCS if in GCP mode
        if args.gcp:
            gcs_path = f"{args.job_id}/instance_{args.instance_id}"
            upload_to_gcs(args.gcs_bucket, gcs_path, aggregated_file)
            logger.info(f"📤 Aggregated sensitivity results uploaded to gs://{args.gcs_bucket}/{gcs_path}")
        
        # Log summary to wandb if enabled
        if not args.no_wandb:
            for method_name, method_summary in aggregated_sensitivity['summary'].items():
                for metric, metric_summary in method_summary.items():
                    wandb.log({
                        f"sensitivity/{method_name}/{metric}/robustness_mean": metric_summary['robustness_mean'],
                        f"sensitivity/{method_name}/{metric}/robustness_std": metric_summary['robustness_std'],
                        f"sensitivity/{method_name}/{metric}/validity_mean": metric_summary['validity_mean'],
                        f"sensitivity/{method_name}/{metric}/validity_std": metric_summary['validity_std']
                    })
        
        print("Sensitivity analysis mode completed successfully")
        return
    
    # Aggregate results across folds by method
    all_results: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}
    for method_name in cfg["methods"]:
        all_results[method_name] = {}
        for target_eps in cfg["target_epsilons"]:
            all_results[method_name][str(target_eps)] = {m: {} for m in pu.ROB_METRICS}
            
            for metric in pu.ROB_METRICS:
                for fold, data in all_folds:
                    if method_name in data and str(target_eps) in data[method_name] and metric in data[method_name][str(target_eps)]:
                        all_results[method_name][str(target_eps)][metric][f'fold_{fold}'] = data[method_name][str(target_eps)][metric]
    
    # Aggregate timing information across folds
    timing_summary = {
        'total_experiment_time': 0,
        'method_timings': {},
        'hyperparameter_tuning_summary': {},
        'epsilon_step_summary': {}
    }
    
    # Calculate timing statistics
    for method_name in cfg["methods"]:
        hyperparameter_times = []
        epsilon_step_times_all = []
        
        for fold, data in all_folds:
            if 'timing' in data and 'method_timings' in data['timing']:
                if method_name in data['timing']['method_timings']:
                    method_timing = data['timing']['method_timings'][method_name]
                    hyperparameter_times.append(method_timing['hyperparameter_tuning_time'])
                    
                    # Collect all epsilon step times
                    for eps_time in method_timing['per_epsilon_times'].values():
                        epsilon_step_times_all.append(eps_time)
        
        if hyperparameter_times:
            timing_summary['method_timings'][method_name] = {
                'mean_hyperparameter_time': np.mean(hyperparameter_times),
                'std_hyperparameter_time': np.std(hyperparameter_times),
                'mean_epsilon_step_time': np.mean(epsilon_step_times_all) if epsilon_step_times_all else 0,
                'std_epsilon_step_time': np.std(epsilon_step_times_all) if epsilon_step_times_all else 0,
                'total_epsilon_steps': len(epsilon_step_times_all)
            }
    
    # Log timing summary
    print("\n==== TIMING SUMMARY ====")
    for method_name, timing_stats in timing_summary['method_timings'].items():
        print(f"\n{method_name}:")
        print(f"  Hyperparameter tuning: {timing_stats['mean_hyperparameter_time']:.2f}s ± {timing_stats['std_hyperparameter_time']:.2f}s")
        print(f"  Average epsilon step: {timing_stats['mean_epsilon_step_time']:.3f}s ± {timing_stats['std_epsilon_step_time']:.3f}s")
        print(f"  Total epsilon steps: {timing_stats['total_epsilon_steps']}")
    
    # Add timing summary to results
    all_results['timing_summary'] = timing_summary
    
    # Log timing to wandb
    if not args.no_wandb:
        for method_name, timing_stats in timing_summary['method_timings'].items():
            wandb.log({
                f"timing/{method_name}/mean_hyperparameter_time": timing_stats['mean_hyperparameter_time'],
                f"timing/{method_name}/std_hyperparameter_time": timing_stats['std_hyperparameter_time'],
                f"timing/{method_name}/mean_epsilon_step_time": timing_stats['mean_epsilon_step_time'],
                f"timing/{method_name}/std_epsilon_step_time": timing_stats['std_epsilon_step_time'],
                f"timing/{method_name}/total_epsilon_steps": timing_stats['total_epsilon_steps']
            })
        
        # Create timing table for wandb
        timing_table_data = []
        for method_name, timing_stats in timing_summary['method_timings'].items():
            timing_table_data.append([
                method_name,
                f"{timing_stats['mean_hyperparameter_time']:.2f} ± {timing_stats['std_hyperparameter_time']:.2f}",
                f"{timing_stats['mean_epsilon_step_time']:.3f} ± {timing_stats['std_epsilon_step_time']:.3f}",
                timing_stats['total_epsilon_steps']
            ])
        
        timing_table = wandb.Table(
            columns=["Method", "Hyperparameter Tuning (s)", "Avg Epsilon Step (s)", "Total Epsilon Steps"],
            data=timing_table_data
        )
        wandb.log({"timing_summary_table": timing_table})

    # Create all types of plots
    print("\n=== Creating fold aggregation plots ===")
    for target_eps in cfg["target_epsilons"]:
        for method_name in cfg["methods"]:
            pu.plot_fold_robustness_vs_distance_aggregated(all_folds, target_eps, cfg["dataset"], cfg["model"]["cls"], method_name, vis_dir, use_wandb=(not args.no_wandb))
    
    print("\n=== Creating method comparison plots ===")
    for target_eps in cfg["target_epsilons"]:
        # Create validity comparison (average across all evaluation methods)
        pu.plot_validity_by_method(all_results, target_eps, cfg["dataset"], cfg["model"]["cls"], vis_dir, use_wandb=(not args.no_wandb))
        
        # Create validity comparison per evaluation method (retrain, awl, rob)
        pu.plot_validity_by_method_per_eval_method(all_results, target_eps, cfg["dataset"], cfg["model"]["cls"], vis_dir, use_wandb=(not args.no_wandb))
        
        # Log validity metrics to wandb
        pu.log_validity_metrics_to_wandb(all_results, target_eps, cfg["dataset"], cfg["model"]["cls"], use_wandb=(not args.no_wandb))
        
        # Create plots for each metric separately
        for metric in pu.ROB_METRICS:
            pu.plot_metric_vs_methods(all_results, target_eps, cfg["dataset"], cfg["model"]["cls"], metric, vis_dir, use_wandb=(not args.no_wandb))
            pu.plot_distance_vs_robustness_by_metric(all_results, target_eps, cfg["dataset"], cfg["model"]["cls"], metric, vis_dir, use_wandb=(not args.no_wandb))
    
    print("\n=== Creating multi-target-eps plots ===")
    for metric in pu.ROB_METRICS:
        pu.plot_robustness_vs_target_eps_by_metric(all_results, cfg["dataset"], cfg["model"]["cls"], metric, vis_dir, use_wandb=(not args.no_wandb))
        pu.plot_distance_vs_target_eps_by_metric(all_results, cfg["dataset"], cfg["model"]["cls"], metric, vis_dir, use_wandb=(not args.no_wandb))
    
    print("\n=== Creating timing plots ===")
    pu.plot_method_timing_comparison(all_folds, cfg["dataset"], cfg["model"]["cls"], vis_dir, use_wandb=(not args.no_wandb))
    pu.plot_epsilon_vs_time_by_method(all_folds, cfg["dataset"], cfg["model"]["cls"], vis_dir, use_wandb=(not args.no_wandb))

    # Save combined results
    combined_results_file = out_dir / "combined_results.json"
    with combined_results_file.open("w") as fp:
        json.dump(all_results, fp, indent=2)
    
    # Upload to GCS if in GCP mode
    if args.gcp:
        gcs_path = f"{args.job_id}/instance_{args.instance_id}"
        upload_to_gcs(args.gcs_bucket, gcs_path, combined_results_file)
        
        # Upload all visualizations
        upload_directory_to_gcs(args.gcs_bucket, f"{gcs_path}/visuals", vis_dir)
        
        logger.info(f"📤 Final results uploaded to gs://{args.gcs_bucket}/{gcs_path}")
    
    # Create publication-ready plots using the new plotting system
    print("\n=== Creating publication-ready plots ===")
    try:
        plotter = Plotting(vis_dir)
        
        # Create plots for this specific configuration
        json_paths = [str(out_dir / "combined_results.json")]
        dataset_names = [cfg["dataset"]]
        
        # Determine appropriate epsilon values for tradeoff plots
        if all_results:
            first_method = list(all_results.keys())[0]
            target_eps_list = sorted([float(eps) for eps in all_results[first_method].keys()])
            # Use middle epsilon for tradeoff plots
            target_eps = target_eps_list[len(target_eps_list) // 2] if target_eps_list else 0.05
        else:
            target_eps = 0.05
        
        # Create all available plots
        saved_plots = plotter.create_all_plots(
            json_paths=json_paths,
            dataset_names=dataset_names,
            eps_list=target_eps,
            model_name=cfg["model"]["cls"],
            method_display_names=None  # Use default display names
        )
        
        print(f"✅ Created {len(saved_plots)} publication-ready plots:")
        for plot_type, plot_path in saved_plots.items():
            print(f"  - {plot_type}: {plot_path}")
            
        # Log plots to WandB if enabled
        if not args.no_wandb:
            for plot_type, plot_path in saved_plots.items():
                if Path(plot_path).exists():
                    wandb.save(plot_path)
                    wandb.log({f"plots/{plot_type}": wandb.Image(plot_path)})
        
    except Exception as e:
        logger.warning(f"Failed to create publication plots: {e}")
        print(f"⚠️  Warning: Could not create publication plots: {e}")
    
    if not args.no_wandb:
        wandb.save(str(out_dir / "combined_results.json"))

# =============================================================================
#  Entry
# =============================================================================
if __name__ == "__main__":
    try:
        # Check if this is a unified config
        if (args.config.endswith("unified_config.yml") or 
            "unified_config" in args.config or 
            args.config.startswith("configs/ucct") or 
            "ucct" in args.config):
            # Handle unified configuration
            config_manager = ConfigManager(args.config)
            
            # List configs if requested
            if args.list_configs:
                config_manager.print_config_summary()
                exit(0)
            
            # Get configurations to execute based on mode
            configs_to_run = config_manager.execute_mode(
                mode=args.mode,
                range_str=args.range,
                specific_str=args.specific
            )
            
            print(f"Unified config execution mode: {args.mode}")
            print(f"Configurations to run: {len(configs_to_run)}")
            for i, (dataset, model_type, _) in enumerate(configs_to_run):
                print(f"  {i}: {dataset}:{model_type}")
            
            # Update wandb config for unified config mode
            if not args.no_wandb:
                wandb.config.update({"mode": args.mode, "total_configs": len(configs_to_run)})
            
            # Execute each configuration
            for i, (dataset, model_type, cfg) in enumerate(configs_to_run):
                print(f"\n{'='*60}")
                print(f"EXECUTING CONFIGURATION {i+1}/{len(configs_to_run)}: {dataset}:{model_type}")
                print(f"{'='*60}")
                
                # Create output directory for this specific config
                config_output_dir = out_dir / f"{dataset}_{model_type}"
                config_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Update output directories for this run
                vis_dir = config_output_dir / "visuals"
                fold_dir = config_output_dir / "fold_results"
                vis_dir.mkdir(parents=True, exist_ok=True)
                fold_dir.mkdir(parents=True, exist_ok=True)
                
                # Update logging to include config info
                logger.info(f"Starting execution for {dataset}:{model_type}")
                if args.only_train:
                    logger.info("Running in only_train mode - will skip counterfactual generation")
                else:
                    logger.info(f"Methods to run: {list(cfg['methods'].keys())}")
                
                # Log to WandB if enabled
                if not args.no_wandb:
                    import wandb
                    if wandb.run is not None:
                        wandb.log({"execution/start_time": time.time()})
                        wandb.log({"execution/dataset": dataset})
                        wandb.log({"execution/model_type": model_type})
                        wandb.log({"execution/config_index": i})
                        wandb.log({"execution/total_configs": len(configs_to_run)})
                        wandb.log({"execution/only_train_mode": args.only_train})
                        
                run_all(cfg, output_dir=config_output_dir, vis_dir=vis_dir, fold_dir=fold_dir)
                logger.info(f"Successfully completed {dataset}:{model_type}")
                
                # Upload config-specific results to GCS if in GCP mode
                if args.gcp:
                    gcs_path = f"{args.job_id}/instance_{args.instance_id}/{dataset}_{model_type}"
                    upload_directory_to_gcs(args.gcs_bucket, gcs_path, config_output_dir)
                    logger.info(f"📤 Config results uploaded to gs://{args.gcs_bucket}/{gcs_path}")
                
                # Log completion to WandB
                if not args.no_wandb:
                    import wandb
                    if wandb.run is not None:
                        wandb.log({"execution/completion_time": time.time()})
                        wandb.log({"execution/status": "completed"})
                try:
                    d = 0
                   
                            
                except Exception as e:
                    
                    logger.error(f"Failed to execute {dataset}:{model_type}: {e}")
                    
                    # Log failure to WandB
                    if not args.no_wandb:
                        import wandb
                        if wandb.run is not None:
                            wandb.log({"execution/completion_time": time.time()})
                            wandb.log({"execution/status": "failed"})
                            wandb.log({"execution/error": str(e)})
                    
                    # Continue with next config instead of failing completely
                    continue
            
            print(f"\n{'='*60}")
            print(f"COMPLETED ALL CONFIGURATIONS")
            print(f"{'='*60}")
            
            # Create aggregated plots across all configurations
            print("\n=== Creating aggregated plots across all configurations ===")
            try:
                # Collect all combined_results.json files
                json_paths = []
                dataset_names = []
                eps_lists = []
                
                for dataset, model_type, _ in configs_to_run:
                    config_output_dir = out_dir / f"{dataset}_{model_type}"
                    combined_results_path = config_output_dir / "combined_results.json"
                    
                    if combined_results_path.exists():
                        json_paths.append(str(combined_results_path))
                        dataset_names.append(dataset)
                        
                        # Load to get epsilon values
                        try:
                            with open(combined_results_path, 'r') as f:
                                data = json.load(f)
                            if data:
                                first_method = list(data.keys())[0]
                                target_eps_list = sorted([float(eps) for eps in data[first_method].keys()])
                                eps_lists.append(target_eps_list[len(target_eps_list) // 2] if target_eps_list else 0.05)
                            else:
                                eps_lists.append(0.05)
                        except Exception as e:
                            print(f"Warning: Could not load epsilon info for {dataset}: {e}")
                            eps_lists.append(0.05)
                
                if json_paths:
                    # Create aggregated plots
                    plotter = Plotting(out_dir / "aggregated_plots")
                    
                    # Determine if we should use single epsilon or multiple
                    unique_eps = len(set(eps_lists)) == 1
                    eps_for_plotting = eps_lists[0] if unique_eps else eps_lists
                    
                    # Create aggregated plots
                    saved_plots = plotter.create_all_plots(
                        json_paths=json_paths,
                        dataset_names=dataset_names,
                        eps_list=eps_for_plotting,
                        model_name="Multi-Dataset",
                        method_display_names=None
                    )
                    
                    print(f"✅ Created {len(saved_plots)} aggregated plots:")
                    for plot_type, plot_path in saved_plots.items():
                        print(f"  - {plot_type}: {plot_path}")
                        
                    # Log aggregated plots to WandB if enabled
                    if not args.no_wandb:
                        for plot_type, plot_path in saved_plots.items():
                            if Path(plot_path).exists():
                                wandb.save(plot_path)
                                wandb.log({f"aggregated_plots/{plot_type}": wandb.Image(plot_path)})
                else:
                    print("⚠️  No combined_results.json files found for aggregated plotting")
                    
            except Exception as e:
                logger.warning(f"Failed to create aggregated plots: {e}")
                print(f"⚠️  Warning: Could not create aggregated plots: {e}")
            
            # Note: wandb.finish() is now handled in the main finally block
            
        else:
            # Handle single configuration file (original behavior)
            cfg = yaml.safe_load(Path(args.config).read_text())
            print(f"Loaded config {args.config}, output ⇒ {out_dir}")
            print(f"Methods to run: {list(cfg['methods'].keys())}")
            
            # Update wandb run name and config for single config mode
            if not args.no_wandb:
                run_name = f"{cfg['dataset']}_{cfg['model']['cls']}_multimethod_{time.strftime('%Y%m%d_%H%M%S')}"
                wandb.run.name = run_name
                wandb.config.update(cfg)
                wandb.run.tags = [cfg['dataset'], cfg['model']['cls']]
            
            run_all(cfg)
            
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        raise
    finally:
        # Finish wandb run if it was initialized
        if not args.no_wandb and wandb.run is not None:
            wandb.finish()
        
        # Clean up temporary directory
        if TMP.exists():
            import shutil
            shutil.rmtree(TMP, ignore_errors=True)
            print(f"Cleaned up temp dir {TMP}")