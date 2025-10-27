import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy
import numpy as np
from tqdm import tqdm
from robustx.lib.models.pytorch_models.SimpleNNModel import SimpleNNModel
import concurrent.futures as cf
import copy

def is_nam_model(model):
    """Check if a model is a NAM model by looking for feature_nns attribute."""
    return hasattr(model, 'feature_nns')

def process_model_output(output, model, apply_sigmoid=True):
    """
    Process model output to handle NAM vs linear model differences.
    
    Args:
        output: Raw model output (could be tuple for NAM, tensor for linear)
        model: The model that produced the output
        apply_sigmoid: Whether to apply sigmoid for NAM models
    
    Returns:
        Processed tensor ready for loss computation
    """
    # Handle NAM models that return tuple (logits, _) vs linear models that return single tensor
    if isinstance(output, tuple):
        output = output[0]  # Extract logits from tuple
    
    # Ensure consistent tensor shape for loss computation
    if output.dim() == 1:
        output = output.unsqueeze(1)  # Add dimension to match target shape
    
    # Apply sigmoid for NAM models (which return logits) to get probabilities for BCELoss
    if apply_sigmoid and is_nam_model(model):
        output = torch.sigmoid(output)
    
    return output
import os
from functools import partial
from typing import Dict, List, Tuple, Any

def compute_combined_loss(model, criterion, outputs, targets, l2_reg_coef=0.001):
    """
    Compute combined loss: data loss + L2 regularization (λ/2 * ||W||²).
    
    Args:
        model: PyTorch model
        criterion: Loss criterion (e.g., BCELoss)
        outputs: Model outputs
        targets: Target values
        l2_reg_coef: L2 regularization coefficient (λ)
    
    Returns:
        Combined loss tensor
    """
    data_loss = criterion(outputs, targets)
    
    # Compute L2 regularization term: λ/2 * ||W||²
    l2_reg = 0.0
    for param in model.parameters():
        l2_reg += torch.sum(param ** 2)
    l2_reg = (l2_reg_coef / 2.0) * l2_reg
    
    return data_loss + l2_reg


def get_available_cpus():
    """Get number of CPUs available to this process from SLURM allocation"""
    if 'SLURM_CPUS_PER_TASK' in os.environ:
        return int(os.environ['SLURM_CPUS_PER_TASK'])
    elif 'SLURM_JOB_CPUS_PER_NODE' in os.environ:
        return int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
    else:
        return os.cpu_count()

from tqdm import trange

import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import matplotlib.pyplot as plt

def generate_combined_rashomon_models(base_model, n_models=10, dropout_param=0.05, 
                                    max_loss=0.1, X_train=None, y_train=None,
                                    device=None, seed=42, combine_methods=True):
    """
    Generate models using combined dropout approaches for better Rashomon set coverage.
    
    This function combines both Bernoulli and Gaussian dropout to explore different
    regions of the Rashomon set, as suggested in the paper.
    
    Parameters: Same as generate_dropout_rashomon_models, except:
      - combine_methods: If True, uses both Bernoulli and Gaussian dropout
    
    Returns:
      - alt_models: List of alternative models from both dropout methods
    """
    if not combine_methods:
        return generate_dropout_rashomon_models(base_model, n_models, 'gaussian', 
                                              dropout_param, max_loss, X_train, y_train, device, seed)
    
    # Generate models with both dropout types
    n_per_method = n_models // 2
    
    print("Generating models with Gaussian dropout...")
    gaussian_models = generate_dropout_rashomon_models(
        base_model, n_per_method, 'gaussian', 
        dropout_param, max_loss, X_train, y_train, device, seed
    )
    
    print("Generating models with Bernoulli dropout...")
    bernoulli_models = generate_dropout_rashomon_models(
        base_model, n_models - n_per_method, 'bernoulli', 
        dropout_param * 2, max_loss, X_train, y_train, device, seed + 1
    )
    
    # Combine both sets
    all_models = gaussian_models + bernoulli_models
    print(f"\nTotal models generated: {len(all_models)} ({len(gaussian_models)} Gaussian + {len(bernoulli_models)} Bernoulli)")
    
    return all_models


def estimate_rashomon_bound_for_dropout(base_model, X_train, y_train, dropout_type='gaussian',
                                      dropout_params=None, n_samples=100, device=None, filter_loss_l2=0.001):
    """
    Estimate the expected loss deviation for different dropout parameters.
    
    This helps in selecting appropriate dropout parameters to achieve desired
    Rashomon bounds, as suggested in the theoretical analysis.
    
    Parameters:
      - base_model: The base model to analyze
      - X_train, y_train: Training data
      - dropout_type: 'bernoulli' or 'gaussian'
      - dropout_params: List of dropout parameters to test
      - n_samples: Number of samples per parameter
      - device: Torch device
    
    Returns:
      - results: Dictionary mapping dropout_param to (mean_loss_diff, std_loss_diff)
    """
    if device is None:
        device = torch.device("cpu")
    
    if dropout_params is None:
        if dropout_type.lower() == 'bernoulli':
            dropout_params = [0.01, 0.05, 0.1, 0.15, 0.2]
        else:
            dropout_params = [0.01, 0.05, 0.1, 0.15, 0.2]
    
    if not isinstance(X_train, torch.Tensor):
        X_tensor = torch.FloatTensor(X_train).to(device)
        y_tensor = torch.FloatTensor(y_train).view(-1, 1).to(device)
    else:
        X_tensor, y_tensor = X_train.to(device), y_train.view(-1, 1).to(device)
    
    criterion = nn.BCELoss()
    base_torch_model = base_model.get_torch_model().to(device)
    
    # Get base loss
    base_torch_model.eval()
    with torch.no_grad():
        base_output = base_torch_model(X_tensor)
        base_loss = compute_combined_loss(base_torch_model, criterion, base_output, y_tensor, filter_loss_l2).item()
    
    results = {}
    
    print(f"Analyzing loss deviations for {dropout_type} dropout...")
    print(f"Base model loss: {base_loss:.4f}")
    
    for param in tqdm(dropout_params, desc="Dropout parameters"):
        loss_diffs = []
        
        for _ in range(n_samples):
            # Create temporary model with dropout
            temp_model = deepcopy(base_torch_model)
            temp_model.eval()
            
            with torch.no_grad():
                for temp_param, base_param in zip(temp_model.parameters(), base_torch_model.parameters()):
                    if dropout_type.lower() == 'bernoulli':
                        dropout_mask = torch.bernoulli(torch.ones_like(base_param) * (1 - param))
                        temp_param.data.copy_(base_param.data * dropout_mask)
                    else:  # gaussian
                        noise = torch.normal(mean=1.0, std=param, size=base_param.shape, device=device)
                        temp_param.data.copy_(base_param.data * noise)
                
                # Calculate combined loss
                output = temp_model(X_tensor)
                loss = compute_combined_loss(temp_model, criterion, output, y_tensor, filter_loss_l2).item()
                loss_diffs.append(loss - base_loss)
        
        mean_diff = np.mean(loss_diffs)
        std_diff = np.std(loss_diffs)
        results[param] = (mean_diff, std_diff)
        
        print(f"Dropout {param}: Loss diff = {mean_diff:.4f} ± {std_diff:.4f}")
    
    return results

import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import matplotlib.pyplot as plt


def generate_dropout_rashomon_models(base_model, n_models=10, dropout_type='gaussian', 
                                   dropout_param=0.05, max_loss=0.1, X_train=None, y_train=None,
                                   device=None, seed=42, filter_loss_l2=0.001):
    """
    Generate models from the Rashomon set using dropout-based exploration 
    as described in the paper "Dropout-Based Rashomon Set Exploration".
    
    This is much faster than retraining as it only applies dropout at inference time.
    
    Parameters:
      - base_model: A RobustX-compatible model (e.g., SimpleNNModel).
      - n_models: Number of models to generate.
      - dropout_type: 'bernoulli' or 'gaussian' dropout.
      - dropout_param: Dropout rate p for Bernoulli or variance α for Gaussian.
      - max_loss: Maximum allowed training loss (Rashomon parameter).
      - X_train: Training features for loss validation.
      - y_train: Training labels for loss validation.
      - device: Torch device to use.
      - seed: Random seed for reproducibility.
    
    Returns:
      - alt_models: List of alternative models in the Rashomon set.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Prepare training data for loss validation
    if X_train is not None and y_train is not None:
        if not isinstance(X_train, torch.Tensor):
            X_tensor = torch.FloatTensor(X_train).to(device)
            y_tensor = torch.FloatTensor(y_train).view(-1, 1).to(device)
        else:
            X_tensor, y_tensor = X_train.to(device), y_train.view(-1, 1).to(device)
        criterion = nn.BCELoss()
    else:
        print("Warning: No training data provided. Models won't be validated against Rashomon bound.")
        X_tensor = y_tensor = criterion = None
    
    # Get base model
    base_torch_model = base_model.get_torch_model().to(device)
    
    alt_models = []
    print(f"Generating {n_models} models using {dropout_type} dropout...")
    
    for i in tqdm(range(n_models), desc="Dropout models"):
        # Create new model instance with proper handling for different model types
        if 'NamAdapter' in str(type(base_model)):
            # For NAM models, we need to create a NamAdapter wrapper with the EXACT same config as base_model
            from robustx.lib.models.pytorch_models.NamAdapter import NamAdapter
            
            # Get the exact configuration from the base model to ensure compatibility
            base_torch_model_for_config = base_model.get_torch_model()
            
            # Extract the actual configuration from the trained model
            if hasattr(base_torch_model_for_config, 'feature_nns'):
                # Get the actual number of basis functions from the trained model
                try:
                    # Method 1: Try to access the first layer's weight
                    first_layer = base_torch_model_for_config.feature_nns[0].model[0]
                    if hasattr(first_layer, 'weight'):
                        actual_num_basis = first_layer.weight.shape[1]
                    else:
                        # Method 2: Try to access through parameters
                        params = list(first_layer.parameters())
                        if params:
                            actual_num_basis = params[0].shape[1]
                        else:
                            actual_num_basis = getattr(base_model, 'num_basis_functions', 4)
                except (AttributeError, IndexError):
                    # Fallback to config value
                    actual_num_basis = getattr(base_model, 'num_basis_functions', 4)
            else:
                actual_num_basis = getattr(base_model, 'num_basis_functions', 4)
            
            new_wrapper = NamAdapter(
                input_dim=base_model.input_dim,
                output_dim=base_model.output_dim,
                num_basis_functions=actual_num_basis,  # Use the actual number from trained model
                activation=getattr(base_model, 'activation', 'relu'),
                dropout=getattr(base_model, 'dropout', 0.1),
                feature_dropout=getattr(base_model, 'feature_dropout', 0.0),
                l2_regularization=getattr(base_model, 'l2_regularization', 0.001),
                lr=getattr(base_model, 'lr', 0.001),
                batch_size=getattr(base_model, 'batch_size', 32),
                early_stopping=getattr(base_model, 'early_stopping', True),
                patience=getattr(base_model, 'patience', 50),
                seed=getattr(base_model, 'seed', None),
                device=device
            )
            new_torch_model = new_wrapper.get_torch_model().to(device)
            
            # Copy parameters from trained NAM model to new NAM wrapper
            with torch.no_grad():
                try:
                    new_torch_model.load_state_dict(base_torch_model.state_dict())
                except Exception as e:
                    print(f"Warning: Could not copy NAM parameters: {e}")
                    print("This is expected when NAM architectures differ - using random initialization")
        else:
            # For other model types (SimpleNNModel, etc.)
            new_wrapper = type(base_model)(
                input_dim=base_model.input_dim,
                hidden_dim=base_model.hidden_dim,
                output_dim=base_model.output_dim
            )
            new_torch_model = new_wrapper.get_torch_model().to(device)
        
        # Apply dropout to each layer
        with torch.no_grad():
            for (name1, param1), (name2, param2) in zip(base_torch_model.named_parameters(), 
                                                        new_torch_model.named_parameters()):
                # Generate dropout mask
                if dropout_type.lower() == 'bernoulli':
                    # Bernoulli dropout: randomly set weights to 0
                    dropout_mask = torch.bernoulli(torch.ones_like(param1) * (1 - dropout_param))
                    param2.data.copy_(param1.data * dropout_mask)
                elif dropout_type.lower() == 'gaussian':
                    # Gaussian dropout: multiply by Gaussian noise
                    noise = torch.normal(mean=1.0, std=dropout_param, size=param1.shape, device=device)
                    param2.data.copy_(param1.data * noise)
                else:
                    raise ValueError(f"Unknown dropout type: {dropout_type}")
        
        # Validate against Rashomon bound if training data is provided
        if X_tensor is not None and y_tensor is not None:
            new_torch_model.eval()
            with torch.no_grad():
                output = new_torch_model(X_tensor)
                output = process_model_output(output, new_torch_model, apply_sigmoid=True)
                loss = compute_combined_loss(new_torch_model, criterion, output, y_tensor, filter_loss_l2).item()
                
                if loss <= max_loss:
                    alt_models.append(new_wrapper)
                else:
                    # If loss exceeds bound, try with smaller dropout
                    if dropout_type.lower() == 'bernoulli':
                        reduced_param = dropout_param * 0.5
                        dropout_mask = torch.bernoulli(torch.ones_like(param1) * (1 - reduced_param))
                        for (name1, param1), (name2, param2) in zip(base_torch_model.named_parameters(), 
                                                                    new_torch_model.named_parameters()):
                            param2.data.copy_(param1.data * dropout_mask)
                    else:  # gaussian
                        reduced_param = dropout_param * 0.5
                        for (name1, param1), (name2, param2) in zip(base_torch_model.named_parameters(), 
                                                                    new_torch_model.named_parameters()):
                            noise = torch.normal(mean=1.0, std=reduced_param, size=param1.shape, device=device)
                            param2.data.copy_(param1.data * noise)
                    
                    # Re-validate
                    new_torch_model.eval()
                    with torch.no_grad():
                        output = new_torch_model(X_tensor)
                        output = process_model_output(output, new_torch_model, apply_sigmoid=True)
                        loss = compute_combined_loss(new_torch_model, criterion, output, y_tensor, filter_loss_l2).item()
                        if loss <= max_loss:
                            alt_models.append(new_wrapper)
        else:
            # No validation, just add the model
            alt_models.append(new_wrapper)
    
    print(f"Generated {len(alt_models)} models in the Rashomon set.\n")
    
    # Evaluate models if training data is available
    if X_tensor is not None and y_tensor is not None:
        print("Evaluating models on training set:")
        for idx, wrapper in enumerate(alt_models[:10], start=1):  # Show first 10
            tm = wrapper.get_torch_model().to(device).eval()
            with torch.no_grad():
                out = tm(X_tensor)
                out = process_model_output(out, tm, apply_sigmoid=True)
                loss_ = compute_combined_loss(tm, criterion, out, y_tensor, filter_loss_l2).item()
                preds = (out >= 0.5).float()
                acc = (preds.eq(y_tensor).float().mean().item())
            print(f"Model {idx}: loss={loss_:.4f}, acc={acc:.4f}")
        if len(alt_models) > 10:
            print(f"... and {len(alt_models) - 10} more models")
    
    return alt_models


# ────────────────────────────────────────────────────────────
def _single_dropout_model_worker(
        task_id: int,
        base_state: Dict[str, torch.Tensor],
        model_kwargs: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        dropout_type: str,
        dropout_param: float,
        max_loss: float,
        seed: int,
        device_str: str = "cpu",
        filter_loss_l2: float = 0.001
) -> Tuple[int, SimpleNNModel, float, float]:
    """
    Worker that creates ONE dropout model and returns:
        (task_id, model, loss, accuracy)
    """
    # Setup device for this worker
    device = torch.device(device_str)
    torch.set_num_threads(1)  # Each worker uses 1 BLAS thread
    
    # Set random seed for this worker
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Re-create model and load base weights
    wrapper = SimpleNNModel(**model_kwargs, device=device)
    torch_model = wrapper.get_torch_model().to(device)
    torch_model.load_state_dict(base_state)
    
    # Prepare data tensors
    if not isinstance(X_train, torch.Tensor):
        X_tensor = torch.FloatTensor(X_train).to(device)
        y_tensor = torch.FloatTensor(y_train).view(-1, 1).to(device)
    else:
        X_tensor, y_tensor = X_train.to(device), y_train.view(-1, 1).to(device)
    
    criterion = nn.BCELoss()
    
    # Apply dropout to each layer
    with torch.no_grad():
        for param in torch_model.parameters():
            if dropout_type.lower() == 'bernoulli':
                # Bernoulli dropout: randomly set weights to 0
                dropout_mask = torch.bernoulli(torch.ones_like(param) * (1 - dropout_param))
                param.data.copy_(param.data * dropout_mask)
            elif dropout_type.lower() == 'gaussian':
                # Gaussian dropout: multiply by Gaussian noise
                noise = torch.normal(mean=1.0, std=dropout_param, size=param.shape, device=device)
                param.data.copy_(param.data * noise)
            else:
                raise ValueError(f"Unknown dropout type: {dropout_type}")
    
    # Validate against Rashomon bound
    torch_model.eval()
    with torch.no_grad():
        output = torch_model(X_tensor)
        loss = compute_combined_loss(torch_model, criterion, output, y_tensor, filter_loss_l2).item()
        preds = (output >= 0.5).float()
        acc = (preds.eq(y_tensor).float().mean().item())
    
    # If loss exceeds bound, try with smaller dropout
    if loss > max_loss:
        # Reload base weights
        torch_model.load_state_dict(base_state)
        
        with torch.no_grad():
            for param in torch_model.parameters():
                if dropout_type.lower() == 'bernoulli':
                    reduced_param = dropout_param * 0.5
                    dropout_mask = torch.bernoulli(torch.ones_like(param) * (1 - reduced_param))
                    param.data.copy_(param.data * dropout_mask)
                else:  # gaussian
                    reduced_param = dropout_param * 0.5
                    noise = torch.normal(mean=1.0, std=reduced_param, size=param.shape, device=device)
                    param.data.copy_(param.data * noise)
        
        # Re-validate
        torch_model.eval()
        with torch.no_grad():
            output = torch_model(X_tensor)
            loss = compute_combined_loss(torch_model, criterion, output, y_tensor, filter_loss_l2).item()
            preds = (output >= 0.5).float()
            acc = (preds.eq(y_tensor).float().mean().item())
    
    return task_id, wrapper, loss, acc


# ────────────────────────────────────────────────────────────
def generate_dropout_rashomon_models_parallel(base_model, n_models=10, dropout_type='gaussian', 
                                            dropout_param=0.05, max_loss=0.1, X_train=None, y_train=None,
                                            device=None, seed=42, n_jobs=None, filter_loss_l2=0.001):
    """
    Parallel version of generate_dropout_rashomon_models.
    
    Parameters:
      - base_model: A RobustX-compatible model (e.g., SimpleNNModel).
      - n_models: Number of models to generate.
      - dropout_type: 'bernoulli' or 'gaussian' dropout.
      - dropout_param: Dropout rate p for Bernoulli or variance α for Gaussian.
      - max_loss: Maximum allowed training loss (Rashomon parameter).
      - X_train: Training features for loss validation.
      - y_train: Training labels for loss validation.
      - device: Torch device to use.
      - seed: Random seed for reproducibility.
      - n_jobs: Number of parallel jobs (defaults to available CPUs).
    
    Returns:
      - alt_models: List of alternative models in the Rashomon set.
    """
    if n_jobs is None:
        n_jobs = get_available_cpus()
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"[Dropout parallel] Generating {n_models} models using {dropout_type} dropout across {n_jobs} processes...")
    
    # Prepare training data for loss validation
    if X_train is not None and y_train is not None:
        if not isinstance(X_train, torch.Tensor):
            X_tensor = torch.FloatTensor(X_train).to(device)
            y_tensor = torch.FloatTensor(y_train).view(-1, 1).to(device)
        else:
            X_tensor, y_tensor = X_train.to(device), y_train.view(-1, 1).to(device)
        criterion = nn.BCELoss()
    else:
        print("Warning: No training data provided. Models won't be validated against Rashomon bound.")
        X_tensor = y_tensor = criterion = None
    
    # Get base model and freeze weights
    base_torch_model = base_model.get_torch_model().to(device)
    base_state = copy.deepcopy(base_torch_model.state_dict())
    
    # Prepare model kwargs for workers
    model_kwargs = dict(
        input_dim=base_model.input_dim,
        hidden_dim=base_model.hidden_dim,
        output_dim=base_model.output_dim,
    )
    
    # Create task list
    tasks = []
    for i in range(n_models):
        tasks.append({
            'task_id': i,
            'base_state': base_state,
            'model_kwargs': model_kwargs,
            'X_train': X_train,
            'y_train': y_train,
            'dropout_type': dropout_type,
            'dropout_param': dropout_param,
            'max_loss': max_loss,
            'seed': seed + i,  # Different seed for each model
            'device_str': "cpu",  # Use CPU for parallel workers
            'filter_loss_l2': filter_loss_l2
        })
    
    # Process tasks in parallel
    alt_models = []
    losses = []
    accuracies = []
    
    worker_fn = partial(_single_dropout_model_worker)
    
    with cf.ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = []
        for task in tasks:
            futures.append(ex.submit(worker_fn, **task))
        
        for f in tqdm(cf.as_completed(futures), total=len(futures), desc="Dropout-model-workers"):
            task_id, wrapper, loss, acc = f.result()
            
            # Only add models that meet the Rashomon bound
            if loss <= max_loss:
                alt_models.append(wrapper)
                losses.append(loss)
                accuracies.append(acc)
    
    print(f"[Dropout parallel] Generated {len(alt_models)} models in the Rashomon set.\n")
    
    # Evaluate models if training data is available
    if X_tensor is not None and y_tensor is not None and alt_models:
        print("Evaluating models on training set:")
        for idx, wrapper in enumerate(alt_models[:10], start=1):  # Show first 10
            tm = wrapper.get_torch_model().to(device).eval()
            with torch.no_grad():
                out = tm(X_tensor)
                loss_ = compute_combined_loss(tm, criterion, out, y_tensor, filter_loss_l2).item()
                preds = (out >= 0.5).float()
                acc = (preds.eq(y_tensor).float().mean().item())
            print(f"Model {idx}: loss={loss_:.4f}, acc={acc:.4f}")
        if len(alt_models) > 10:
            print(f"... and {len(alt_models) - 10} more models")
    
    return alt_models


def estimate_optimal_dropout_params(base_model, X_train, y_train, max_loss=0.1,
                                  dropout_type='gaussian', n_test_samples=250, device=None,
                                  plot_results=False, filter_loss_l2=0.001):
    """
    Automatically estimate optimal dropout parameters for achieving a Rashomon bound.
    
    Based on the theoretical analysis in the paper, we can estimate the dropout parameters
    that will keep models within the specified loss bound with high probability.
    
    Parameters:
      - base_model: The base model
      - X_train, y_train: Training data
      - max_loss: Maximum acceptable loss for Rashomon set (upper bound)
      - dropout_type: 'bernoulli' or 'gaussian'
      - n_test_samples: Number of samples for empirical validation
      - device: Torch device
      - plot_results: Whether to plot the estimation process
    
    Returns:
      - optimal_param: Estimated optimal dropout parameter
      - estimation_results: Dictionary with detailed results
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Prepare data
    if not isinstance(X_train, torch.Tensor):
        X_tensor = torch.FloatTensor(X_train).to(device)
        y_tensor = torch.FloatTensor(y_train).view(-1, 1).to(device)
    else:
        X_tensor, y_tensor = X_train.to(device), y_train.view(-1, 1).to(device)
    
    criterion = nn.BCELoss()
    base_torch_model = base_model.get_torch_model().to(device)
    
    # Get base loss
    base_torch_model.eval()
    with torch.no_grad():
        base_output = base_torch_model(X_tensor)
        base_output = process_model_output(base_output, base_torch_model, apply_sigmoid=True)
        base_loss = compute_combined_loss(base_torch_model, criterion, base_output, y_tensor, filter_loss_l2).item()
    
    print(f"Base model loss: {base_loss:.4f}")
    print(f"Rashomon bound (max_loss): {max_loss:.4f}")
    print(f"Allowable loss deviation: {max_loss - base_loss:.4f}")
    
    # Estimate model properties for theoretical bounds
    d = sum(p.numel() for p in base_torch_model.parameters())  # Total parameters
    print(f"Model has {d} total parameters")
    
    # Use theoretical guidance to set search range
    # From the paper: dropout parameters should scale as O(d^(-δ))
    if dropout_type.lower() == 'bernoulli':
        # For Bernoulli, search in reasonable range
        param_range = np.logspace(-4, np.log10(0.8), 20)  # From 0.0001 to 0.5
    else:  # gaussian
        # For Gaussian, search in smaller range for standard deviation
        param_range = np.logspace(-4, np.log10(0.8), 20)  # From 0.0001 to 0.2
    
    results = []
    
    print(f"\nTesting {len(param_range)} dropout parameter values...")
    
    for param in tqdm(param_range, desc="Testing parameters", disable=False, file=None):
        losses = []
        
        # Sample multiple models with this dropout parameter
        for _ in range(n_test_samples):
            # Create temporary model with dropout
            temp_model = deepcopy(base_torch_model)
            temp_model.eval()
            
            with torch.no_grad():
                for temp_param, base_param in zip(temp_model.parameters(), base_torch_model.parameters()):
                    if dropout_type.lower() == 'bernoulli':
                        dropout_mask = torch.bernoulli(torch.ones_like(base_param) * (1 - param))
                        temp_param.data.copy_(base_param.data * dropout_mask)
                    else:  # gaussian
                        noise = torch.normal(mean=1.0, std=param, size=base_param.shape, device=device)
                        temp_param.data.copy_(base_param.data * noise)
                
                # Calculate combined loss
                output = temp_model(X_tensor)
                output = process_model_output(output, temp_model, apply_sigmoid=True)
                loss = compute_combined_loss(temp_model, criterion, output, y_tensor, filter_loss_l2).item()
                losses.append(loss)
        
        mean_loss = np.mean(losses)
        std_loss = np.std(losses)
        
        # Calculate percentage of models within Rashomon bound
        in_rashomon = np.mean(np.array(losses) <= max_loss)
        
        results.append({
            'param': param,
            'mean_loss': mean_loss,
            'std_loss': std_loss,
            'min_loss': np.min(losses),
            'max_loss': np.max(losses),
            'in_rashomon_pct': in_rashomon,
            'mean_loss_diff': mean_loss - base_loss
        })
    
    # Find optimal parameter
    # Prefer parameters where most models are within Rashomon bound
    valid_params = [r for r in results if r['in_rashomon_pct'] > 0.05]  # At least 80% within bound
    
    if not valid_params:
        print("Warning: No parameters achieved 80% Rashomon bound compliance. Using best available.")
        valid_params = results
    
    # Among valid parameters, choose the one that maximizes Rashomon set membership
    # If tie, prefer lower mean loss (closer to base)
    optimal_result = valid_params[-1]#max(valid_params, key=lambda r: (r['in_rashomon_pct'], -abs(r['mean_loss'] - base_loss)))
    optimal_param = optimal_result['param']
    
    print(f"\nOptimal {dropout_type} dropout parameter: {optimal_param:.6f}")
    print(f"Expected loss: {optimal_result['mean_loss']:.4f} ± {optimal_result['std_loss']:.4f}")
    print(f"Percentage within Rashomon bound: {optimal_result['in_rashomon_pct']:.1%}")
    
    # Theoretical validation using paper's formulas
    if dropout_type.lower() == 'bernoulli':
        # From Proposition 1: E[loss_diff] = p(1-p) * w^T diag(X^T X) w
        # Simplified approximation for validation
        expected_loss_diff_theoretical = optimal_param * (1 - optimal_param) * 0.01  # Rough estimate
        print(f"Theoretical expected loss diff (approx): {expected_loss_diff_theoretical:.4f}")
        
        plt.tight_layout()
        plt.show()
    
    return optimal_param, {
        'optimal_param': optimal_param,
        'optimal_result': optimal_result,
        'all_results': results,
        'base_loss': base_loss,
        'max_loss': max_loss
    }


# ────────────────────────────────────────────────────────────
def _single_dropout_test_worker(
        task_id: int,
        param: float,
        base_state: Dict[str, torch.Tensor],
        model_kwargs: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        dropout_type: str,
        n_samples: int,
        device_str: str = "cpu",
        filter_loss_l2: float = 0.001
) -> Tuple[int, float, List[float]]:
    """
    Worker that tests ONE dropout parameter and returns:
        (task_id, param, losses_list)
    """
    # Setup device for this worker
    device = torch.device(device_str)
    torch.set_num_threads(1)  # Each worker uses 1 BLAS thread
    
    # Re-create model and load base weights
    wrapper = SimpleNNModel(**model_kwargs, device=device)
    torch_model = wrapper.get_torch_model().to(device)
    torch_model.load_state_dict(base_state)
    
    # Prepare data tensors
    if not isinstance(X_train, torch.Tensor):
        X_tensor = torch.FloatTensor(X_train).to(device)
        y_tensor = torch.FloatTensor(y_train).view(-1, 1).to(device)
    else:
        X_tensor, y_tensor = X_train.to(device), y_train.view(-1, 1).to(device)
    
    criterion = nn.BCELoss()
    losses = []
    
    # Sample multiple models with this dropout parameter
    for _ in range(n_samples):
        # Create temporary model with dropout
        temp_model = copy.deepcopy(torch_model)
        temp_model.eval()
        
        with torch.no_grad():
            for temp_param, base_param in zip(temp_model.parameters(), torch_model.parameters()):
                if dropout_type.lower() == 'bernoulli':
                    dropout_mask = torch.bernoulli(torch.ones_like(base_param) * (1 - param))
                    temp_param.data.copy_(base_param.data * dropout_mask)
                else:  # gaussian
                    noise = torch.normal(mean=1.0, std=param, size=base_param.shape, device=device)
                    temp_param.data.copy_(base_param.data * noise)
            
            # Calculate combined loss
            output = temp_model(X_tensor)
            loss = compute_combined_loss(temp_model, criterion, output, y_tensor, filter_loss_l2).item()
            losses.append(loss)
    
    return task_id, param, losses


# ────────────────────────────────────────────────────────────
def estimate_optimal_dropout_params_parallel(base_model, X_train, y_train, max_loss=0.1,
                                           dropout_type='gaussian', n_test_samples=1000, device=None,
                                           plot_results=False, n_jobs=None, filter_loss_l2=0.001):
    """
    Parallel version of estimate_optimal_dropout_params.
    
    Parameters:
      - base_model: The base model
      - X_train, y_train: Training data
      - max_loss: Maximum acceptable loss for Rashomon set (upper bound)
      - dropout_type: 'bernoulli' or 'gaussian'
      - n_test_samples: Number of samples for empirical validation
      - device: Torch device
      - plot_results: Whether to plot the estimation process
      - n_jobs: Number of parallel jobs (defaults to CPU count)
    
    Returns:
      - optimal_param: Estimated optimal dropout parameter
      - estimation_results: Dictionary with detailed results
    """
    if n_jobs is None:
        n_jobs = get_available_cpus()
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Prepare data
    if not isinstance(X_train, torch.Tensor):
        X_tensor = torch.FloatTensor(X_train).to(device)
        y_tensor = torch.FloatTensor(y_train).view(-1, 1).to(device)
    else:
        X_tensor, y_tensor = X_train.to(device), y_train.view(-1, 1).to(device)
    
    criterion = nn.BCELoss()
    base_torch_model = base_model.get_torch_model().to(device)
    
    # Get base loss
    base_torch_model.eval()
    with torch.no_grad():
        base_output = base_torch_model(X_tensor)
        base_loss = compute_combined_loss(base_torch_model, criterion, base_output, y_tensor, filter_loss_l2).item()
    
    print(f"[Dropout parallel] Base model loss: {base_loss:.4f}")
    print(f"[Dropout parallel] Rashomon bound (max_loss): {max_loss:.4f}")
    print(f"[Dropout parallel] Allowable loss deviation: {max_loss - base_loss:.4f}")
    
    # Estimate model properties for theoretical bounds
    d = sum(p.numel() for p in base_torch_model.parameters())  # Total parameters
    print(f"[Dropout parallel] Model has {d} total parameters")
    
    # Use theoretical guidance to set search range
    if dropout_type.lower() == 'bernoulli':
        param_range = np.logspace(-4, np.log10(0.8), 20)  # From 0.0001 to 0.5
    else:  # gaussian
        param_range = np.logspace(-4, np.log10(0.8), 20)  # From 0.0001 to 0.2
    
    print(f"[Dropout parallel] Testing {len(param_range)} dropout parameter values across {n_jobs} processes...")
    
    # Freeze base weights once
    base_state = copy.deepcopy(base_torch_model.state_dict())
    
    # Prepare model kwargs for workers
    model_kwargs = dict(
        input_dim=base_model.input_dim,
        hidden_dim=base_model.hidden_dim,
        output_dim=base_model.output_dim,
    )
    
    # Create task list
    tasks = []
    for i, param in enumerate(param_range):
        tasks.append({
            'task_id': i,
            'param': param,
            'base_state': base_state,
            'model_kwargs': model_kwargs,
            'X_train': X_train,
            'y_train': y_train,
            'dropout_type': dropout_type,
            'n_samples': n_test_samples,
            'device_str': "cpu",  # Use CPU for parallel workers
            'filter_loss_l2': filter_loss_l2
        })
    
    # Process tasks in parallel
    results = []
    worker_fn = partial(_single_dropout_test_worker)
    
    with cf.ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = []
        for task in tasks:
            futures.append(ex.submit(worker_fn, **task))
        
        for f in tqdm(cf.as_completed(futures), total=len(futures), desc="Dropout-param-workers"):
            task_id, param, losses = f.result()
            
            mean_loss = np.mean(losses)
            std_loss = np.std(losses)
            
            # Calculate percentage of models within Rashomon bound
            in_rashomon = np.mean(np.array(losses) <= max_loss)
            
            results.append({
                'param': param,
                'mean_loss': mean_loss,
                'std_loss': std_loss,
                'min_loss': np.min(losses),
                'max_loss': np.max(losses),
                'in_rashomon_pct': in_rashomon,
                'mean_loss_diff': mean_loss - base_loss
            })
    
    # Sort results by parameter value to maintain order
    results.sort(key=lambda x: x['param'])
    
    # Find optimal parameter
    valid_params = [r for r in results if r['in_rashomon_pct'] > 0.05]  # At least 5% within bound
    
    if not valid_params:
        print("[Dropout parallel] Warning: No parameters achieved 5% Rashomon bound compliance. Using best available.")
        valid_params = results
    
    # Among valid parameters, choose the one that maximizes Rashomon set membership
    optimal_result = valid_params[-1]
    optimal_param = optimal_result['param']
    
    print(f"\n[Dropout parallel] Optimal {dropout_type} dropout parameter: {optimal_param:.6f}")
    print(f"[Dropout parallel] Expected loss: {optimal_result['mean_loss']:.4f} ± {optimal_result['std_loss']:.4f}")
    print(f"[Dropout parallel] Percentage within Rashomon bound: {optimal_result['in_rashomon_pct']:.1%}")
    
    # Theoretical validation using paper's formulas
    if dropout_type.lower() == 'bernoulli':
        expected_loss_diff_theoretical = optimal_param * (1 - optimal_param) * 0.01  # Rough estimate
        print(f"[Dropout parallel] Theoretical expected loss diff (approx): {expected_loss_diff_theoretical:.4f}")
    
    return optimal_param, {
        'optimal_param': optimal_param,
        'optimal_result': optimal_result,
        'all_results': results,
        'base_loss': base_loss,
        'max_loss': max_loss
    }


def generate_rashomon_models_with_auto_dropout(base_model, X_train, y_train, config=None,
                                             max_loss=0.1, dropout_type='gaussian',
                                             device=None, X_val=None, y_val=None, filter_loss_l2=0.001):
    """
    Complete pipeline: estimate optimal dropout parameters and generate Rashomon set models.
    
    This function:
    1. Automatically estimates the best dropout parameters
    2. Generates n_models using those parameters
    3. Validates the results
    
    Parameters:
      - base_model: The base model
      - config: Configuration dictionary containing n_drop_models and n_test_samples
      - X_train, y_train: Training data
      - max_loss: Maximum acceptable loss for Rashomon set (upper bound)
      - dropout_type: 'bernoulli' or 'gaussian'
      - device: Torch device
      - X_val, y_val: Optional validation data for additional evaluation
    
    Returns:
      - models: List of generated models in Rashomon set
      - estimation_results: Results from parameter estimation
    """
    print("=== Automatic Dropout-based Rashomon Set Exploration ===\n")
    
    # Extract parameters from config
    n_models = config.get('n_drop_models', 20) if config is not None else 20
    n_test_samples = config.get('n_test_samples', 500) if config is not None else 500
    cpu_parallel = config.get('cpu_parallel', False) if config is not None else False
    
    # Step 1: Estimate optimal dropout parameters
    print("Step 1: Estimating optimal dropout parameters...")
    available_cpus = get_available_cpus()
    if cpu_parallel and available_cpus > 1:
        print(f"  Using parallel dropout parameter estimation with {available_cpus} CPUs")
        optimal_param, estimation_results = estimate_optimal_dropout_params_parallel(
            base_model, X_train, y_train, max_loss, dropout_type, n_test_samples, device,
            n_jobs=available_cpus, filter_loss_l2=filter_loss_l2
        )
    else:
        print("  Using sequential dropout parameter estimation")
        optimal_param, estimation_results = estimate_optimal_dropout_params(
            base_model, X_train, y_train, max_loss, dropout_type, n_test_samples, device,
            filter_loss_l2=filter_loss_l2
        )
    
    # Step 2: Generate models using optimal parameters
    print(f"\nStep 2: Generating {n_models} models using optimal dropout parameter {optimal_param:.6f}...")
    if cpu_parallel and available_cpus > 1:
        print(f"  Using parallel dropout model generation with {available_cpus} CPUs")
        models = generate_dropout_rashomon_models_parallel(
            base_model=base_model,
            n_models=n_models,
            dropout_type=dropout_type,
            dropout_param=optimal_param,
            max_loss=max_loss,
            X_train=X_train,
            y_train=y_train,
            device=device,
            n_jobs=available_cpus,
            filter_loss_l2=filter_loss_l2
        )
    else:
        print("  Using sequential dropout model generation")
        models = generate_dropout_rashomon_models(
            base_model=base_model,
            n_models=n_models,
            dropout_type=dropout_type,
            dropout_param=optimal_param,
            max_loss=max_loss,
            X_train=X_train,
            y_train=y_train,
            device=device,
            filter_loss_l2=filter_loss_l2
        )
    
    # Step 3: Validate results
    print(f"\nStep 3: Validation - Generated {len(models)} models in Rashomon set")
    
    if X_val is not None and y_val is not None:
        print("\nEvaluating on validation set:")
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if not isinstance(X_val, torch.Tensor):
            X_val_tensor = torch.FloatTensor(X_val).to(device)
            y_val_tensor = torch.FloatTensor(y_val).view(-1, 1).to(device)
        else:
            X_val_tensor, y_val_tensor = X_val.to(device), y_val.view(-1, 1).to(device)
        
        criterion = nn.BCELoss()
        val_losses = []
        val_accs = []
        
        for model in models[:5]:  # Show first 5 models
            torch_model = model.get_torch_model().to(device).eval()
            with torch.no_grad():
                output = torch_model(X_val_tensor)
                loss = compute_combined_loss(torch_model, criterion, output, y_val_tensor, filter_loss_l2).item()
                preds = (output >= 0.5).float()
                acc = (preds.eq(y_val_tensor).float().mean().item())
                val_losses.append(loss)
                val_accs.append(acc)
        
        print(f"Validation - Mean loss: {np.mean(val_losses):.4f} ± {np.std(val_losses):.4f}")
        print(f"Validation - Mean acc: {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}")
    
    return models, estimation_results


# Additional utility function for adaptive dropout estimation
def adaptive_dropout_estimation(base_model, X_train, y_train, max_loss=0.1,
                              initial_param=0.05, max_iterations=10, tolerance=0.01, device=None, filter_loss_l2=0.001):
    """
    Iteratively refine dropout parameter estimation to stay within Rashomon bound.
    
    Uses binary search-like approach to converge on optimal parameter.
    
    Parameters:
      - base_model: The base model
      - X_train, y_train: Training data
      - max_loss: Maximum acceptable loss (Rashomon bound)
      - initial_param: Starting dropout parameter value
      - max_iterations: Maximum number of refinement iterations
      - tolerance: Convergence tolerance for mean loss
      - device: Torch device
      
    Returns:
      - optimal_param: Converged dropout parameter
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Prepare data
    if not isinstance(X_train, torch.Tensor):
        X_tensor = torch.FloatTensor(X_train).to(device)
        y_tensor = torch.FloatTensor(y_train).view(-1, 1).to(device)
    else:
        X_tensor, y_tensor = X_train.to(device), y_train.view(-1, 1).to(device)
    
    criterion = nn.BCELoss()
    base_torch_model = base_model.get_torch_model().to(device)
    
    # Get base loss
    base_torch_model.eval()
    with torch.no_grad():
        base_output = base_torch_model(X_tensor)
        base_loss = compute_combined_loss(base_torch_model, criterion, base_output, y_tensor, filter_loss_l2).item()
    
    current_param = initial_param
    param_range = [0.001, 0.5]  # Search range
    
    print(f"Adaptive estimation starting with parameter {current_param}")
    print(f"Target: keep models within Rashomon bound {max_loss:.4f}")
    
    for iteration in range(max_iterations):
        # Test current parameter
        losses = []
        for _ in range(20):  # Sample fewer models for speed
            temp_model = deepcopy(base_torch_model)
            temp_model.eval()
            
            with torch.no_grad():
                for temp_param, base_param in zip(temp_model.parameters(), base_torch_model.parameters()):
                    noise = torch.normal(mean=1.0, std=current_param, size=base_param.shape, device=device)
                    temp_param.data.copy_(base_param.data * noise)
                
                output = temp_model(X_tensor)
                loss = compute_combined_loss(temp_model, criterion, output, y_tensor, filter_loss_l2).item()
                losses.append(loss)
        
        mean_loss = np.mean(losses)
        in_bound_pct = np.mean(np.array(losses) <= max_loss)
        
        print(f"Iteration {iteration + 1}: param={current_param:.6f}, mean_loss={mean_loss:.4f}, in_bound={in_bound_pct:.1%}")
        
        # Check convergence - aim for high percentage within bound
        if in_bound_pct >= 0.9:  # 90% of models within bound
            print(f"Converged! Optimal parameter: {current_param:.6f}")
            break
        
        # Adjust parameter using binary search logic
        if mean_loss > max_loss or in_bound_pct < 0.8:
            # Too many models exceed bound, reduce dropout
            param_range[1] = current_param
            current_param = (param_range[0] + param_range[1]) / 2
        else:
            # Models are within bound, we can increase dropout for more diversity
            param_range[0] = current_param
            current_param = (param_range[0] + param_range[1]) / 2
    
    return current_param



def generate_awp_models(base_model, X_train, y_train, max_loss=0.1, n_models=10, reg_coef=0.001, device=None,
                         retrain_each_time=False, train_both_sides=False, X_val=None, y_val=None, retrain_epochs=400,
                         retrain_seed_start=100, mlp_l2_reg=0.001, filter_loss_l2=0.001):
    """
    Generate models from the Rashomon set using Adversarial Weight Perturbation (AWP) with L2 regularization.
    Now uses validation loss for stopping criteria while training on train data.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure inputs are tensors on device
    if not isinstance(X_train, torch.Tensor):
        X_tensor = torch.FloatTensor(X_train).to(device)
        y_tensor = torch.FloatTensor(y_train).view(-1, 1).to(device)
    else:
        X_tensor, y_tensor = X_train.to(device), y_train.view(-1, 1).to(device)
    
    # IMPORTANT: Add validation data preparation
    if X_val is None or y_val is None:
        raise ValueError("X_val and y_val must be provided for validation stopping criteria")
    
    if not isinstance(X_val, torch.Tensor):
        X_val_tensor = torch.FloatTensor(X_val).to(device)
        y_val_tensor = torch.FloatTensor(y_val).view(-1, 1).to(device)
    else:
        X_val_tensor, y_val_tensor = X_val.to(device), y_val.view(-1, 1).to(device)
        
    # Criterion and base-model evaluation
    criterion = nn.BCELoss()
    rashomon_bound = max_loss

    alt_models = []
    
    # Calculate how many models to generate per point
    if train_both_sides:
        models_per_point = 2
        total_points = min(n_models, X_train.shape[0]) if n_models > 1 else 1
        actual_n_models = total_points * models_per_point
    else:
        models_per_point = 1
        total_points = min(n_models, X_train.shape[0])
        actual_n_models = total_points
    
    print(f"Generating {actual_n_models} alternative models using AWP...")
    print(f"Perturbation points: {total_points}, Models per point: {models_per_point}")

    for i in tqdm(range(total_points), desc="AWP points", disable=False, file=None):
        # Pick a slice of X for this run
        start = (max(X_tensor.size(0) // total_points, 1)) * i
        x_point = X_tensor[start : start + 1]
        
        # Determine targets to train for this point
        if train_both_sides:
            targets_to_train = [0.0, 1.0]
        else:
            # Get current prediction and set target as opposite
            with torch.no_grad():
                # Use base model to get prediction
                base_torch = base_model.get_torch_model().to(device)
                base_torch.eval()
                current_pred = base_torch(x_point)
                current_pred = process_model_output(current_pred, base_torch, apply_sigmoid=True)
                
                # Ensure we have a scalar for comparison
                if current_pred.dim() > 0:
                    current_pred = current_pred.squeeze()
                
                # Set target as opposite to current prediction
                targets_to_train = [0.0 if current_pred.item() > 0.5 else 1.0]
        
        # Generate models for each target
        for target_value in targets_to_train:
            # Initialize model
            if retrain_each_time:
                # Create and retrain a new model
                if X_val is None or y_val is None:
                    raise ValueError("X_val and y_val must be provided when retrain_each_time=True")
                
                # Create new model with different seed
                current_seed = retrain_seed_start + len(alt_models)
                model_wrapper = SimpleNNModel(
                    input_dim=base_model.input_dim,
                    hidden_dim=base_model.hidden_dim,
                    output_dim=base_model.output_dim,
                    seed=current_seed,
                    device=device,
                    mlp_l2_reg=mlp_l2_reg
                )
                
                # Retrain the model
                model_wrapper.train(X_train, y_train, X_val, y_val, epochs=retrain_epochs, not_numpy=False)
                torch_model = model_wrapper.get_torch_model().to(device)
            else:
                # Copy base model
                torch_model = deepcopy(base_model.get_torch_model()).to(device)
            
            # Setup optimizer
            #optimizer = optim.Adam(torch_model.parameters(), lr=0.01)
            optimizer = optim.SGD(torch_model.parameters(), lr=0.001, momentum=0.9)
            
            # Save initial weights
            prev_state = deepcopy(torch_model.state_dict())
            
            # Create target tensor
            target = torch.tensor([[target_value]], device=device)
            
            # Perturb weights to change prediction on x_point
            for step in range(retrain_epochs):
                # CHANGED: Check validation loss instead of training loss
                torch_model.eval()
                with torch.no_grad():
                    val_out = torch_model(X_val_tensor)  # Use validation data
                    val_out = process_model_output(val_out, torch_model, apply_sigmoid=True)
                    val_loss = compute_combined_loss(torch_model, criterion, val_out, y_val_tensor, filter_loss_l2).item()  # Validation loss with L2 reg
                
                if val_loss > rashomon_bound:
                    # Revert if validation loss exceeds bound
                    torch_model.load_state_dict(prev_state)
                    break
                
                prev_state = deepcopy(torch_model.state_dict())
                
                # One-step adversarial update on x_point (still training on train data)
                torch_model.train()
                optimizer.zero_grad()
                out = torch_model(x_point)  # This is still from training data
                out = process_model_output(out, torch_model, apply_sigmoid=True)
                awp_loss = nn.BCELoss()(out, target)
                
                # Use combined loss with L2 regularization for consistency
                loss = compute_combined_loss(torch_model, nn.BCELoss(), out, target, filter_loss_l2)
                
                loss.backward()
                optimizer.step()
            
            # Create appropriate wrapper based on model type
            if is_nam_model(torch_model):
                # For NAM models, we need to create a NamAdapter wrapper with the EXACT same config as base_model
                from robustx.lib.models.pytorch_models.NamAdapter import NamAdapter
                
                # Get the exact configuration from the base model to ensure compatibility
                base_torch_model = base_model.get_torch_model()
                
                # Extract the actual configuration from the trained model
                if hasattr(base_torch_model, 'feature_nns'):
                    # Get the actual number of basis functions from the trained model
                    # Try different ways to access the weight shape
                    try:
                        # Method 1: Try to access the first layer's weight
                        first_layer = base_torch_model.feature_nns[0].model[0]
                        if hasattr(first_layer, 'weight'):
                            actual_num_basis = first_layer.weight.shape[1]
                        else:
                            # Method 2: Try to access through parameters
                            params = list(first_layer.parameters())
                            if params:
                                actual_num_basis = params[0].shape[1]
                            else:
                                actual_num_basis = getattr(base_model, 'num_basis_functions', 4)
                    except (AttributeError, IndexError):
                        # Fallback to config value
                        actual_num_basis = getattr(base_model, 'num_basis_functions', 4)
                else:
                    actual_num_basis = getattr(base_model, 'num_basis_functions', 4)
                
                new_wrapper = NamAdapter(
                    input_dim=base_model.input_dim,
                    output_dim=base_model.output_dim,
                    num_basis_functions=actual_num_basis,  # Use the actual number from trained model
                    activation=getattr(base_model, 'activation', 'relu'),
                    dropout=getattr(base_model, 'dropout', 0.1),
                    feature_dropout=getattr(base_model, 'feature_dropout', 0.0),
                    l2_regularization=getattr(base_model, 'l2_regularization', 0.001),
                    lr=getattr(base_model, 'lr', 0.001),
                    batch_size=getattr(base_model, 'batch_size', 32),
                    early_stopping=getattr(base_model, 'early_stopping', True),
                    patience=getattr(base_model, 'patience', 50),
                    seed=getattr(base_model, 'seed', None),
                    device=device
                )
                new_torch = new_wrapper.get_torch_model().to(device)
                
                # Copy parameters from trained NAM model to new NAM wrapper
                with torch.no_grad():
                    try:
                        new_torch.load_state_dict(torch_model.state_dict())
                        print(f"Successfully copied NAM parameters with {actual_num_basis} basis functions")
                    except Exception as e:
                        print(f"Warning: Could not copy NAM parameters: {e}")
                        print("This is expected when NAM architectures differ - using random initialization")
            else:
                # For SimpleNNModel, create SimpleNNModel wrapper
                new_wrapper = SimpleNNModel(
                    input_dim=base_model.input_dim,
                    hidden_dim=base_model.hidden_dim,
                    output_dim=base_model.output_dim
                )
                new_torch = new_wrapper.get_torch_model().to(device)
                
                # Copy parameters from trained SimpleNNModel to new SimpleNNModel wrapper
                with torch.no_grad():
                    try:
                        for src, dst in zip(torch_model.parameters(), new_torch.parameters()):
                            if src.shape == dst.shape:
                                dst.data.copy_(src.data)
                            else:
                                print(f"Warning: Skipping parameter copy due to shape mismatch: {src.shape} vs {dst.shape}")
                                break
                    except Exception as e:
                        print(f"Warning: Could not copy SimpleNNModel parameters: {e}")
            
            alt_models.append(new_wrapper)
            
            # Keep reference to the last model for prediction in next iteration
            model = torch_model

    print(f"Generated {len(alt_models)} alternative models.\n")

    return alt_models


# ────────────────────────────────────────────────────────────
def _single_awp_worker(
        task_id        : int,
        base_state     : Dict[str, torch.Tensor],
        model_kwargs   : Dict[str, Any],
        X_train        : np.ndarray,
        y_train        : np.ndarray,
        X_val          : np.ndarray,
        y_val          : np.ndarray,
        x_row_idx      : int,
        target_value   : float,
        rashomon_bound : float,
        reg_coef       : float,
        retrain_seed   : int,
        retrain_epochs : int,
        mlp_l2_reg     : float,
        filter_loss_l2 : float,
        awp_steps      : int = 400,
) -> Tuple[int, Dict[str, torch.Tensor], float, float]:
    """
    Worker that builds ONE alternative model and returns:
        (task_id, state_dict, train_loss, train_acc)
    """
    # ----- setup ------------------------------------------------------------
    device = torch.device("cpu")             # stay on CPU inside workers
    torch.set_num_threads(1)                 # each worker uses 1 BLAS thread

    # Re-create wrapper & torch model
    wrapper = SimpleNNModel(**model_kwargs, seed=retrain_seed, device=device, mlp_l2_reg=mlp_l2_reg)
    torch_model = wrapper.get_torch_model().to(device)

    # Load base weights
    torch_model.load_state_dict(base_state)

    # Prepare tensors once
    X_tr   = torch.FloatTensor(X_train).to(device)
    y_tr   = torch.FloatTensor(y_train).view(-1, 1).to(device)
    X_val_ = torch.FloatTensor(X_val).to(device)
    y_val_ = torch.FloatTensor(y_val).view(-1, 1).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.SGD(torch_model.parameters(), lr=0.001, momentum=0.9)#optim.Adam(torch_model.parameters(), lr=0.001)

    # ------ AWP core (exactly your loop, but CPU) ---------------------------
    x_point = torch.FloatTensor(X_train[x_row_idx : x_row_idx + 1]).to(device)
    target  = torch.tensor([[target_value]], device=device)

    prev_state = copy.deepcopy(torch_model.state_dict())

    for step in range(awp_steps):
        # validation-based early stop (using combined loss with L2 regularization)
        torch_model.eval()
        with torch.no_grad():
            val_out = torch_model(X_val_)
            val_out = process_model_output(val_out, torch_model, apply_sigmoid=True)
            val_loss = compute_combined_loss(torch_model, criterion, val_out, y_val_, filter_loss_l2).item()
        if val_loss > rashomon_bound:
            torch_model.load_state_dict(prev_state)   # revert last step
            break

        prev_state = copy.deepcopy(torch_model.state_dict())

        # one adversarial update on x_point with L2 regularization
        torch_model.train()
        optimizer.zero_grad()
        
        # Compute combined loss
        out_point = torch_model(x_point)
        out_point = process_model_output(out_point, torch_model, apply_sigmoid=True)
        loss = compute_combined_loss(torch_model, criterion, out_point, target, filter_loss_l2)
        
        loss.backward()
        optimizer.step()

    # ------------- final train metrics (using combined loss with L2 regularization) -------------------------------------
    torch_model.eval()
    with torch.no_grad():
        out  = torch_model(X_tr)
        out = process_model_output(out, torch_model, apply_sigmoid=True)
        total_loss = compute_combined_loss(torch_model, criterion, out, y_tr, filter_loss_l2).item()
        
        acc  = ((out >= 0.5).float().eq(y_tr).float().mean().item())

    return task_id, torch_model.state_dict(), total_loss, acc


# ────────────────────────────────────────────────────────────
def generate_awp_models_parallel(
    base_model        : SimpleNNModel,
    X_train           : np.ndarray,
    y_train           : np.ndarray,
    X_val             : np.ndarray,
    y_val             : np.ndarray,
    *,
    max_loss          : float = 0.1,
    n_models          : int   = 10,
    reg_coef          : float = 0.001,        # kept for completeness (not used in current loop)
    retrain_each_time : bool  = True,
    train_both_sides  : bool  = False,
    retrain_epochs    : int   = 100,
    retrain_seed_start: int   = 100,
    mlp_l2_reg        : float = 0.001,
    filter_loss_l2    : float = 0.001,
    n_jobs            : int   = None,
) -> List[SimpleNNModel]:
    """
    Parallel CPU implementation – returns list of SimpleNNModel.
    """
    assert retrain_each_time, "parallel version currently expects retrain_each_time=True"
    
    if n_jobs is None:
        n_jobs = get_available_cpus()

    device_parent = torch.device("cpu")
    criterion     = nn.BCELoss().to(device_parent)

    # — base loss on train set (for information only) —
    if not isinstance(X_train, torch.Tensor):
        X_tr = torch.FloatTensor(X_train).to(device_parent)
        y_tr = torch.FloatTensor(y_train).view(-1, 1).to(device_parent)
    else:
        X_tr, y_tr = X_train.to(device_parent), y_train.view(-1, 1).to(device_parent)

    base_loss = compute_combined_loss(base_model.get_torch_model().to(device_parent), criterion, base_model.get_torch_model().to(device_parent)(X_tr), y_tr, filter_loss_l2).item()
    print(f"[AWP parallel] base-train-loss = {base_loss:.4f}   Rashomon bound = {max_loss:.4f}")

    # — decide how many models per point (same logic as sequential) —
    if train_both_sides:
        models_per_point = 2
        total_points     = min(n_models, X_train.shape[0]) if n_models > 1 else 1
        actual_n_models  = total_points * models_per_point
    else:
        models_per_point = 1
        total_points     = min(n_models, X_train.shape[0])
        actual_n_models  = total_points

    print(f"Spawning {actual_n_models} worker tasks across {n_jobs} processes…")

    # — pre-compute (row_idx, target_value) task list —
    tasks = []
    for i in range(total_points):
        row_idx = (max(X_train.shape[0] // total_points, 1)) * i
        if train_both_sides:
            tvals = [0.0, 1.0]
        else:
            # use base model to decide the "flip" target
            with torch.no_grad():
                pred = base_model.predict(X_train[row_idx : row_idx + 1])[0]
            tvals = [0.0 if pred.item() > 0.5 else 1.0]

        for t in tvals:
            tasks.append((row_idx, t))

    # make sure len(tasks) == actual_n_models
    assert len(tasks) == actual_n_models

    # — freeze base weights once —
    base_state = copy.deepcopy(base_model.get_torch_model().state_dict())

    # — kwargs to rebuild SimpleNNModel inside workers —
    model_kwargs = dict(
        input_dim  = base_model.input_dim,
        hidden_dim = base_model.hidden_dim,
        output_dim = base_model.output_dim,
    )

    # — parallel map —
    alt_states, train_losses, train_accs = [], [], []
    worker_fn = partial(
        _single_awp_worker,
        base_state     = base_state,
        model_kwargs   = model_kwargs,
        X_train        = X_train,
        y_train        = y_train,
        X_val          = X_val,
        y_val          = y_val,
        rashomon_bound = max_loss,
        reg_coef       = reg_coef,
        retrain_epochs = retrain_epochs,
        mlp_l2_reg     = mlp_l2_reg,
        filter_loss_l2 = filter_loss_l2,
    )

    with cf.ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = []
        for task_id, (row_idx, tval) in enumerate(tasks):
            futures.append(ex.submit(
                worker_fn,
                task_id          = task_id,
                x_row_idx        = row_idx,
                target_value     = tval,
                retrain_seed     = retrain_seed_start + task_id,
            ))

        for f in tqdm(cf.as_completed(futures), total=len(futures), desc="AWP-workers"):
            tid, state, tl, acc = f.result()
            alt_states.append(state)
            train_losses.append(tl)
            train_accs  .append(acc)

    # — wrap into SimpleNNModel objects (on parent device) —
    alternative_models: List[SimpleNNModel] = []
    for idx, sd in enumerate(alt_states):
        w = SimpleNNModel(**model_kwargs)
        w.get_torch_model().load_state_dict(sd)
        alternative_models.append(w)
        #print(f"✓ model {idx+1:2d}: train-loss {train_losses[idx]:.4f}, acc {train_accs[idx]:.4f}")

    print(f"[AWP parallel] generated {len(alternative_models)} alternative models.")
    return alternative_models



def generate_random_retrained_models(X_train, y_train, X_val, y_val, hidden_dims, n_models=10, epochs=100, max_loss=0.1, start_seed=100, loss_tol=0.05, dropout=0.0, mlp_l2_reg=0.001, filter_loss_l2=0.001):
    """
    Generate randomly initialized and trained models that belong to the Rashomon set
    (i.e., models with training loss within loss_tol of the best model).
    
    Parameters:
      - X_train: Training features.
      - y_train: Training labels.
      - hidden_dims: List of hidden layer dimensions.
      - n_models: Number of models to generate.
      - epochs: Number of training epochs.
      - loss_tol: Allowed increase over the optimal loss to be in the Rashomon set.
      
    Returns:
      - List of models within the Rashomon set.
    """
    models = []
    filtered_models = []
    input_dim = X_train.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    criterion = nn.BCELoss().to(device)
    
    # # Ensure inputs are tensors on device
    if not isinstance(X_train, torch.Tensor):
        X_tensor = torch.FloatTensor(X_train.values).to(device)
        y_tensor = torch.FloatTensor(y_train.values).view(-1, 1).to(device)
    else:
        X_tensor, y_tensor = X_train.to(device), y_train.view(-1, 1).to(device)

    
    # Set Rashomon bound
    rashomon_bound = max_loss#base_loss + loss_tol
    print(f"Base model loss: {max_loss:.4f}")#, Rashomon bound: {rashomon_bound:.4f}")
    
    # We might need to generate more models than requested to ensure we get enough within the bound
    max_attempts = n_models * 2  # Generate up to 3 times as many models to filter
    attempts = 0
    
    print(f"Training models with loss filtering (target: {n_models} models)...")
    pbar = tqdm(total=n_models)
    losses = []
    accuracies = []
    
    while len(filtered_models) < n_models and attempts < max_attempts:
        model = SimpleNNModel(input_dim, hidden_dims, 1, seed=start_seed + attempts+42, device=device, dropout=dropout, mlp_l2_reg=mlp_l2_reg)
        model.train(X_train, y_train, X_val, y_val, epochs=epochs)#, desired_loss=rashomon_bound)
        
        # Evaluate model loss to check if it's within the Rashomon bound
        torch_model = model.get_torch_model().to(device)
        torch_model.eval()
        
        with torch.no_grad():
            output = torch_model(X_tensor)
            total_loss = compute_combined_loss(torch_model, criterion, output, y_tensor, filter_loss_l2).item()
            
            acc = model.compute_accuracy(X_train.values, y_train.values)
        
        # Check if model is within the Rashomon set (using total loss with L2 regularization)
        if total_loss <= rashomon_bound:
            filtered_models.append(model)
            pbar.update(1)
            #print(f"  Accepted model {len(filtered_models)}: loss = {total_loss:.4f}, accuracy = {acc:.4f}")
            losses.append(total_loss)
            accuracies.append(acc)
        else:
            pass
            #print(f"  Rejected model: loss = {loss:.4f} > {rashomon_bound:.4f}")
        
        attempts += 1
       
    print("first 10 losses: ", losses[:10])
    print("first 10 accuracies: ", accuracies[:10])
    
    pbar.close()
    
    if len(filtered_models) < n_models:
        print(f"Warning: Could only find {len(filtered_models)} models within the Rashomon bound after {attempts} attempts.")
    
    print(f"Generated {len(filtered_models)} models within the Rashomon bound.")
    return filtered_models


# ────────────────────────────────────────────────────────────
def _single_retrained_worker(
        task_id: int,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        input_dim: int,
        hidden_dims: List[int],
        epochs: int,
        seed: int,
        dropout: float,
        mlp_l2_reg: float,
        filter_loss_l2: float,
        device_str: str = "cpu"
) -> Tuple[int, SimpleNNModel, float, float]:
    """
    Worker that trains ONE model and returns:
        (task_id, model, loss, accuracy)
    """
    # Setup device for this worker
    device = torch.device(device_str)
    torch.set_num_threads(1)  # Each worker uses 1 BLAS thread
    
    # Create and train model
    model = SimpleNNModel(input_dim, hidden_dims, 1, seed=seed, device=device, dropout=dropout, mlp_l2_reg=mlp_l2_reg)
    model.train(X_train, y_train, X_val, y_val, epochs=epochs)
    
    # Evaluate model
    torch_model = model.get_torch_model().to(device)
    torch_model.eval()
    
    criterion = nn.BCELoss().to(device)
    
    # Prepare data tensors
    if not isinstance(X_train, torch.Tensor):
        X_tensor = torch.FloatTensor(X_train.values).to(device)
        y_tensor = torch.FloatTensor(y_train.values).view(-1, 1).to(device)
    else:
        X_tensor, y_tensor = X_train.to(device), y_train.view(-1, 1).to(device)
    
    with torch.no_grad():
        output = torch_model(X_tensor)
        total_loss = compute_combined_loss(torch_model, criterion, output, y_tensor, filter_loss_l2).item()
        
        acc = model.compute_accuracy(X_train.values, y_train.values)
    
    return task_id, model, total_loss, acc


# ────────────────────────────────────────────────────────────
def generate_random_retrained_models_parallel(
    X_train, y_train, X_val, y_val, hidden_dims, n_models=10, epochs=100, 
    max_loss=0.1, start_seed=100, loss_tol=0.05, dropout=0.0, mlp_l2_reg=0.001, filter_loss_l2=0.001, n_jobs=None
) -> List[SimpleNNModel]:
    """
    Parallel version of generate_random_retrained_models.
    
    Parameters:
      - X_train: Training features.
      - y_train: Training labels.
      - X_val: Validation features.
      - y_val: Validation labels.
      - hidden_dims: List of hidden layer dimensions.
      - n_models: Number of models to generate.
      - epochs: Number of training epochs.
      - max_loss: Maximum allowed loss (Rashomon bound).
      - start_seed: Starting seed for random initialization.
      - loss_tol: Allowed increase over the optimal loss (unused in parallel version).
      - dropout: Dropout rate.
      - n_jobs: Number of parallel jobs (defaults to CPU count).
      
    Returns:
      - List of models within the Rashomon bound.
    """
    if n_jobs is None:
        n_jobs = get_available_cpus()
    
    input_dim = X_train.shape[1]
    rashomon_bound = max_loss
    
    print(f"[Retrained parallel] Target: {n_models} models, Rashomon bound: {rashomon_bound:.4f}")
    print(f"Spawning training jobs across {n_jobs} processes...")
    
    # Generate more models than needed to account for filtering
    # Use a more conservative multiplier for parallel version
    total_models_to_train = max(int(n_models * 1.2), n_models + 50)  # At least 10 extra models
    
    # Create task list
    tasks = []
    for i in range(total_models_to_train):
        tasks.append({
            'task_id': i,
            'seed': start_seed + i + 42,
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'input_dim': input_dim,
            'hidden_dims': hidden_dims,
            'epochs': epochs,
            'dropout': dropout,
            'mlp_l2_reg': mlp_l2_reg,
            'filter_loss_l2': filter_loss_l2,
            'device_str': "cpu"  # Use CPU for parallel workers
        })
    
    # Train models in parallel
    trained_models = []
    losses = []
    accuracies = []
    
    worker_fn = partial(_single_retrained_worker)
    
    with cf.ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = []
        for task in tasks:
            futures.append(ex.submit(worker_fn, **task))
        
        for f in tqdm(cf.as_completed(futures), total=len(futures), desc="Retrained-workers"):
            task_id, model, loss, acc = f.result()
            trained_models.append((model, loss, acc))
            losses.append(loss)
            accuracies.append(acc)
    
    # Filter models based on Rashomon bound
    filtered_models = []
    for model, loss, acc in trained_models:
        if loss <= rashomon_bound:
            filtered_models.append(model)
            if len(filtered_models) >= n_models:
                break
    
    print(f"[Retrained parallel] Generated {len(filtered_models)} models within Rashomon bound")
    print(f"First 10 losses: {losses[:10]}")
    print(f"First 10 accuracies: {accuracies[:10]}")
    
    if len(filtered_models) < n_models:
        print(f"Warning: Could only find {len(filtered_models)} models within the Rashomon bound")
    
    return filtered_models