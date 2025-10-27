import os, logging
from typing import Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# robustx imports
from robustx.generators.robust_CE_methods.EllipsoidCEBase import EllipsoidCEBase
from robustx.lib.tasks.Task import Task

from functools import lru_cache

from sklearn.neighbors import KDTree

from tqdm import tqdm

from sklearn.neighbors import BallTree

from typing import List, Tuple, Optional, Any, Dict, Union

from functools import lru_cache

import gurobipy as gp
from gurobipy import GRB
from sklearn.linear_model import LogisticRegression

from robustx.generators.CEGenerator import CEGenerator
from robustx.lib.tasks.ClassificationTask import ClassificationTask
# ------------------------------------------------------------------
__all__: List[str] = ["LastLayerEllipsoidCE"]


import torch

DEVICE = torch.device("cpu")

# DEVICE = torch.device(
#     "cuda" if torch.cuda.is_available()
#     else "mps"   if torch.backends.mps.is_available()
#     else "cpu"
# )

# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------
def safe_log1pexp(x: torch.Tensor) -> torch.Tensor:
    """Stable computation of log(1+exp(x)) that avoids overflow."""
    return torch.where(x > 20, x, torch.log1p(torch.exp(x)))


def _get_last_linear(model: nn.Module) -> nn.Linear:
    """Find the final nn.Linear layer in a model (assumed sequential)."""
    for module in reversed(list(model.modules())):
        if isinstance(module, nn.Linear):
            return module
    raise ValueError("No nn.Linear layer found in model")


def flatten_last_layer(model: nn.Module) -> torch.Tensor:
    """Flatten weight and bias of the final linear layer into a single parameter vector."""
    layer = _get_last_linear(model)
    return torch.cat([layer.weight.detach().flatten(), layer.bias.detach()])


def unflatten_last_layer(model: nn.Module, theta: torch.Tensor) -> None:
    """Write back a parameter vector into the final linear layer's weight and bias."""
    layer = _get_last_linear(model)
    num_w = layer.weight.numel()
    w_flat = theta[:num_w]
    b_flat = theta[num_w:]
    layer.weight.data.copy_(w_flat.view_as(layer.weight))
    layer.bias.data.copy_(b_flat.view_as(layer.bias))




class LastLayerEllipsoidCEOHCNT(EllipsoidCEBase):
    """
    Continuous gradient-based counterfactual explanation generator using ellipsoidal 
    approximation of the Rashomon set. This method generates robust counterfactuals
    by directly optimizing input features while ensuring robustness across
    the entire Rashomon set of models.
    
    Unlike data-supported methods, this approach can generate counterfactuals
    in continuous feature space, potentially finding more optimal solutions.
    """

    def __init__(
        self,
        task: Task,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        learning_rate: float = 0.1,
        max_iterations: int = 400,
        early_stopping: int = 100,
        robust_coef: float = 0.5,
        sparsity_coef: float = 0.0,
        proximity_coef: float = 0.1,
        optimizer: str = "adam",
        **params
    ) -> None:
        # Initialize base class with ellipsoidal logic
        super().__init__(task, device, dtype, **params)
        
        # Set optimization-specific parameters
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.early_stopping = early_stopping
        self.robust_coef = robust_coef
        self.sparsity_coef = sparsity_coef
        self.proximity_coef = proximity_coef
        self.optimizer = optimizer.lower()  # Normalize to lowercase


    def _optimize_counterfactual(self, x_orig: torch.Tensor) -> torch.Tensor:
        """
        Optimize a counterfactual example using gradient descent.
        This continuously updates the input to push it towards the target class
        while maintaining robustness against the worst-case models.
        """
        # Create a copy of the input that requires gradients
        x = x_orig.clone().to(self.device, self.dtype).requires_grad_(True)
        
        # Setup optimizer based on ellice_opt parameter
        if self.optimizer == "adam":
            optimizer = optim.Adam([x], lr=self.learning_rate)
        elif self.optimizer == "sgd":
            optimizer = optim.SGD([x], lr=self.learning_rate)
        else:
            # Default to Adam if unknown optimizer specified
            optimizer = optim.Adam([x], lr=self.learning_rate)
        
        # Target label is 1 - ensure shape is [1]
        target = torch.ones(1, device=self.device, dtype=self.dtype)
        
        # Keep track of worst models encountered
        worst_models = []
        
        # For early stopping
        best_robust_logit = float('-inf')
        best_x = x.clone().detach()
        no_improve_count = 0
        
        # Local copy of robust_coef for dynamic adjustment
        local_robust_coef = self.robust_coef
        
        # Local copy of prediction loss coefficient starting from 1
        local_pred_coef = 1.0
        
        # Optimization loop
        for iteration in range(self.max_iterations):
            optimizer.zero_grad()
            
            # Compute penultimate features - DIRECTLY USE TENSOR WITHOUT DETACHING
            h_flat = self._penult_features(x)
            bias = torch.ones(h_flat.size(0), 1, device=self.device, dtype=self.dtype)
            h_aug = torch.cat([h_flat, bias], dim=1)
            
            # 1. Prediction loss using current central model
            logit_center = torch.matmul(h_aug, self.omega_c)
            pred_loss = F.binary_cross_entropy_with_logits(logit_center, target)
            
            # 2. For robustness loss, we need to detach to compute worst case models
            # but create a new computation graph for the robust loss
            with torch.no_grad():
                # Compute worst-case model using detached input
                current_worst = self._compute_worst_model_from_h_aug(h_aug.detach())
                worst_models.append(current_worst)
                
                # Keep only a reasonable number of worst models
                if len(worst_models) > 1:
                    worst_models = worst_models[-1:]
            
            # 3. Robustness loss using all worst models encountered so far
            # These computations need to use the non-detached input to maintain gradients
            robust_logits = []
            for worst_model in worst_models:
                # Recompute h_flat to maintain gradient flow (if needed)
                # Or just use the already computed h_aug since it connects to x
                logit = torch.matmul(h_aug, worst_model)
                robust_logits.append(logit)
            
            # Compute the mean logit across all worst models
            robust_logit = torch.mean(torch.stack(robust_logits), dim=0)
            worst_logit = torch.min(torch.stack(robust_logits), dim=0)[0]
            robust_loss = F.binary_cross_entropy_with_logits(robust_logit, target)
           
            
            # Project back to feature bounds
            with torch.no_grad():
                x.data = torch.min(torch.max(x.data, self.feature_mins), self.feature_maxs)
            
            # Check if we've crossed the decision boundary with robustness
            with torch.no_grad():
                if robust_logit.item() > best_robust_logit:
                    best_robust_logit = robust_logit.item()
                    best_x = x.clone().detach()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    
            
            # Early stopping if no improvement for several iterations
            if worst_logit > 0 or no_improve_count >= self.early_stopping:
                #print(f"Early stopping at iteration {iteration+1}. Best robust logit: {best_robust_logit:.4f}")
                break
            
            
            total_loss = (
                local_pred_coef * pred_loss + # + 
                local_robust_coef * robust_loss #+#+ # + 
                #self.proximity_coef * proximity_loss 
                #self.sparsity_coef * sparsity_loss
            )
            
            total_loss.backward()
            optimizer.step()
            

        return best_x
        #return x.detach()

    def _generation_method(self, x: pd.Series, **kwargs: Any) -> pd.DataFrame:
        """Generate a counterfactual for the given input."""
        #print(f"Generating continuous counterfactual for input with shape: {x.shape}")
        
        # Extract ElliCE-specific parameters from kwargs, with fallback to instance defaults
        ellice_lr = kwargs.get('ellice_lr', self.learning_rate)
        ellice_robust_coef = kwargs.get('ellice_robust_coef', self.robust_coef)
        ellice_sparsity_coef = kwargs.get('ellice_sparsity_coef', self.sparsity_coef)
        ellice_proximity_coef = kwargs.get('ellice_proximity_coef', self.proximity_coef)
        ellice_max_iterations = kwargs.get('ellice_max_iterations', self.max_iterations)
        ellice_opt = kwargs.get('ellice_opt', self.optimizer)
        
        # Temporarily update instance parameters for this generation
        original_lr = self.learning_rate
        original_robust_coef = self.robust_coef
        original_sparsity_coef = self.sparsity_coef
        original_proximity_coef = self.proximity_coef
        original_max_iterations = self.max_iterations
        original_optimizer = self.optimizer
        
        self.learning_rate = ellice_lr
        self.robust_coef = ellice_robust_coef
        self.sparsity_coef = ellice_sparsity_coef
        self.proximity_coef = ellice_proximity_coef
        self.max_iterations = ellice_max_iterations
        self.optimizer = ellice_opt.lower()  # Normalize to lowercase
        
        # Convert pandas series to tensor
        x_tensor = torch.tensor(x.values, dtype=self.dtype, device=self.device).unsqueeze(0)
        
        # Check original prediction
        with torch.no_grad():
            h_flat = self._penult_features(x_tensor.cpu().numpy())
            bias = torch.ones(h_flat.size(0), 1, device=self.device, dtype=self.dtype)
            h_aug = torch.cat([h_flat, bias], dim=1)
            orig_logit = torch.matmul(h_aug, self.omega_c)
            orig_prob = torch.sigmoid(orig_logit).item()
        
        #print(f"Original prediction probability: {orig_prob:.4f}")
        
        # # If already predicted as class 1 with high confidence, return original
        # if orig_prob > 0.9:
        #     #print("Input already predicted as target class with high confidence.")
        #     return pd.DataFrame(x).T
        
        # Optimize the counterfactual
        cf_tensor = self._optimize_counterfactual(x_tensor)
        
        # Verify robustness of the counterfactual
        with torch.no_grad():
            h_flat = self._penult_features(cf_tensor.cpu().numpy())
            bias = torch.ones(h_flat.size(0), 1, device=self.device, dtype=self.dtype)
            h_aug = torch.cat([h_flat, bias], dim=1)
            
            # Check with worst model
            worst_theta = self._compute_worst_model_from_h_aug(h_aug)
            robust_logit = torch.matmul(h_aug, worst_theta)
            robust_prob = torch.sigmoid(robust_logit).item()
            
            # Check with center model
            center_logit = torch.matmul(h_aug, self.omega_c)
            center_prob = torch.sigmoid(center_logit).item()
        
        #print(f"Counterfactual robust prediction: {robust_prob:.4f}, center prediction: {center_prob:.4f}")
        #print(f"Feature changes: {(cf_tensor - x_tensor).abs().sum().item():.4f}")
        
        # Restore original parameters
        self.learning_rate = original_lr
        self.robust_coef = original_robust_coef
        self.sparsity_coef = original_sparsity_coef
        self.proximity_coef = original_proximity_coef
        self.max_iterations = original_max_iterations
        self.optimizer = original_optimizer
        
        # Convert to DataFrame
        cf_df = pd.DataFrame(cf_tensor.cpu().detach().numpy(), columns=x.index)
        return cf_df
        
    def getCandidates(self) -> pd.DataFrame:
        """
        This method is required by the interface but unused in the continuous version.
        In this implementation, counterfactuals are generated on-the-fly for each input.
        """
        # Return an empty DataFrame to indicate we don't pre-compute candidates
        return pd.DataFrame()

    def _generation_method(self, x: pd.Series, **kwargs: Any) -> pd.DataFrame:
        """Generate a counterfactual for the given input."""
        #print(f"Generating continuous counterfactual for input with shape: {x.shape}")
        
        # Extract ElliCE-specific parameters from kwargs, with fallback to instance defaults
        ellice_lr = kwargs.get('ellice_lr', self.learning_rate)
        ellice_robust_coef = kwargs.get('ellice_robust_coef', self.robust_coef)
        ellice_sparsity_coef = kwargs.get('ellice_sparsity_coef', self.sparsity_coef)
        ellice_proximity_coef = kwargs.get('ellice_proximity_coef', self.proximity_coef)
        ellice_max_iterations = kwargs.get('ellice_max_iterations', self.max_iterations)
        ellice_opt = kwargs.get('ellice_opt', self.optimizer)
        
        # Temporarily update instance parameters for this generation
        original_lr = self.learning_rate
        original_robust_coef = self.robust_coef
        original_sparsity_coef = self.sparsity_coef
        original_proximity_coef = self.proximity_coef
        original_max_iterations = self.max_iterations
        original_optimizer = self.optimizer
        
        self.learning_rate = ellice_lr
        self.robust_coef = ellice_robust_coef
        self.sparsity_coef = ellice_sparsity_coef
        self.proximity_coef = ellice_proximity_coef
        self.max_iterations = ellice_max_iterations
        self.optimizer = ellice_opt.lower()  # Normalize to lowercase
        
        # Convert pandas series to tensor
        x_tensor = torch.tensor(x.values, dtype=self.dtype, device=self.device).unsqueeze(0)
        
        # Check original prediction
        with torch.no_grad():
            h_flat = self._penult_features(x_tensor.cpu().numpy())
            bias = torch.ones(h_flat.size(0), 1, device=self.device, dtype=self.dtype)
            h_aug = torch.cat([h_flat, bias], dim=1)
            orig_logit = torch.matmul(h_aug, self.omega_c)
            orig_prob = torch.sigmoid(orig_logit).item()
        
        #print(f"Original prediction probability: {orig_prob:.4f}")
        
        # # If already predicted as class 1 with high confidence, return original
        # if orig_prob > 0.9:
        #     #print("Input already predicted as target class with high confidence.")
        #     return pd.DataFrame(x).T
        
        # Optimize the counterfactual
        cf_tensor = self._optimize_counterfactual(x_tensor)
        
        # Verify robustness of the counterfactual
        with torch.no_grad():
            h_flat = self._penult_features(cf_tensor.cpu().numpy())
            bias = torch.ones(h_flat.size(0), 1, device=self.device, dtype=self.dtype)
            h_aug = torch.cat([h_flat, bias], dim=1)
            
            # Check with worst model
            worst_theta = self._compute_worst_model_from_h_aug(h_aug)
            robust_logit = torch.matmul(h_aug, worst_theta)
            robust_prob = torch.sigmoid(robust_logit).item()
            
            # Check with center model
            center_logit = torch.matmul(h_aug, self.omega_c)
            center_prob = torch.sigmoid(center_logit).item()
        
        #print(f"Counterfactual robust prediction: {robust_prob:.4f}, center prediction: {center_prob:.4f}")
        #print(f"Feature changes: {(cf_tensor - x_tensor).abs().sum().item():.4f}")
        
        # Restore original parameters
        self.learning_rate = original_lr
        self.robust_coef = original_robust_coef
        self.sparsity_coef = original_sparsity_coef
        self.proximity_coef = original_proximity_coef
        self.max_iterations = original_max_iterations
        self.optimizer = original_optimizer
        
        # Convert to DataFrame
        cf_df = pd.DataFrame(cf_tensor.cpu().detach().numpy(), columns=x.index)
        return cf_df
        
    def getCandidates(self) -> pd.DataFrame:
        """
        This method is required by the interface but unused in the continuous version.
        In this implementation, counterfactuals are generated on-the-fly for each input.
        """
        # Return an empty DataFrame to indicate we don't pre-compute candidates
        return pd.DataFrame()
    
    
class BKTree:
    """
    A BK-Tree for fast generalized searching in a metric space.

    This tree is effective for finding items within a certain distance
    of a query item, given a distance function that satisfies the
    triangle inequality (e.g., Hamming distance, Levenshtein distance, L0/L1/L2 norms).
    The items themselves are stored in the tree nodes.
    """
    def __init__(self, dist_fn):
        """
        Initializes the BK-Tree.

        Args:
            dist_fn: A function that takes two items and returns an integer distance.
                     The distance must be a non-negative integer.
        """
        if not callable(dist_fn):
            raise TypeError("dist_fn must be a callable function.")
        self._root: Optional[Tuple[Any, Dict[int, Any]]] = None
        self.dist_fn = dist_fn
        self.size = 0

    def add(self, item: Any):
        """Adds an item to the tree."""
        if self._root is None:
            self._root = (item, {})
            self.size += 1
            return

        current_node = self._root
        while True:
            parent_item, children = current_node
            dist = self.dist_fn(item, parent_item)
            if dist == 0: # Item is already in the tree
                return

            if dist in children:
                current_node = children[dist]
            else:
                children[dist] = (item, {})
                self.size += 1
                break

    def search(self, query_item: Any, radius: int) -> List[Tuple[int, Any]]:
        """
        Finds all items in the tree within a given distance of the query item.

        Args:
            query_item: The item to search for.
            radius: The maximum distance (inclusive) from the query item.

        Returns:
            A list of tuples (distance, item) for all matches found.
        """
        if self._root is None:
            return []

        results: List[Tuple[int, Any]] = []
        # Use a stack for iterative search to avoid recursion depth issues
        stack: List[Tuple[Any, Dict[int, Any]]] = [self._root]

        while stack:
            current_item, children = stack.pop()
            dist_to_query = self.dist_fn(query_item, current_item)

            if dist_to_query <= radius:
                results.append((dist_to_query, current_item))

            # Pruning condition based on the triangle inequality:
            # We only need to explore children in the distance range
            # [dist_to_query - radius, dist_to_query + radius].
            min_dist = max(0, dist_to_query - radius)
            max_dist = dist_to_query + radius

            for d, child_node in children.items():
                if min_dist <= d <= max_dist:
                    stack.append(child_node)

        return results



class LastLayerEllipsoidCEOHC(EllipsoidCEBase):
    """Closed‑form last‑layer ellipsoidal Rashomon approximation with optional
    last‑layer re‑fitting via logistic regression and initial‑model fallback.
    """

    def __init__(
        self,
        task: Task,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        # kept for backwards‑compatibility
        ellipsoid_iters: int = 50,
        ellipsoid_lr: float = 1e-3,
        ellipsoid_C: float = 1000,
        ellipsoid_samples: int = 512 * 16,
        **params
    ) -> None:
        # Initialize base class with ellipsoidal logic
        super().__init__(task, device, dtype, **params)
        
        # Set specific parameters for this variant
        self.use_initial = bool(getattr(self.task, 'use_initial', True))
        
        # Compute penultimate features for data_support (for candidate generation)
        df_support = self.task.training_data.data
        X_support = df_support.drop(columns=[self.TARGET_COLUMN]).values.astype(np.float32)
        y_support = df_support[self.TARGET_COLUMN].values.astype(np.float32)
        
        with torch.no_grad():
            H_flat_support = self._penult_features(X_support)
            bias_support = torch.ones(H_flat_support.size(0), 1, device=self.device, dtype=self.dtype)
            H_aug_support = torch.cat([H_flat_support, bias_support], dim=1)
        H_feats_support = H_aug_support.cpu().numpy()
        
        # persist data support info for candidate generation
        self.y_signed_support = 2 * y_support - 1
        self.H_feats_support = H_feats_support
        
        # Compute base probabilities on data_support for candidate generation if refitted
        if self.refit:
            logits_support = H_feats_support @ self.theta_star.cpu().numpy()
            self._base_probs_support = 1.0 / (1.0 + np.exp(-logits_support))
        else:
            self._base_probs_support = None

        # diagnostics
        candidates = self.getCandidates()
        print(f"Candidate points: {len(candidates)} / {len(df_support)}")
        H_tensor_support = torch.from_numpy(self.H_feats_support).to(self.device, self.dtype)
        y_t_support = torch.from_numpy(self.y_signed_support).to(self.device, self.dtype)
        prec = self._measure_rashomon_precision(H_tensor_support, y_t_support, self.theta_threshold)
        print(f"Rashomon precision on data support: {prec:.2f}%")
        #self._print_parameter_ranges()


    @torch.no_grad()
    def _robust_logit_with_initial(
        self,
        h_aug: torch.Tensor,
        scaling_factor: float = 0.99,
        adaptation_rate: float = 0.99,
        max_attempts: int = 200,
    ) -> float:
        """Robust logit computation with use_initial parameter support."""
        base = float(torch.dot(self.omega_c, h_aug))
        inv_sqrt = self.Q_inv_sqrt
        # Use training data for validation
        Ht = torch.from_numpy(self.H_feats_train).to(self.device, self.dtype)
        ys = torch.from_numpy(self.y_signed_train).to(self.device, self.dtype)
        current_scaling = scaling_factor
        
        if self.use_initial:
            u = inv_sqrt @ h_aug
            direction = inv_sqrt @ u / (u.norm() / current_scaling)
            theta = self.omega_c - direction
            new_pred = float(theta @ h_aug)
            return new_pred
        
        for _ in range(max_attempts):
            u = inv_sqrt @ h_aug
            direction = inv_sqrt @ u / (u.norm() / current_scaling)
            theta = self.omega_c - direction
            new_pred = float(theta @ h_aug)
            loss = F.softplus(-ys * (Ht @ theta)).mean()
            if loss <= self.theta_threshold:
                return new_pred
            current_scaling *= adaptation_rate
        return base

    @lru_cache(maxsize=None)
    @torch.no_grad()
    def getCandidates(self) -> pd.DataFrame:
        # Process data_support for candidate generation
        print(f"Data support size: {self.task.data_support.data.shape[0]}")
        feats = self.task.data_support.data.drop(columns=[self.TARGET_COLUMN])
        Ht = torch.from_numpy(self.H_feats_support).to(self.device, self.dtype)
        logits = torch.tensor([
            self._robust_logit_with_initial(Ht[i]) for i in range(Ht.size(0))
        ], device=self.device, dtype=self.dtype)
        mask = (torch.sigmoid(logits) > 0.5).cpu().numpy()
        # Use either re-fitted probabilities or original model predictions on data_support
        base = self._base_probs_support if self._base_probs_support is not None else self.task.model.predict(feats).values.ravel()
        keep = mask & (base >= 0.5)
        return feats.iloc[keep].reset_index(drop=True)

    def _generation_method(self, x: pd.Series, **_: Any) -> pd.DataFrame:
        S = self.getCandidates()
        if S.empty:
            return pd.DataFrame(x).T
        tree = KDTree(S.values)
        idx = tree.query(x.values.reshape(1, -1), k=1)[1][0, 0]
        return S.iloc[[idx]]
    
def combined_hamming_l1_distance(x, y):
    """
    Combined metric: BIG_CONSTANT * hamming_distance + l1_distance
    
    This ensures hamming distance is prioritized, with L1 as tie-breaker.
    """
    # Hamming distance (L0 norm - count of differing elements)
    hamming_dist = np.sum(np.abs(x - y) > 1e-4)  # Use a small epsilon to avoid floating-point issues
    
    # L1 distance (Manhattan distance)
    l1_dist = np.sum(np.abs(x - y))
    
    # Choose BIG_CONSTANT larger than max possible L1 distance
    # For n features with values in [0,1], max L1 is n, so use 100*n to be safe
    BIG_CONSTANT = 100 #* len(x)
    
    return BIG_CONSTANT * hamming_dist + l1_dist


class LastLayerEllipsoidCEOHCBall(EllipsoidCEBase):
    """Closed‑form last‑layer ellipsoidal Rashomon approximation with optional
    last‑layer re‑fitting via logistic regression and initial‑model fallback.
    
    This version is updated to use a BallTree for efficient counterfactual search,
    prioritizing Hamming distance (L0) and using L1 distance as a tie-breaker.
    """

    def __init__(
        self,
        task: Task,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        # kept for backwards‑compatibility
        ellipsoid_iters: int = 50,
        ellipsoid_lr: float = 1e-3,
        ellipsoid_C: float = 1000,
        ellipsoid_samples: int = 512 * 16,
        **params
    ) -> None:
        # Initialize base class with ellipsoidal logic
        super().__init__(task, device, dtype, **params)
        
        # Set specific parameters for this variant
        self.use_initial = bool(getattr(self.task, 'use_initial', True))
        
        # Compute penultimate features for data_support (for candidate generation)
        df_support = self.task.data_support.data
        X_support = df_support.drop(columns=[self.TARGET_COLUMN]).values.astype(np.float32)
        y_support = df_support[self.TARGET_COLUMN].values.astype(np.float32)
        
        with torch.no_grad():
            H_flat_support = self._penult_features(X_support)
            bias_support = torch.ones(H_flat_support.size(0), 1, device=self.device, dtype=self.dtype)
            H_aug_support = torch.cat([H_flat_support, bias_support], dim=1)
        H_feats_support = H_aug_support.cpu().numpy()
        
        # persist data support info for candidate generation
        self.y_signed_support = 2 * y_support - 1
        self.H_feats_support = H_feats_support
        
        # Compute base probabilities on data_support for candidate generation if refitted
        if self.refit:
            logits_support = H_feats_support @ self.theta_star.cpu().numpy()
            self._base_probs_support = 1.0 / (1.0 + np.exp(-logits_support))
        else:
            self._base_probs_support = None

        # diagnostics
        candidates = self.getCandidates()
        print(f"Candidate points: {len(candidates)} / {len(df_support)}")
        H_tensor_support = torch.from_numpy(self.H_feats_support).to(self.device, self.dtype)
        y_t_support = torch.from_numpy(self.y_signed_support).to(self.device, self.dtype)
        prec = self._measure_rashomon_precision(H_tensor_support, y_t_support, self.theta_threshold)
        print(f"Rashomon precision on data support: {prec:.2f}%")
        #self._print_parameter_ranges()

    @torch.no_grad()
    def _robust_logit_with_initial(
        self,
        h_aug: torch.Tensor,
        scaling_factor: float = 0.99,
        adaptation_rate: float = 0.99,
        max_attempts: int = 200,
    ) -> float:
        """Robust logit computation with use_initial parameter support."""
        base = float(torch.dot(self.omega_c, h_aug))
        inv_sqrt = self.Q_inv_sqrt
        # Use training data for validation
        Ht = torch.from_numpy(self.H_feats_train).to(self.device, self.dtype)
        ys = torch.from_numpy(self.y_signed_train).to(self.device, self.dtype)
        current_scaling = scaling_factor
        for _ in range(max_attempts):
            u = inv_sqrt @ h_aug
            direction = inv_sqrt @ u / (u.norm() / current_scaling)
            theta = self.omega_c - direction
            new_pred = float(theta @ h_aug)
            if self.use_initial:
                return new_pred
            loss = F.softplus(-ys * (Ht @ theta)).mean()
            if loss <= self.theta_threshold:
                return new_pred
            current_scaling *= adaptation_rate
        return base

    @lru_cache(maxsize=None)
    @torch.no_grad()
    def getCandidates(self) -> pd.DataFrame:
        # Process data_support for candidate generation
        print(f"Data support size: {self.task.data_support.data.shape[0]}")
        feats = self.task.data_support.data.drop(columns=[self.TARGET_COLUMN])
        Ht = torch.from_numpy(self.H_feats_support).to(self.device, self.dtype)
        logits = torch.tensor([
            self._robust_logit_with_initial(Ht[i]) for i in range(Ht.size(0))
        ], device=self.device, dtype=self.dtype)
        mask = (torch.sigmoid(logits) > 0.5).cpu().numpy()
        # Use either re-fitted probabilities or original model predictions on data_support
        base = self._base_probs_support if self._base_probs_support is not None else self.task.model.predict(feats).values.ravel()
        keep = mask & (base >= 0.5)
        return feats.iloc[keep].reset_index(drop=True)

    @lru_cache(maxsize=None)
    def _get_candidate_balltree(self) -> Tuple[pd.DataFrame, Optional[BallTree]]:
        """
        Builds a BallTree on the candidate set using the combined Hamming-L1 distance.
        This method is cached to avoid rebuilding the tree on every call.
        """
        S = self.getCandidates()
        if S.empty:
            return S, None

        print(f"[ℹ] Building BallTree for combined Hamming-L1 distance on {len(S)} candidates...")
        
        # BallTree with custom metric
        tree = BallTree(S.values, metric=combined_hamming_l1_distance)
        
        print(f"[ℹ] BallTree built with {len(S)} candidates.")
        return S, tree

    def _generation_method(self, x: pd.Series, **_: Any) -> pd.DataFrame:
        """
        Finds the best counterfactual using BallTree with combined metric.
        The combined metric prioritizes Hamming distance (L0) with L1 as tie-breaker.
        """
        S, ball_tree = self._get_candidate_balltree()
        
        if ball_tree is None or S.empty:
            print("[WARN] Candidate set is empty. Returning original instance.")
            return pd.DataFrame(x).T

        x_array = x.values.reshape(1, -1)  # BallTree expects 2D array
        
        # Find the single nearest neighbor using combined metric
        distances, indices = ball_tree.query(x_array, k=1)
        
        best_idx = indices[0][0]
        best_cf = S.iloc[[best_idx]]
        
        # Optional: print debug info
        best_cf_array = best_cf.values[0]
        hamming_dist = np.sum(x.values != best_cf_array)
        l1_dist = np.sum(np.abs(x.values - best_cf_array))
        print(f"[DEBUG] Best CF: Hamming={hamming_dist}, L1={l1_dist:.3f}")
        
        return best_cf
