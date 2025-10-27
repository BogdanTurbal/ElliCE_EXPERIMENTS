
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NAM-specific Ellipsoidal Counterfactual Explanation methods.

This module provides NAM-specific implementations of LastLayerEllipsoidCE methods
that handle the unique architecture of Neural Additive Models (NAMs).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from functools import lru_cache
from typing import Optional, List, Any, Dict, Union
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KDTree, BallTree

from robustx.generators.CEGenerator import CEGenerator
from robustx.lib.tasks.Task import Task
from robustx.generators.robust_CE_methods.EllipsoidCEBase import EllipsoidCEBase
from robustx.lib.models.pytorch_models.NamAdapter import NAMPenult, NAMFreeHead, extract_nam_bias

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def combined_hamming_l1_distance(x, y):
    """
    Combined metric: BIG_CONSTANT * hamming_distance + l1_distance
    This prioritizes Hamming distance (L0) with L1 as tie-breaker.
    """
    BIG_CONSTANT = 1000.0
    
    # Hamming distance (L0)
    hamming_dist = np.sum(x != y)
    
    # L1 distance
    l1_dist = np.sum(np.abs(x - y))
    
    return BIG_CONSTANT * hamming_dist + l1_dist

# --------------------------------------------------------------------------- #
# NAM-specific Ellipsoidal CE Classes
# --------------------------------------------------------------------------- #

class LastLayerEllipsoidCEOHCNam(EllipsoidCEBase):
    """
    NAM-specific implementation of LastLayerEllipsoidCEOHC (data-supported).
    
    This class extends EllipsoidCEBase to handle NAM models by:
    - Using NAMPenult for penultimate feature extraction
    - Using NAMFreeHead for last layer parameter handling
    - Supporting data-supported counterfactual generation with KDTree search
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
        
        # Store support data for candidate generation
        self.X_support = X_support
        self.y_support = y_support
        self.H_feats_support = H_feats_support
        
    def _split_model(self, model: nn.Module) -> tuple[nn.Module, torch.Tensor]:
        """
        Split NAM model into penultimate feature extractor and last layer parameters.
        
        For NAMs, this involves:
        1. Creating a NAMPenult module to extract per-feature contributions
        2. Extracting the bias and constructing theta_star (last layer parameters)
        """
        # Get input dimension from task data if not already set
        if not hasattr(self, 'input_dim'):
            df_train = self.task.training_data.data
            X_train = df_train.drop(columns=[self.TARGET_COLUMN]).values.astype(np.float32)
            self.input_dim = X_train.shape[1]
        
        # Check if this is a NAM model
        if hasattr(model, "feature_nns"):
            # Create NAM penultimate feature extractor
            penult = NAMPenult(model)
            
            # Extract NAM bias
            try:
                bias = extract_nam_bias(model)
            except AttributeError:
                # Fallback: create a zero bias
                bias = torch.zeros(1, device=self.device, dtype=self.dtype)
            
            # Create NAMFreeHead to handle last layer parameters
            # Get input dimension from task data if not already set
            if not hasattr(self, 'input_dim'):
                df_train = self.task.training_data.data
                X_train = df_train.drop(columns=[self.TARGET_COLUMN]).values.astype(np.float32)
                self.input_dim = X_train.shape[1]
            
            free_head = NAMFreeHead(
                num_features=self.input_dim,
                bias_init=bias,
                device=self.device,
                dtype=self.dtype
            )
            
            # Construct theta_star (last layer parameters)
            theta_star = free_head.pack_theta()
            
            return penult, theta_star
        else:
            # Fallback to standard model splitting
            return super()._split_model(model)
    
    def _penult_features(self, X: torch.Tensor) -> torch.Tensor:
        """Extract penultimate features using NAM-specific logic."""
        # Ensure X is a PyTorch tensor
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, device=self.device, dtype=self.dtype)
        
        if hasattr(self.penult, 'feature_nns'):
            # NAM-specific feature extraction
            features = self.penult(X)
            # Flatten the features to 2D if needed (NAM returns per-feature contributions)
            if features.dim() == 3:
                features = features.view(features.size(0), -1)  # Flatten to (batch_size, features)
            return features
        else:
            # Standard feature extraction
            return super()._penult_features(X)
    
    def _robust_logit(self, X: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        Compute robust logit for NAM models.
        
        This method handles the NAM-specific logit computation using
        the penultimate features and theta parameters.
        """
        if hasattr(self.penult, 'feature_nns'):
            # NAM-specific robust logit computation
            h_flat = self._penult_features(X)
            bias = torch.ones(h_flat.size(0), 1, device=self.device, dtype=self.dtype)
            h_aug = torch.cat([h_flat, bias], dim=1)
            logits = torch.matmul(h_aug, theta)
            return logits
        else:
            # Standard robust logit computation
            return super()._robust_logit(X, theta)
    
    def _robust_emp_logit(self, X: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        Compute robust empirical logit for NAM models.
        
        This method filters models based on use_initial parameter and computes
        the empirical logit using the filtered model set.
        """
        if hasattr(self.penult, 'feature_nns'):
            # NAM-specific empirical logit computation
            if self.use_initial:
                # Use only the initial model (theta_star)
                return self._robust_logit(X, self.theta_star)
            else:
                # Use all models in the Rashomon set: compute h_aug, then delegate to base
                h_flat = self._penult_features(X)
                bias = torch.ones(h_flat.size(0), 1, device=self.device, dtype=self.dtype)
                h_aug = torch.cat([h_flat, bias], dim=1)
                h_aug_vec = h_aug[0] if h_aug.dim() > 1 else h_aug
                return super()._robust_emp_logit(h_aug_vec)
        else:
            # Standard empirical logit computation
            return super()._robust_emp_logit(X, theta)
    
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
        """Generate candidate counterfactuals from the support data."""
        # Process data_support for candidate generation
        print(f"Data support size: {self.task.data_support.data.shape[0]}")
        feats = self.task.data_support.data.drop(columns=[self.TARGET_COLUMN])
        Ht = torch.from_numpy(self.H_feats_support).to(self.device, self.dtype)
        logits = torch.tensor([
            self._robust_logit_with_initial(Ht[i]) for i in range(Ht.size(0))
        ], device=self.device, dtype=self.dtype)
        mask = (torch.sigmoid(logits) > 0.5).cpu().numpy()
        # Use original model predictions on data_support
        base = self.task.model.predict(feats).values.ravel()
        keep = mask & (base >= 0.5)
        return feats.iloc[keep].reset_index(drop=True)
    
    def _generation_method(self, x: pd.Series, **_: Any) -> pd.DataFrame:
        """Generate a counterfactual for the given input using data-supported search."""
        S = self.getCandidates()
        if S.empty:
            return pd.DataFrame(x).T
        tree = KDTree(S.values)
        idx = tree.query(x.values.reshape(1, -1), k=1)[1][0, 0]
        return S.iloc[[idx]]


class LastLayerEllipsoidCEOHCNTNam(EllipsoidCEBase):
    """
    NAM-specific implementation of LastLayerEllipsoidCEOHCNT (continuous).
    
    This class extends EllipsoidCEBase to handle NAM models with continuous
    optimization, similar to LastLayerEllipsoidCEOHCNam but with
    gradient-based counterfactual generation.
    """
    
    def __init__(
        self,
        task: Task,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        learning_rate: float = 0.1,
        max_iterations: int = 400,
        early_stopping: int = 20,
        robust_coef: float = 1.0,
        sparsity_coef: float = 0.0,
        proximity_coef: float = 0.1,
        optimizer: str = "adam",
        **params
    ) -> None:
        # Initialize base class
        super().__init__(task, device, dtype, **params)
        
        # NAM-specific parameters
        self.use_initial = params.get('use_initial', True)
        
        # Set optimization-specific parameters
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.early_stopping = early_stopping
        self.robust_coef = robust_coef
        self.sparsity_coef = sparsity_coef
        self.proximity_coef = proximity_coef
        self.optimizer = optimizer.lower()  # Normalize to lowercase
        
    def _split_model(self, model: nn.Module) -> tuple[nn.Module, torch.Tensor]:
        """
        Split NAM model into penultimate feature extractor and last layer parameters.
        
        For NAMs, this involves:
        1. Creating a NAMPenult module to extract per-feature contributions
        2. Extracting the bias and constructing theta_star (last layer parameters)
        """
        # Get input dimension from task data if not already set
        if not hasattr(self, 'input_dim'):
            df_train = self.task.training_data.data
            X_train = df_train.drop(columns=[self.TARGET_COLUMN]).values.astype(np.float32)
            self.input_dim = X_train.shape[1]
        
        # Check if this is a NAM model
        if hasattr(model, "feature_nns"):
            # Create NAM penultimate feature extractor
            penult = NAMPenult(model)
            
            # Extract NAM bias
            try:
                bias = extract_nam_bias(model)
            except AttributeError:
                # Fallback: create a zero bias
                bias = torch.zeros(1, device=self.device, dtype=self.dtype)
            
            # Create NAMFreeHead to handle last layer parameters
            # Get input dimension from task data if not already set
            if not hasattr(self, 'input_dim'):
                df_train = self.task.training_data.data
                X_train = df_train.drop(columns=[self.TARGET_COLUMN]).values.astype(np.float32)
                self.input_dim = X_train.shape[1]
            
            free_head = NAMFreeHead(
                num_features=self.input_dim,
                bias_init=bias,
                device=self.device,
                dtype=self.dtype
            )
            
            # Construct theta_star (last layer parameters)
            theta_star = free_head.pack_theta()
            
            return penult, theta_star
        else:
            # Fallback to standard model splitting
            return super()._split_model(model)
    
    def _penult_features(self, X: torch.Tensor) -> torch.Tensor:
        """Extract penultimate features using NAM-specific logic."""
        # Ensure X is a PyTorch tensor
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, device=self.device, dtype=self.dtype)
        
        if hasattr(self.penult, 'feature_nns'):
            # NAM-specific feature extraction
            features = self.penult(X)
            # Flatten the features to 2D if needed (NAM returns per-feature contributions)
            if features.dim() == 3:
                features = features.view(features.size(0), -1)  # Flatten to (batch_size, features)
            return features
        else:
            # Standard feature extraction
            return super()._penult_features(X)
    
    def _robust_logit(self, X: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        Compute robust logit for NAM models.
        
        This method handles the NAM-specific logit computation using
        the penultimate features and theta parameters.
        """
        if hasattr(self.penult, 'feature_nns'):
            # NAM-specific robust logit computation
            h_flat = self._penult_features(X)
            bias = torch.ones(h_flat.size(0), 1, device=self.device, dtype=self.dtype)
            h_aug = torch.cat([h_flat, bias], dim=1)
            logits = torch.matmul(h_aug, theta)
            return logits
        else:
            # Standard robust logit computation
            return super()._robust_logit(X, theta)
    
    def _robust_emp_logit(self, X: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        Compute robust empirical logit for NAM models.
        
        This method filters models based on use_initial parameter and computes
        the empirical logit using the filtered model set.
        """
        if hasattr(self.penult, 'feature_nns'):
            # NAM-specific empirical logit computation
            if self.use_initial:
                # Use only the initial model (theta_star)
                return self._robust_logit(X, self.theta_star)
            else:
                # Use all models in the Rashomon set: compute h_aug, then delegate to base
                h_flat = self._penult_features(X)
                bias = torch.ones(h_flat.size(0), 1, device=self.device, dtype=self.dtype)
                h_aug = torch.cat([h_flat, bias], dim=1)
                h_aug_vec = h_aug[0] if h_aug.dim() > 1 else h_aug
                return super()._robust_emp_logit(h_aug_vec)
        else:
            # Standard empirical logit computation
            return super()._robust_emp_logit(X, theta)
    
    def _generation_method(self, x: pd.Series, **kwargs: Any) -> pd.DataFrame:
        """Generate a counterfactual for the given input using continuous optimization."""
        # Convert pandas Series to numpy array
        x_np = x.values.astype(np.float32)
        x_tensor = torch.tensor(x_np, device=self.device, dtype=self.dtype).unsqueeze(0)
        
        # Generate counterfactual using the optimization method
        # This will use the NAM-specific _penult_features and _robust_logit methods
        cf_tensor = self._optimize_counterfactual(x_tensor)
        
        # Convert back to pandas DataFrame
        cf_np = cf_tensor.squeeze(0).cpu().numpy()
        cf_df = pd.DataFrame([cf_np], columns=x.index)
        
        return cf_df
    
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
        
        # For early stopping - aligned with standard CNT
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
            
            if iteration % 50 == 0:
                print('-' * 100)
                with torch.no_grad():
                    print(f"Iteration {iteration+1}/{self.max_iterations} - "
                        f"Pred logit_center: {logit_center.item():.4f}, "
                        f"Pred Loss: {pred_loss.item():.4f}")
            
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
            
            if iteration % 50 == 0:
                print(f"Iteration {iteration+1}/{self.max_iterations} - "
                    f"Robust logit: {robust_logit.item():.4f}, "
                    f"Robust Loss: {robust_loss.item():.4f}")
            
            # Combined loss - aligned with standard CNT
            total_loss = (
                local_pred_coef * pred_loss + 
                local_robust_coef * robust_loss
            )
            
            # Backward and optimize
            total_loss.backward()
            optimizer.step()
            
            # Project back to feature bounds
            with torch.no_grad():
                x.data = torch.min(torch.max(x.data, self.feature_mins), self.feature_maxs)
            
            # Check if we've crossed the decision boundary with robustness
            # Criterion: robust logic > previous robust (aligned with standard CNT)
            with torch.no_grad():
                if robust_logit.item() > best_robust_logit:
                    best_robust_logit = robust_logit.item()
                    best_x = x.clone().detach()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            
            # Early stopping if no improvement for several iterations OR worst_logit > 0
            if worst_logit > 0 or no_improve_count >= self.early_stopping:
                break
        
        # Return the best solution found (aligned with standard CNT)
        return best_x


# --------------------------------------------------------------------------- #
# Export classes
# --------------------------------------------------------------------------- #

__all__ = [
    "LastLayerEllipsoidCEOHCNam",
    "LastLayerEllipsoidCEOHCNTNam",
]