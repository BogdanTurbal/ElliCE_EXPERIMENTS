#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Base class for Ellipsoidal Counterfactual Explanation methods.

This module provides a common base class that implements the shared ellipsoidal
approximation logic used across different LastLayerEllipsoidCE variants.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import lru_cache
from typing import Optional, List, Any, Dict, Union
from sklearn.linear_model import LogisticRegression

from robustx.generators.CEGenerator import CEGenerator
from robustx.lib.tasks.Task import Task

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def flatten_last_layer(model: nn.Module) -> torch.Tensor:
    """Flatten weight and bias of the final linear layer into a single parameter vector."""
    layer = None
    for module in reversed(list(model.modules())):
        if isinstance(module, nn.Linear):
            layer = module
            break
    if layer is None:
        raise ValueError("No nn.Linear layer found in model")
    return torch.cat([layer.weight.detach().flatten(), layer.bias.detach()])


def safe_log1pexp(x: torch.Tensor) -> torch.Tensor:
    """Stable computation of log(1+exp(x)) that avoids overflow."""
    return torch.where(x > 20, x, torch.log1p(torch.exp(x)))


# --------------------------------------------------------------------------- #
# Base EllipsoidCE Class
# --------------------------------------------------------------------------- #

class EllipsoidCEBase(CEGenerator):
    """
    Base class for ellipsoidal counterfactual explanation methods.
    
    This class implements the common ellipsoidal approximation logic shared across
    different LastLayerEllipsoidCE variants, including:
    - Model splitting and penultimate feature extraction
    - Ellipsoid construction from empirical Hessian
    - Worst-case model computation
    - Robust logit computation
    - Model sampling from the Rashomon set
    """
    
    TARGET_COLUMN: str = "target"
    
    def __init__(
        self,
        task: Task,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        **params
    ) -> None:
        super().__init__(task)
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        
        # Get task parameters
        self.eps = float(self.task.eps)
        self.reg_coef = float(self.task.reg_coef)
        
        self.refit = bool(getattr(self.task, 'refit_last_layer', False))
        print("Using refit_last_layer: ", self.refit)
        print(f"Reg coef: {self.reg_coef}")
        print(f"Eps: {self.eps}")
        
        # Setup model and extract components
        model = self.task.model.get_torch_model().to(self.device, self.dtype).eval()
        self.penult, self.theta_star = self._split_model(model)
        
        # Process training data for Rashomon set characterization
        df_train = self.task.training_data.data
        X_train = df_train.drop(columns=[self.TARGET_COLUMN]).values.astype(np.float32)
        y_train = df_train[self.TARGET_COLUMN].values.astype(np.float32)
        self.input_dim = X_train.shape[1]
        
        # Generate penultimate features for training data
        with torch.no_grad():
            H_flat_train = self._penult_features(X_train)
            bias_train = torch.ones(H_flat_train.size(0), 1, device=self.device, dtype=self.dtype)
            H_aug_train = torch.cat([H_flat_train, bias_train], dim=1)
        H_feats_train = H_aug_train.cpu().numpy()
        
        # Optional refit via logistic regression using training data
        if self.refit:
            # Compute loss before refit
            theta_orig_np = self.theta_star.cpu().numpy()
            logits_orig = H_feats_train @ theta_orig_np
            loss_vals_orig = np.log1p(np.exp(- (2 * y_train - 1) * logits_orig))
            L_star_orig = float(np.mean(loss_vals_orig))
            print(f"[ℹ] Before re‑fit: L★ = {L_star_orig:.6f}")
            
            # Fit logistic regression
            clf = LogisticRegression(
                penalty="l2",
                C=1.0 / max(self.reg_coef, 1e-12),
                fit_intercept=False,
                solver="lbfgs",
                max_iter=2000,
            )
            clf.fit(H_feats_train, y_train)
            theta_np = clf.coef_.ravel().astype(np.float32)
            self.theta_star = torch.from_numpy(theta_np).to(self.device, self.dtype)
            print("[ℹ] Re‑fitted last layer via logistic regression.")
            
            # Compute loss after refit
            logits_refit = H_feats_train @ theta_np
            loss_vals_refit = np.log1p(np.exp(- (2 * y_train - 1) * logits_refit))
            L_star_refit = float(np.mean(loss_vals_refit))
            print(f"[ℹ] After re‑fit: L★ = {L_star_refit:.6f}")
            
            # Update model's last layer
            self._update_model_last_layer(model, self.theta_star)
        else:
            self.theta_star = self.theta_star.to(self.device, self.dtype)
        
        # Store central parameter vector
        self.omega_c = self.theta_star.detach()
        
        # Compute empirical Hessian on training data
        m = H_feats_train.shape[1]
        I = np.eye(m)
        logits_train = H_feats_train @ self.theta_star.cpu().numpy()
        p_train = 1.0 / (1.0 + np.exp(-logits_train))
        W_train = H_feats_train * (p_train * (1 - p_train))[:, None]
        H = (W_train.T @ H_feats_train) / H_feats_train.shape[0] + self.reg_coef * I
        
        # Compute loss threshold based on training data
        self.y_signed_train = 2 * y_train - 1
        loss_vals_train = np.log1p(np.exp(-self.y_signed_train * logits_train))
        self.L_star = float(np.mean(loss_vals_train))
        self.theta_threshold = self.L_star + self.eps
        
        # Construct ellipsoid representation
        self.Q = torch.from_numpy(H / (2 * self.eps)).to(self.device, self.dtype)
        self.Q_inv_sqrt = self._inv_sqrt(self.Q)
        
        # Store training data for model validation
        self.H_feats_train = H_feats_train
        
        # Print diagnostics
        print(f"L★ = {self.L_star:.6f}; θ‑threshold = {self.theta_threshold:.6f}")
        print(f"Ellipsoid constructed with {m} parameters")
        
        # Feature bounds for constraining counterfactuals
        self.feature_mins = torch.tensor(np.min(X_train, axis=0), device=self.device, dtype=self.dtype)
        self.feature_maxs = torch.tensor(np.max(X_train, axis=0), device=self.device, dtype=self.dtype)
    
    def _update_model_last_layer(self, model: nn.Module, theta: torch.Tensor):
        """Update the weights of the last layer in the model."""
        last_linear = None
        for m in reversed(list(model.modules())):
            if isinstance(m, nn.Linear): 
                last_linear = m
                break
        if not last_linear:
            print("[WARN] No Linear layer found.")
            return
        
        with torch.no_grad():
            w = theta[:-1].view_as(last_linear.weight)
            b = theta[-1:].view_as(last_linear.bias)
            last_linear.weight.copy_(w)
            last_linear.bias.copy_(b)
            print("[ℹ] Model last layer updated.")
    
    def _split_model(self, model: nn.Module):
        """Split model into penultimate features extractor and last layer parameters."""
        children = list(model.children())
        penult = nn.Sequential(*children[:-2]).to(self.device, self.dtype).eval()
        theta = self._flatten_last_layer(model)
        return penult, theta
    
    def _flatten_last_layer(self, model: nn.Module):
        """Extract and flatten the parameters of the last layer."""
        last_linear = None
        for m in reversed(list(model.modules())):
            if isinstance(m, nn.Linear):
                last_linear = m
                break
        if last_linear is None:
            raise ValueError("No linear layer found in the model")
            
        weight = last_linear.weight.detach().view(-1)
        bias = last_linear.bias.detach()
        return torch.cat([weight, bias])
    
    def _penult_features(self, X_input):
        """Compute penultimate layer features for the input.
        
        Args:
            X_input: Either a numpy array or a torch tensor
            
        Returns:
            Penultimate layer features as tensor
        """
        # Check if input is a tensor or numpy array
        if isinstance(X_input, np.ndarray):
            X_t = torch.from_numpy(X_input).to(self.device, self.dtype)
            with torch.no_grad():
                H = self.penult(X_t)
            return H.view(H.size(0), -1)
        else:
            # Input is already a tensor that may require gradients
            H = self.penult(X_input)
            return H.view(H.size(0), -1)
    
    @staticmethod
    def _inv_sqrt(Q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Compute the inverse square root of a matrix."""
        n = Q.shape[-1]
        I = torch.eye(n, dtype=Q.dtype, device=Q.device)
        Q = Q + eps * I
        w, V = torch.linalg.eigh(Q)
        return (V * w.clamp(min=eps).rsqrt().unsqueeze(0)) @ V.T
    
    def _sample_omegas(self, k: int) -> torch.Tensor:
        """Sample k models from the Rashomon ellipsoid."""
        m = self.Q.shape[0]
        u = torch.randn(k, m, device=self.device, dtype=self.dtype)
        u = u / u.norm(dim=1, keepdim=True)
        r = torch.rand(k, 1, device=self.device, dtype=self.dtype).pow(1 / m)
        y = u * r
        return (self.Q_inv_sqrt @ y.T).T + self.omega_c
    
    def _compute_worst_model(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the worst-case model from the Rashomon set for a given input.
        This is the model that minimizes the logit (maximizes the loss) for the input.
        """
        with torch.no_grad():
            # Extract penultimate features
            h_flat = self._penult_features(x.cpu().numpy())
            # Add bias term
            h_aug = torch.cat([h_flat, torch.ones(h_flat.size(0), 1, device=self.device, dtype=self.dtype)], dim=1)
            h_aug = h_aug[0]  # Single instance
            
            # Compute the worst model analytically
            inv_sqrt = self.Q_inv_sqrt
            u = inv_sqrt @ h_aug
            direction = inv_sqrt @ u / u.norm()
            worst_theta = self.omega_c - direction
            
            # Validate the model is within the Rashomon set
            H_tensor = torch.from_numpy(self.H_feats_train).to(self.device, self.dtype)
            y_signed = torch.from_numpy(self.y_signed_train).to(self.device, self.dtype)
            loss = F.softplus(-y_signed * (H_tensor @ worst_theta)).mean()
            
            return worst_theta
    
    def _compute_worst_model_from_h_aug(self, h_aug: torch.Tensor) -> torch.Tensor:
        """
        Compute the worst-case model from the Rashomon set for given penultimate features.
        This is the model that minimizes the logit (maximizes the loss) for the input.
        """
        with torch.no_grad():
            # h_aug is already the augmented penultimate features
            if h_aug.dim() > 1:
                h_aug = h_aug[0]  # Single instance
            
            # Compute the worst model analytically
            inv_sqrt = self.Q_inv_sqrt
            u = inv_sqrt @ h_aug
            direction = inv_sqrt @ u / u.norm()
            worst_theta = self.omega_c - direction
            
            return worst_theta
    
    @torch.no_grad()
    def _robust_logit(
        self,
        h_aug: torch.Tensor,
        scaling_factor: float = 0.99,
        adaptation_rate: float = 0.99,
        max_attempts: int = 200,
    ) -> float:
        """Compute robust logit using worst-case model."""
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
            loss = F.softplus(-ys * (Ht @ theta)).mean()
            if loss <= self.theta_threshold:
                return new_pred
            current_scaling *= adaptation_rate
        return base
    
    @torch.no_grad()
    def _robust_emp_logit(
        self,
        h_aug: torch.Tensor,
        n_samples: int = 20_000,
        percentile: float = 0.0,
        filter_models: bool = True,
    ) -> float:
        """Compute robust logit using empirical sampling."""
        base = float(torch.dot(self.omega_c, h_aug))
        sampled = self._sample_omegas(n_samples)
        if filter_models:
            # Use training data for model filtering
            Ht = torch.from_numpy(self.H_feats_train).to(self.device, self.dtype)
            ys = torch.from_numpy(self.y_signed_train).to(self.device, self.dtype)
            losses = F.softplus(-ys.unsqueeze(1) * (Ht @ sampled.T))
            mask = losses.mean(0) <= self.theta_threshold
            if mask.sum() == 0:
                return base
            sampled = sampled[mask]
        logits = (sampled @ h_aug).sort().values
        idx = 0 if percentile <= 0 else (len(logits) - 1 if percentile >= 1 else int(percentile * (len(logits) - 1)))
        return logits[idx].item()
    
    def _measure_rashomon_precision(
        self,
        H_tensor: torch.Tensor,
        y_signed: torch.Tensor,
        theta_threshold: float,
        num_samples: int = 10_000,
        omegas: Optional[torch.Tensor] = None,
    ) -> float:
        """Measure precision of the Rashomon set approximation."""
        with torch.no_grad():
            omegas = omegas if omegas is not None else self._sample_omegas(num_samples)
            losses = F.softplus(-y_signed.unsqueeze(1) * (H_tensor @ omegas.T))
            return 100.0 * (losses.mean(0) <= theta_threshold).float().mean().item()
    
    def _print_parameter_ranges(self):
        """Print parameter ranges on the ellipsoid."""
        Q_inv = torch.linalg.inv(self.Q)
        radii = torch.sqrt(torch.diag(Q_inv))
        center = self.omega_c.cpu().numpy()
        print("\n--- Parameter Ranges on Ellipsoid ---")
        print(f"{'Param':>8} {'Min':>12} {'Max':>12} {'Range':>12}")
        print("-"*50)
        for i, c in enumerate(center):
            r = radii[i].item()
            name = f"θ_{i}" if i < len(center)-1 else 'bias'
            print(f"{name:>8} {c-r:12.6f} {c+r:12.6f} {2*r:12.6f}")
    
    # Abstract methods that subclasses must implement
    def _generation_method(self, x: pd.Series, **kwargs: Any) -> pd.DataFrame:
        """Generate a counterfactual for the given input. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _generation_method")
    
    def getCandidates(self) -> pd.DataFrame:
        """Get candidate counterfactuals. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement getCandidates")
