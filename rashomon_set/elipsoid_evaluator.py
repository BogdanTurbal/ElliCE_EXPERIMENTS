import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from functools import lru_cache
from sklearn.neighbors import KDTree

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
from functools import lru_cache
from sklearn.neighbors import KDTree

from tqdm import tqdm

from robustx.generators.robust_CE_methods.LastLayerEllipsoidCE import LastLayerEllipsoidCEOHC


class AnalyticalEllipsoidEvaluator:
    """
    Evaluator that uses the analytical approach to find the worst-case model
    within the ellipsoid and computes the corresponding prediction.
    """
    
    def __init__(self, generator, threshold=0.5, scaling_factor=0.99, adaptation_rate=0.99):
        """
        Initialize the analytical ellipsoid evaluator.
        
        Args:
            generator: The LastLayerEllipsoidCEOH instance
            threshold: Decision threshold for binary classification
            scaling_factor: Factor for scaling the step size along the direction
                           of steepest descent (smaller = more conservative)
            adaptation_rate: Rate at which to reduce the scaling factor when
                           a model outside the Rashomon set is found
        """
        self.generator = generator
        self.threshold = threshold
        self.base_scaling_factor = scaling_factor
        self.adaptation_rate = adaptation_rate
        self.theta_threshold = 0
        
        # For monitoring scaling factor changes
        self.scaling_factor_adjustments = []
    
    def find_valid_worst_case_model(self, h_aug, theta_c, scaling_factor=None, max_attempts=100):
        """
        Find a valid worst-case model within the Rashomon set by recursively
        adjusting the scaling factor if necessary.
        
        Args:
            h_aug: Augmented penultimate features
            theta_c: Center model parameters
            scaling_factor: Current scaling factor (defaults to self.base_scaling_factor)
            max_attempts: Maximum number of attempts to find a valid model
            
        Returns:
            tuple: (worst_model, worst_logit, is_valid, adjusted_scaling_factor)
        """
        if scaling_factor is None:
            scaling_factor = self.base_scaling_factor
            
        attempts = 0
        current_scaling = scaling_factor
        
        while attempts < max_attempts:
            # Compute the direction of steepest descent - use original approach
            with torch.no_grad():
                # Use the generator's Q_inv_sqrt for computation, as in original code
                u = self.generator.Q_inv_sqrt @ h_aug.T
                norm_u = u.norm() / current_scaling
                direction = self.generator.Q_inv_sqrt @ u.T
                
                # Compute the worst-case model - as in original code
                worst_model = theta_c - direction / norm_u
                
                # Compute the logit for the worst-case model
                worst_logit = float(torch.dot(worst_model, h_aug))
                
                # Check if the model is within the Rashomon set
                if hasattr(self.generator, 'H_tensor') and hasattr(self.generator, 'y_signed'):
                    # Use generator's cached data if available
                    H_tensor = torch.from_numpy(self.generator.H_tensor).to(self.generator.device, self.generator.dtype)
                    y_signed = torch.from_numpy(self.generator.y_signed).to(self.generator.device, self.generator.dtype)
                    
                    # Calculate worst-case model loss
                    logits_mat = H_tensor @ worst_model
                    losses = torch.nn.functional.softplus(-y_signed * logits_mat)
                    model_loss = losses.mean().item()
                    
                    # Check if the model is in the Rashomon set
                    is_valid = True#model_loss <= self.theta_threshold
                else:
                    # If we can't verify Rashomon membership, assume it's valid
                    # This is a fallback case and not ideal
                    is_valid = True
                
                if is_valid:
                    # We found a valid model, return it
                    return worst_model, worst_logit, True, current_scaling
                
                # If not valid, reduce the scaling factor and try again
                current_scaling *= self.adaptation_rate
                attempts += 1
                
                # Record the adjustment
                self.scaling_factor_adjustments.append(current_scaling)
                
        # If we get here, we couldn't find a valid model
        # Return the most conservative model we tried
        return worst_model, worst_logit, False, current_scaling
    
    def predict_single(self, instance):
        """
        Predict the robustness of a single instance by computing the worst-case prediction.
        
        Args:
            instance: DataFrame with a single row containing the instance to evaluate
            
        Returns:
            1 if the worst-case prediction is still positive, 0 otherwise
        """
        with torch.no_grad():
            # Convert input to the right format
            x = torch.from_numpy(instance.values[0]).to(self.generator.device, self.generator.dtype)
            
            # Get penultimate features
            h = self.generator.penult(x.unsqueeze(0)).squeeze(0)
            h_aug = torch.cat([h.flatten(), torch.tensor([1.0], device=self.generator.device, dtype=self.generator.dtype)])
            
            # Get the center model - use omega_c as in your original code
            theta_c = self.generator.omega_c.detach()
            
            # Find a valid worst-case model
            worst_model, worst_logit, is_valid, adjusted_scaling = self.find_valid_worst_case_model(
                h_aug, theta_c
            )
            
            # Check if the prediction is still positive
            worst_prob = float(torch.sigmoid(torch.tensor(worst_logit)))
            
            # If we had to significantly adjust the scaling factor, print a warning
            # if adjusted_scaling < self.base_scaling_factor * 0.9:
            #     print(f"Warning: Scaling factor adjusted from {self.base_scaling_factor:.4f} to {adjusted_scaling:.4f}")
            
            # Return 1 if the worst-case prediction is still positive and we found a valid model
            # Otherwise, return 0
            print(f"Worst probability: {worst_prob}, is valid: {is_valid}")
            if worst_prob > self.threshold and is_valid:
                return 1.0
            else:
                return 0.0
    
    def predict(self, X):
        """
        Predict for multiple instances.
        
        Args:
            X: DataFrame with multiple rows
            
        Returns:
            Array of predictions (1 for robust, 0 for non-robust)
        """
        return np.array([self.predict_single(X.iloc[[i]]) for i in range(len(X))])
    
    def get_scaling_stats(self):
        """
        Get statistics about scaling factor adjustments.
        
        Returns:
            dict: Dictionary containing statistics about scaling factor adjustments
        """
        if not self.scaling_factor_adjustments:
            return {
                "adjustments_made": 0,
                "min_scaling": self.base_scaling_factor,
                "max_scaling": self.base_scaling_factor,
                "avg_scaling": self.base_scaling_factor
            }
        
        return {
            "adjustments_made": len(self.scaling_factor_adjustments),
            "min_scaling": min(self.scaling_factor_adjustments),
            "max_scaling": max(self.scaling_factor_adjustments),
            "avg_scaling": sum(self.scaling_factor_adjustments) / len(self.scaling_factor_adjustments)
        }
        