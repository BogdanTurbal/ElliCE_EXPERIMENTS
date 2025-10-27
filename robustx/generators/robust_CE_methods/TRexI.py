import torch
import numpy as np
import pandas as pd
import datetime
from typing import Dict, Any, List, Tuple, Optional
from robustx.generators.CEGenerator import CEGenerator
from robustx.lib.tasks.Task import Task

class TRexI(CEGenerator):
    """
    T-Rex:I: Theoretically Robust EXplanations - Iterative Version
    
    This counterfactual explanation generator implements the continuous gradient-based
    approach from the paper "Robust Counterfactual Explanations for Neural Networks
    With Probabilistic Guarantees". It optimizes counterfactuals by maximizing
    a stability measure using gradient ascent.
    
    The stability measure quantifies the robustness of a counterfactual to model changes.
    """
    
    def __init__(self, task: Task, device=None, **kwargs):
        """
        Initialize the T-Rex:I counterfactual explanation generator.
        
        Parameters:
        -----------
        task : Task
            The task containing the model and dataset
        device : torch.device, optional
            Device to perform computations on (CPU/GPU)
        """
        super().__init__(task)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def _generate_wachter_counterfactual(self, 
                                         x, 
                                         column_name="target", 
                                         neg_value=0, 
                                         lamb=0.1, 
                                         lr=0.02,
                                         max_iter=200, 
                                         max_allowed_minutes=0.5, 
                                         epsilon=0.001) -> torch.Tensor:
        """
        Generate an initial counterfactual using Wachter's method.
        
        Parameters:
        -----------
        x : pd.Series or pd.DataFrame
            The instance for which to generate a counterfactual
        column_name : str
            Name of the target column
        neg_value : int
            The negative class (0 or 1)
        lamb : float
            Regularization coefficient for the proximity loss
        lr : float
            Learning rate for gradient descent
        max_iter : int
            Maximum number of iterations
        max_allowed_minutes : float
            Maximum time allowed for optimization (in minutes)
        epsilon : float
            Small value for convergence check
            
        Returns:
        --------
        torch.Tensor
            The generated counterfactual
        """
        # Convert input to tensor
        if isinstance(x, pd.DataFrame):
            x_np = x.values[0]
        else:
            x_np = x.values
            
        x_tensor = torch.tensor(x_np, dtype=torch.float32, requires_grad=False, device=self.device)
        
        # Initialize counterfactual at input point
        cf = torch.tensor(x_np, dtype=torch.float32, requires_grad=True, device=self.device)
        
        # Setup optimizer
        optimizer = torch.optim.Adam([cf], lr=lr)
        
        # Define loss functions
        validity_loss = torch.nn.BCELoss()
        
        # Set target label (opposite of current prediction)
        y_target = torch.tensor([1 - neg_value], dtype=torch.float32, device=self.device)
        
        # Initial validity check
        with torch.no_grad():
            pred = self._model_predict_tensor(cf.unsqueeze(0))
            valid = (neg_value == 0 and pred >= 0.5) or (neg_value == 1 and pred < 0.5)
        
        # Set maximum allowed time
        t0 = datetime.datetime.now()
        t_max = datetime.timedelta(minutes=max_allowed_minutes)
        
        # Optimization loop
        iterations = 0
        while not valid and iterations < max_iter:
            
            optimizer.zero_grad()
            
            # Compute model prediction
            pred = self._model_predict_tensor(cf.unsqueeze(0))
            
            # Calculate the proximity cost (L1 distance)
            #proximity_cost = torch.sum(torch.abs(cf - x_tensor))
            
            # Combined loss
            loss = validity_loss(pred, y_target) #+ lamb * proximity_cost
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            # Check if valid
            with torch.no_grad():
                pred_val = pred.item()
                if (neg_value == 0 and pred_val >= 0.5 + epsilon) or (neg_value == 1 and pred_val < 0.5 - epsilon):
                    valid = True
                    
            # Check time limit
            if datetime.datetime.now() - t0 > t_max:
                break
                
            iterations += 1
        
        return cf.detach()
        
    def _generation_method(self, 
                          x, 
                          trex_k: int = 1000, 
                          sigma_squared: float = 0.01, 
                          eta: float = 0.01,
                          trex_threshold: float = 0.7, 
                          max_steps: int = 200, 
                          column_name: str = "target", 
                          neg_value: int = 0,
                          **kwargs) -> pd.DataFrame:
        """
        Implements Algorithm 1 (T-Rex:I) from the paper, with Wachter initialization.
        
        Parameters:
        -----------
        x : pd.Series or pd.DataFrame
            The instance for which to generate a counterfactual
        k : int
            Number of samples for the stability estimation
        sigma_squared : float
            Variance for Gaussian sampling around the counterfactual
        eta : float
            Learning rate for gradient ascent
        tau : float
            Threshold for the stability measure
        max_steps : int
            Maximum number of gradient ascent steps
        column_name : str
            Name of the target column
        neg_value : int
            The negative class (0 or 1)
            
        Returns:
        --------
        pd.DataFrame
            The generated robust counterfactual
        """
        k = trex_k
        tau = trex_threshold
        # First, generate an initial counterfactual using Wachter's method
        # print(x)
        initial_cf = self._generate_wachter_counterfactual(
            x, 
            column_name=column_name, 
            neg_value=neg_value,
            lamb=0.1,  # Balancing factor between validity and proximity
            lr=0.1,   # Learning rate
            max_iter=100,  # Max iterations
            max_allowed_minutes=0.5  # Time limit
        )
        # print("initial", initial_cf)
        # Initialize robust counterfactual
        x_c = initial_cf.clone().requires_grad_(True).to(self.device)
        
        # Track original input for regularization
        x_orig = torch.tensor(x.values if isinstance(x, pd.Series) else x.iloc[0].values, 
                             dtype=torch.float32, device=self.device)
        
        # Setup optimizer
        optimizer = torch.optim.Adam([x_c], lr=eta)
        
        # Main optimization loop
        steps = 0
        while steps < max_steps:
            # Reset gradients
            optimizer.zero_grad()
            
            # Compute stability measure
            stability = self._compute_stability(x_c.unsqueeze(0), k, sigma_squared)
            
            # Check if stability threshold is met
            if stability.item() >= tau:
                break
                
            # Compute loss (negative stability to maximize)
            loss = -stability
            
            # Add optional regularization for proximity to original instance
            # loss += 0.01 * torch.norm(x_c - x_orig, p=1)
            
            # Compute gradients and update
            loss.backward()
            optimizer.step()
            
            # Increment steps
            steps += 1
            
            # print(f"Step {steps}: Stability = {stability.item():.4f}, Loss = {loss.item():.4f}")
            # print(x_c)
            
        # Convert to DataFrame
        if isinstance(x, pd.Series):
            columns = x.index
        else:
            columns = x.columns
            
        result = pd.DataFrame(x_c.detach().cpu().numpy().reshape(1, -1), columns=columns)
        
        return result
    
    def _compute_stability(self, x_c: torch.Tensor, k: int, sigma_squared: float) -> torch.Tensor:
        """
        Compute the stability measure (relaxed version from the paper).
        
        The stability is computed as:
        R̂k,σ2(x, m) = (1/k) Σ [m(xi) - |m(x) - m(xi)|]
        
        Parameters:
        -----------
        x_c : torch.Tensor
            The counterfactual being optimized
        k : int
            Number of Gaussian samples
        sigma_squared : float
            Variance for Gaussian sampling
            
        Returns:
        --------
        torch.Tensor
            The computed stability measure
        """
        # Create k Gaussian samples around x_c
        noise = torch.randn(k, x_c.shape[1], device=self.device) * np.sqrt(sigma_squared)
        samples = x_c.repeat(k, 1) + noise
        
        # Get model prediction for x_c
        m_x_c = self._model_predict_tensor(x_c)
        
        # Get predictions for samples
        m_samples = torch.stack([self._model_predict_tensor(sample.unsqueeze(0)) for sample in samples])
        
        # Compute stability measure
        diff = torch.abs(m_x_c - m_samples)
        stability = torch.mean(m_samples - diff)
        
        return stability
        
    def _model_predict_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions on a tensor input, directly using the PyTorch model
        to maintain gradient flow throughout the computation.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor
            
        Returns:
        --------
        torch.Tensor
            Model prediction (probability)
        """
        # Get the raw PyTorch model
        raw_model = self.task.model._model
        
        # Ensure the input is on the correct device
        x = x.to(self.device)
        
        # Temporarily set to eval mode without affecting the training state
        training_mode = raw_model.training
        raw_model.eval()
        
        # Forward pass through the model, keeping gradients intact
        outputs = raw_model(x)
        
        # Get probabilities - handle different output formats
        if isinstance(outputs, torch.Tensor):
            if outputs.shape[-1] > 1:  # Multi-class output
                probs = torch.nn.functional.softmax(outputs, dim=-1)
                # Take the probability for the positive class (typically index 1)
                result = probs[:, 1] if probs.shape[-1] > 1 else probs[:, 0]
            else:  # Binary output (already sigmoid)
                result = outputs.view(-1)
        else:
            # Handle any unexpected output format
            raise TypeError(f"Unexpected model output type: {type(outputs)}")
        
        # Restore the original training mode
        raw_model.train(training_mode)
        
        return result