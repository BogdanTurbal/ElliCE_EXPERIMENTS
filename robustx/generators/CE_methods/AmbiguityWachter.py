import datetime
import pandas as pd
import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam
from robustx.generators.CEGenerator import CEGenerator

# We reuse the CostLoss from Wachter as a helper.
# class CostLoss(nn.Module):
#     def forward(self, x1, x2):
#         return torch.abs(x1 - x2)

class CostLoss(nn.Module):
    def forward(self, x1, x2):
        return torch.norm(x1 - x2, p=2)#torch.sqrt(torch.sum((x1 - x2) ** 2))

class AmbiguityWachter(CEGenerator):
    """
    A counterfactual explanation generator that is based on Wachter’s method but augments
    the loss with two additional terms:
      - an ambiguity term r(x), weighted by ambiguity_weight, which is intended to approximate 
        the viable prediction range (e.g. the difference between the max and min predictions over 
        a set of alternative models produced via AWP), and 
      - a density term d(x), weighted by density_weight (with a negative sign) so that points 
        in high-density regions are favored.
    
    The final loss used during gradient descent is:
        l_fin(x) = validity_loss + lamb * cost_loss + ambiguity_weight * r_model(x) - density_weight * d_model(x)
    
    Note: r_model and d_model must be provided as callables (or PyTorch modules) that accept a tensor 
    (the candidate counterfactual) and return a scalar tensor.
    """
    #2 -> 0.2
    def _generation_method(self, instance, column_name="target", neg_value=0,
                             lamb=0.6, lr=0.02, max_iter=10000000, max_allowed_minutes=0.05,
                             epsilon=0.001, ambiguity_weight=-1, density_weight=1.0,
                             r_model=None, d_model=None, level=0.5, **kwargs):
        """
        Generates a counterfactual explanation using a gradient-based approach with an augmented loss.
        
        Parameters:
            instance (pd.DataFrame or pd.Series): The input instance for which to generate a counterfactual.
            column_name (str): Name of the target column.
            neg_value (int): The negative class label.
            lamb (float): Weight for the cost loss term.
            lr (float): Learning rate for gradient descent.
            max_iter (int): Maximum number of iterations.
            max_allowed_minutes (float): Maximum allowed runtime (in minutes).
            epsilon (float): Tolerance threshold for class prediction.
            ambiguity_weight (float): Weight for the ambiguity term.
            density_weight (float): Weight for the density term.
            r_model (callable): A pre-trained regressor (or callable) that approximates r(x).
            d_model (callable): A pre-trained density estimator (or callable) that returns d(x).
            **kwargs: Additional keyword arguments.
        
        Returns:
            pd.DataFrame: A DataFrame containing the generated counterfactual explanation.
        """
        if r_model is None or d_model is None:
            raise ValueError("Both r_model and d_model must be provided for AmbiguityWachter method.")
        
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Convert the instance to a tensor and initialize the counterfactual candidate (wac)
        x = torch.Tensor(instance.to_numpy()).to(DEVICE)
        wac = Variable(x.clone(), requires_grad=True).to(DEVICE)
        
        optimiser = Adam([wac], lr=lr, amsgrad=True)
        
        validity_loss = torch.nn.BCELoss()
        cost_loss = CostLoss()
        
        # Target is the opposite of the negative class (i.e. 1 - neg_value)
        y_target = torch.Tensor([1 - neg_value]).to(DEVICE)
        
        # Get the model's prediction on the candidate counterfactual
        class_prob = self.task.model.predict_proba_tensor(wac)
        wac_valid = False
        iterations = 0
        
        # Check if the initial candidate is already valid
        if (neg_value == 0 and class_prob.item() >= 0.5) or (neg_value == 1 and class_prob.item() < 0.5):
            wac_valid = True
        
        t0 = datetime.datetime.now()
        t_max = datetime.timedelta(minutes=max_allowed_minutes)
        
        # Gradient descent loop
        while not wac_valid and iterations < max_iter:
            optimiser.zero_grad()
            class_prob = self.task.model.predict_proba_tensor(wac)
            p = class_prob.item()
            
            br = ((neg_value == 0 and p >= 0.5 - epsilon) or (neg_value == 1 and p < 0.5 + epsilon))
            
            # Compute the extra loss terms:
            # r_model: ambiguity predictor; d_model: density estimator.
            ambiguity = r_model(wac)  # should return a scalar tensor
            density = d_model(wac)    # should return a scalar tensor
            
            # Combined loss: original Wachter loss + extra terms
            
            # loss = (1 - int(br)) * validity_loss(class_prob, y_target) + lamb * cost_loss(x, wac) + ambiguity_weight * ambiguity #- density_weight * density
            # loss.sum().backward()
            # optimiser.step()
            #coof = 0.5 if br else 1.0
            #1.5 * max(1.2 - p * 2, 0) * .sum()
            #print(lamb)
            cost_l = lamb * cost_loss(x, wac)
            #print(cost_l)
            loss = validity_loss(class_prob, y_target) + cost_l + ambiguity_weight * ambiguity #- density_weight * density
            loss.backward()
            optimiser.step()
            
            
            class_prob = self.task.model.predict_proba_tensor(wac)
            p = class_prob.item()
            
            br = ((neg_value == 0 and p >= 0.5 - epsilon) or (neg_value == 1 and p < 0.5 + epsilon))
            
            if br and ambiguity.item() > level:
                wac_valid = True
                
            if datetime.datetime.now() - t0 > t_max:
                #print('Time limit reached')
                break
            iterations += 1
        
        # Return the counterfactual as a DataFrame with the same index as the input instance.
        res = pd.DataFrame(wac.detach().cpu().numpy()).T
        res.columns = instance.index
        return res
