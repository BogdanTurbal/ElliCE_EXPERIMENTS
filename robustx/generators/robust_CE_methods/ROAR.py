import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import datetime
from scipy.optimize import linprog
from lime.lime_tabular import LimeTabularExplainer
from robustx.generators.CEGenerator import CEGenerator

class ROAR(CEGenerator):
    def __init__(
        self, task, **kwargs
    ):
        super().__init__(
            task
        )
    
    def _get_lime_coefficients(self, instance: pd.DataFrame, neg_value=0):
        # same as before
        data = self.task.training_data.X
        explainer = LimeTabularExplainer(
            training_data=data.values,
            feature_names=data.columns.tolist(),
            discretize_continuous=False,
            random_state=42
        )
        def predict_two_cols(x: np.ndarray) -> np.ndarray:
            # get your model’s single-column output
            probs = self.task.model\
                        .get_torch_model()(
                            torch.tensor(x, dtype=torch.float32)
                        )\
                        .detach().cpu().numpy().reshape(-1)
            # stack into two‐column array
            return np.vstack([1 - probs, probs]).T
        
        inst = instance.values[0]
        exp = explainer.explain_instance(
            data_row=inst,
            predict_fn=predict_two_cols,
            num_features=inst.shape[0]
        )
        coeff = np.zeros(inst.shape[0])
        for feat, wt in exp.as_list():
            idx = data.columns.tolist().index(feat)
            coeff[idx] = wt
        intercept_value = float(exp.intercept[1 - neg_value])
        return np.vstack([coeff]*len(instance)), np.array([intercept_value]*len(instance))

    # def _reconstruct_encoding_constraints(self, x_new, cat_idxs, binary_cat_features):
    #     # original encoding constraints logic
    #     x_np = x_new.detach().cpu().numpy().flatten()
    #     for group in self.categorical_groups:
    #         sub = x_np[group]
    #         i = np.argmax(sub)
    #         x_np[group] = 0
    #         x_np[group[i]] = 1
    #     return torch.tensor(x_np, dtype=torch.float32, device=x_new.device).view_as(x_new)
    def _reconstruct_encoding_constraints(self, x_new, cat_idxs, binary_cat_features):
        """
        Enforce one-hot encoding constraints based on provided index groups.
        """
        if binary_cat_features and cat_idxs:
        # we do this in no_grad so it doesn't get tracked as a gradient op
            with torch.no_grad():
                # assume x_new has shape (1, n_features)
                for group in cat_idxs:
                    # gather the slice, pick the max, zero out, set the argmax
                    sub = x_new[0, group]
                    idx = torch.argmax(sub).item()
                    x_new.data[0, group] = 0.0
                    x_new.data[0, group[idx]] = 1.0
        return x_new


    def _calc_max_perturbation(self, x_np, coeff, intercept, delta, target_class, device = None):
        # as per old ROAR
        z = coeff.dot(x_np) + intercept.item()
        p = 1/(1+np.exp(-z))
        target_val = float(target_class.cpu().numpy())      # now a scalar
        grad_W     = (p - target_val) * x_np               # shape (n,)
        grad_b     = (p - target_val)                      # Python float

        # build g as a 1-D array of length n+1
        g = np.concatenate([grad_W, np.array([grad_b])])
        c = -g
        bounds = [(-delta, delta)] * g.size
        res = linprog(c, bounds=bounds, method='highs')
        delta = res.x
        return delta[:-1], delta[-1]

    def _generation_method(
        self,
        instance,
        column_name="target",
        neg_value=0,
        lambda_param=0.01,
        lr=0.01,
        delta=0.02,
        norm=1,
        t_max_min=0.5,
        loss_type="BCE",
        binary_cat_features=True,
        seed=42,
        loss_threshold=0.001,
        max_iter=2000,
        device=None,
        **kwargs
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        np.random.seed(seed)
        if isinstance(instance, pd.Series):
            instance = pd.DataFrame(instance).T
        cat_feature_indices = []
        if hasattr(self.task.training_data, 'categorical'):
            for col in self.task.training_data.categorical:
                if col in instance.columns:
                    cat_feature_indices.append(instance.columns.get_loc(col))

        model = self.task.model.get_torch_model()
        # print(model.children())
        is_linear = self.task.model.is_linear_only#all(isinstance(m, (nn.Linear)) for m in model.children())

        if is_linear:
            layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
            linear = layers[-1] if len(layers) > 0 else None

            if linear is None:
                raise ValueError("No linear layer found in model")
            
            W = linear.weight.data.cpu().numpy()
            b = linear.bias.data.cpu().numpy()

            if W.shape[0] > 1:
                coeff = W[1] - W[0]
                intercept = b[1] - b[0]
            else:
                coeff = W.flatten()
                intercept = b.flatten()[0]
            #print("coeff", coeff)
            coeffs = np.vstack([coeff] * instance.shape[0])
            intercepts = np.array([intercept] * instance.shape[0])

        else:
            coeffs, intercepts = self._get_lime_coefficients(instance, neg_value)
            coeff = coeffs[0]
            intercept = intercepts[0]

        coeff = torch.tensor(coeff, dtype=torch.float32, device=device)
        intercept = torch.tensor([intercept], dtype=torch.float32, device=device)
        x = torch.tensor(instance.values.astype(float), dtype=torch.float32, device=device)
        
        target = torch.tensor([1 - neg_value], dtype=torch.float32, device=device)
        loss_fn = nn.BCELoss() if loss_type=="BCE" else nn.MSELoss()
        
        x_new = Variable(x.clone(), requires_grad=True)
        optimizer = optim.Adam([x_new], lr=lr, amsgrad=True)
        loss = torch.tensor(0.0, device=device)
        loss_diff = loss_threshold +1
        iterations=0
        t0 = datetime.datetime.now()
        t_max = datetime.timedelta(minutes=t_max_min)
        
        while loss_diff>loss_threshold and iterations<max_iter:
            
            loss_prev = loss.clone().detach()
            
            x_new = self._reconstruct_encoding_constraints(x_new, cat_feature_indices, binary_cat_features)
            x_np = x_new.squeeze().detach().cpu().numpy()
            
            delta_W, delta_W0 = self._calc_max_perturbation(x_np, coeff.cpu().numpy(), intercept, delta, target, device)
            delta_W = torch.tensor(delta_W, dtype=torch.float32, device=device)
            delta_W0 = torch.tensor(delta_W0, dtype=torch.float32, device=device)
            
            optimizer.zero_grad()
            
            f_x_new = torch.sigmoid((coeff + delta_W) @ x_new.squeeze() + intercept + delta_W0)

            
            if loss_type=="MSE":
                f_x_new = torch.log(f_x_new/(1-f_x_new))
            
            cost = torch.sum(torch.abs(x_new - x)) if norm==1 else torch.norm(x_new-x, p=norm)
            
            loss = loss_fn(f_x_new, target) + lambda_param*cost
            loss.backward()

            optimizer.step()
            loss_diff = torch.abs(loss_prev-loss).item()
            iterations+=1
            
            if datetime.datetime.now()-t0>t_max:
                print("Timeout - ROAR didn't converge")
                break
        

        with torch.no_grad():
            x_new_enc = self._reconstruct_encoding_constraints(x_new, cat_feature_indices, binary_cat_features)
        return pd.DataFrame(x_new_enc.cpu().detach().numpy(), columns=instance.columns)

    def getCandidates(self):
        return pd.DataFrame()
