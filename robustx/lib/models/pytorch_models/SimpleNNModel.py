import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from robustx.lib.models.BaseModel import BaseModel
from sklearn.linear_model import LogisticRegression
import torch

from copy import deepcopy

DEVICE = torch.device("cpu")

# DEVICE = torch.device(
#     "cuda" if torch.cuda.is_available()
#     else "mps"   if torch.backends.mps.is_available()
#     else "cpu"
# )

import torch.nn as nn
from copy import deepcopy




class SimpleNNModel(BaseModel):
    """
    A simple neural network model using PyTorch. This model can be customized with different numbers of hidden layers and units.
    When hidden_dim is empty (no hidden layers), it uses scikit-learn's LogisticRegression for training efficiency.

    Attributes
    ----------
    input_dim: int
        The number of input features for the model.
    hidden_dim: list of int
        The number of units in each hidden layer. An empty list means no hidden layers.
    output_dim: int
        The number of output units for the model.
    criterion: nn.BCELoss
        The loss function used for training.
    optimizer: optim.Adam
        The optimizer used for training the model.

    Methods
    -------
    __create_model() -> nn.Sequential:
        Creates and returns the PyTorch model architecture.

    train(X: pd.DataFrame, y: pd.DataFrame, epochs: int = 100) -> None:
        Trains the model on the provided data for a specified number of epochs.
        Uses LogisticRegression when no hidden layers are present.

    set_weights(weights: Dict[str, torch.Tensor]) -> None:
        Sets custom weights for the model.

    predict(X: pd.DataFrame) -> pd.DataFrame:
        Predicts the outcomes for the provided instances.

    predict_single(x: pd.DataFrame) -> int:
        Predicts the outcome of a single instance and returns an integer.

    evaluate(X: pd.DataFrame, y: pd.DataFrame) -> float:
        Evaluates the model's accuracy on the provided data.

    predict_proba(x: torch.Tensor) -> pd.DataFrame:
        Predicts the probability of outcomes for the provided instances.

    predict_proba_tensor(x: torch.Tensor) -> torch.Tensor:
        Predicts the probability of outcomes for the provided instances using tensor input.

    get_torch_model() -> nn.Module:
        Returns the underlying PyTorch model.
    """

    def __init__(self, input_dim, hidden_dim=None, output_dim=None, seed=None, device=DEVICE, early_stopping=True, patience=10, dropout=0.0, delete_dropout=False, mlp_l2_reg=0.001, **params):
        """
        Initializes the SimpleNNModel with specified dimensions.

        @param input_dim: Number of input features.
        @param hidden_dim: List specifying the number of neurons in each hidden layer.
        @param output_dim: Number of output neurons.
        @param early_stopping: Whether to use early stopping.
        @param patience: Number of epochs to wait for improvement before stopping.
        @param mlp_l2_reg: L2 regularization parameter for Adam optimizer.
        """
        self.delete_dropout = delete_dropout
        self.device = device
        self.dropout = dropout
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else []
        self.output_dim = output_dim
        self.early_stopping = early_stopping
        self.patience = patience
        self.mlp_l2_reg = mlp_l2_reg
        self.is_linear_only = len(self.hidden_dim) == 0
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
        super().__init__(self.__create_model())
        
        self.criterion = nn.BCELoss().to(self.device)
        self.optimizer = optim.Adam(self._model.parameters(), lr=0.001)  # Removed weight_decay for explicit L2 reg
        
        

    def __create_model(self):
        model = nn.Sequential()

        if self.hidden_dim:
            model.append(nn.Linear(self.input_dim, self.hidden_dim[0]))
            model.append(nn.ReLU())
            if not self.delete_dropout:
                model.append(nn.Dropout(self.dropout))

            for i in range(0, len(self.hidden_dim) - 1):
                model.append(nn.Linear(self.hidden_dim[i], self.hidden_dim[i + 1]))
                model.append(nn.ReLU())
                if i < len(self.hidden_dim) - 2:
                    if not self.delete_dropout:
                        model.append(nn.Dropout(self.dropout))
                

            model.append(nn.Linear(self.hidden_dim[-1], self.output_dim))

        else:
            model.append(nn.Linear(self.input_dim, self.output_dim))

        if self.output_dim == 1:
            model.append(nn.Sigmoid())
            
        model.to(self.device)
        model.eval()
        return model

    def _train_with_logistic_regression(self, X, y, not_numpy=True):
        """
        Train using scikit-learn's LogisticRegression when no hidden layers are present.
        
        @param not_numpy: Deprecated parameter kept for backward compatibility. The function 
                         now automatically detects the input type.
        """
        # Automatically determine input type instead of using not_numpy parameter
        if hasattr(X, 'values'):  # pandas DataFrame/Series
            X_np = X.values
            y_np = y.values.ravel() if hasattr(y, 'values') else y.ravel()
        else:  # numpy array
            X_np = X
            y_np = y.ravel() if y.ndim > 1 else y
        
        # Create and train logistic regression with L2 regularization
        # alpha=0.001 corresponds to C=1/0.001=1000 in sklearn
        lr = LogisticRegression(C=1000.0, penalty='l2', max_iter=1000, solver='lbfgs')
        lr.fit(X_np, y_np)
        
        # Extract weights and bias
        weights = lr.coef_[0]  # Shape: (n_features,)
        bias = lr.intercept_[0]  # Scalar
        
        # Find the linear layer in our model
        linear_layer = None
        for module in self._model:
            if isinstance(module, nn.Linear):
                linear_layer = module
                break
        
        if linear_layer is None:
            raise RuntimeError("No linear layer found in model")
        
        # Set the weights and bias in the PyTorch model
        with torch.no_grad():
            linear_layer.weight.data = torch.from_numpy(weights.reshape(1, -1)).float().to(self.device)
            linear_layer.bias.data = torch.from_numpy(np.array([bias])).float().to(self.device)
        
        return lr

    def train(
        self,
        X,
        y,
        X_val=None,
        y_val=None,
        X_test=None,
        y_test=None,
        epochs=100,
        to_print=False,
        desired_loss: float = None,
        not_numpy=True,  # Deprecated parameter kept for backward compatibility
        **kwargs
    ):
        """
        Trains the neural network model and prints training/validation losses.
        If no hidden layers are present, uses LogisticRegression instead of gradient descent.
        If `desired_loss` is provided, stops early when train loss < desired_loss.
        If early_stopping is True and no desired_loss is provided, stops when validation loss 
        doesn't improve for 'patience' epochs.
        
        @param not_numpy: Deprecated parameter kept for backward compatibility. The function 
                         now automatically detects the input type.
        """
        
        # If it's a linear-only model (no hidden layers), use LogisticRegression
        if self.is_linear_only:
            if to_print:
                print("Using LogisticRegression for linear model (no hidden layers)")
            
            lr_model = self._train_with_logistic_regression(X, y, not_numpy=not_numpy)
            
            # Calculate final losses for consistency with neural network training
            self._model.eval()
            
            # Compute training loss - automatically detect input type
            if hasattr(X, 'values'):  # pandas DataFrame/Series
                X_tensor = torch.tensor(X.values, dtype=torch.float32).to(self.device)
                y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1).to(self.device)
            else:  # numpy array
                X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
                y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(self.device)
            
            with torch.no_grad():
                train_outputs = self._model(X_tensor)
                train_data_loss = self.criterion(train_outputs, y_tensor)
                
                # Compute L2 regularization term for final train loss
                train_l2_reg = 0.0
                for param in self._model.parameters():
                    train_l2_reg += torch.sum(param ** 2)
                train_l2_reg = (self.mlp_l2_reg / 2.0) * train_l2_reg
                
                train_loss = (train_data_loss + train_l2_reg).item()
            
            if to_print:
                print(f"Final Train Loss (LogisticRegression): {train_loss:.4f}")
            
            # Compute validation loss if provided
            if X_val is not None and y_val is not None:
                if hasattr(X_val, 'values'):  # pandas DataFrame/Series
                    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32).to(self.device)
                    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1).to(self.device)
                else:  # numpy array
                    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
                    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(self.device)
                
                with torch.no_grad():
                    val_outputs = self._model(X_val_tensor)
                    val_data_loss = self.criterion(val_outputs, y_val_tensor)
                    
                    # Compute L2 regularization term for final validation loss
                    val_l2_reg = 0.0
                    for param in self._model.parameters():
                        val_l2_reg += torch.sum(param ** 2)
                    val_l2_reg = (self.mlp_l2_reg / 2.0) * val_l2_reg
                    
                    val_loss = (val_data_loss + val_l2_reg).item()
                
                if to_print:
                    print(f"Final Validation Loss (LogisticRegression): {val_loss:.4f}")
            
            # Compute test loss if provided
            if X_test is not None and y_test is not None:
                if hasattr(X_test, 'values'):  # pandas DataFrame/Series
                    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(self.device)
                    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1).to(self.device)
                else:  # numpy array
                    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
                    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(self.device)
                
                with torch.no_grad():
                    test_outputs = self._model(X_test_tensor)
                    test_data_loss = self.criterion(test_outputs, y_test_tensor)
                    
                    # Compute L2 regularization term for final test loss
                    test_l2_reg = 0.0
                    for param in self._model.parameters():
                        test_l2_reg += torch.sum(param ** 2)
                    test_l2_reg = (self.mlp_l2_reg / 2.0) * test_l2_reg
                    
                    test_loss = (test_data_loss + test_l2_reg).item()
                
                if to_print:
                    print(f"Test Loss (LogisticRegression): {test_loss:.4f}")
            
            return
        
        # Otherwise, proceed with standard neural network training
        self._model.train()
        
        # Automatically detect input type for training data
        if hasattr(X, 'values'):  # pandas DataFrame/Series
            X_tensor = torch.tensor(X.values, dtype=torch.float32).to(self.device)
            y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1).to(self.device)
        else:  # numpy array
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(self.device)

        losses = {"train_loss": [], "val_loss": []}
        
        # For early stopping
        best_val_loss = float('inf')
        no_improve_count = 0
        best_model_state = None
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()

            # --- forward & backward ---
            outputs = self._model(X_tensor)
            data_loss = self.criterion(outputs, y_tensor)
            
            # Compute L2 regularization term: λ/2 * ||W||²
            l2_reg = 0.0
            for param in self._model.parameters():
                l2_reg += torch.sum(param ** 2)
            l2_reg = (self.mlp_l2_reg / 2.0) * l2_reg
            
            # Combined loss
            loss = data_loss + l2_reg
            loss.backward()
            self.optimizer.step()

            # record
            train_loss = loss.item()
            losses["train_loss"].append(train_loss)

            # optional val-loss
            if X_val is not None and y_val is not None:
                self._model.eval()
                
                # Automatically detect input type for validation data
                if hasattr(X_val, 'values'):  # pandas DataFrame/Series
                    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32).to(self.device)
                    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1).to(self.device)
                else:  # numpy array
                    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
                    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(self.device)
                
                with torch.no_grad():
                    val_outputs = self._model(X_val_tensor)
                    val_data_loss = self.criterion(val_outputs, y_val_tensor)
                    
                    # Compute L2 regularization term for validation
                    val_l2_reg = 0.0
                    for param in self._model.parameters():
                        val_l2_reg += torch.sum(param ** 2)
                    val_l2_reg = (self.mlp_l2_reg / 2.0) * val_l2_reg
                    
                    val_loss = (val_data_loss + val_l2_reg).item()
                losses["val_loss"].append(val_loss)
                self._model.train()
                
                # Check for early stopping (only if desired_loss is None)
                if self.early_stopping and desired_loss is None:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        no_improve_count = 0
                        best_model_state = deepcopy(self._model.state_dict())
                    else:
                        no_improve_count += 1
                        if no_improve_count >= self.patience:
                            if to_print:
                                print(f"Early stopping at epoch {epoch+1}. No improvement for {self.patience} epochs.")
                            # Restore best model
                            if best_model_state is not None:
                                self._model.load_state_dict(best_model_state)
                            break

            # print every 10 epochs
            if to_print and (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}"
                if X_val is not None and y_val is not None:
                    msg += f", Val Loss: {losses['val_loss'][-1]:.4f}"
                print(msg)

            # ** early stopping based on desired_loss - overrides patience-based stopping **
            if desired_loss is not None and train_loss < desired_loss:
                if to_print:
                    print(f"Reached desired_loss={desired_loss:.4f} at epoch {epoch+1}, stopping early.")
                break

        # final-loss printout
        if to_print:
            print(f"\nFinal Train Loss: {losses['train_loss'][-1]:.4f}")
            if X_val is not None and y_val is not None:
                print(f"Final Validation Loss: {losses['val_loss'][-1]:.4f}")

        # optional test-set loss
        if X_test is not None and y_test is not None:
            self._model.eval()
            
            # Automatically detect input type for test data
            if hasattr(X_test, 'values'):  # pandas DataFrame/Series
                X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(self.device)
                y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1).to(self.device)
            else:  # numpy array
                X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
                y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(self.device)
            
            with torch.no_grad():
                test_outputs = self._model(X_test_tensor)
                test_data_loss = self.criterion(test_outputs, y_test_tensor)
                
                # Compute L2 regularization term for test loss
                test_l2_reg = 0.0
                for param in self._model.parameters():
                    test_l2_reg += torch.sum(param ** 2)
                test_l2_reg = (self.mlp_l2_reg / 2.0) * test_l2_reg
                
                test_loss = (test_data_loss + test_l2_reg).item()
            print(f"Test Loss: {test_loss:.4f}")
        
        # leave the model in eval mode
        self._model.eval()


    def set_weights(self, weights):
        """
        Sets custom weights for the model.

        @param weights: Dictionary containing weights and biases for each layer.
        """
        # Initialize layer index for Sequential model
        layer_idx = 0
        for i, layer in enumerate(self._model):
            if isinstance(layer, nn.Linear):
                # Extract weights and biases from the weights dictionary
                with torch.no_grad():
                    layer.weight = nn.Parameter(weights[f'fc{layer_idx}_weight'])
                    layer.bias = nn.Parameter(weights[f'fc{layer_idx}_bias'])
                layer_idx += 1
        self._model.eval()

    def predict(self, X) -> pd.DataFrame:
        """
        Predicts outcomes for the given input data.

        @param X: Input data as a pandas DataFrame or torch tensor.
        @return: Predictions as a pandas DataFrame.
        """
        if not isinstance(X, torch.Tensor):
            # X = torch.tensor(X.values, dtype=torch.float32)
            if hasattr(X, 'values'):  # pandas DataFrame/Series
                X = X.values
            X = torch.from_numpy(X.astype(float)).float().to(self.device)
        #print(X, X.type(), X.dtype)
        #print(print(next(self._model.parameters()).dtype)
        return pd.DataFrame(self._model(X).cpu().detach().numpy())

    def predict_single(self, x) -> int:
        """
        Predicts the outcome for a single instance.

        @param x: Single input instance as a pandas DataFrame or torch tensor.
        @return: Prediction as an integer (0 or 1).
        """
        if not isinstance(x, torch.Tensor):
            # x = torch.tensor(x.values, dtype=torch.float32)
            x = torch.from_numpy(x.values.astype(float)).float()
        return 0 if self.predict_proba(x).iloc[0, 0] > 0.5 else 1

    def evaluate(self, X, y):
        """
        Evaluates the model's accuracy.

        @param X: Feature variables as a pandas DataFrame.
        @param y: Target variable as a pandas DataFrame.
        @return: Accuracy of the model as a float.
        """
        predictions = self.predict(X)
        accuracy = (predictions.view(-1) == torch.tensor(y.values)).float().mean()
        return accuracy.item()
    
    def compute_accuracy(self, X_test, y_test):
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test).to(self.device)
            y_tensor = torch.FloatTensor(y_test).view(-1, 1).to(self.device)
            y_pred = self._model(X_tensor)
            y_pred_classes = (y_pred > 0.5).float()
            accuracy = (y_pred_classes.view(-1) == y_tensor.view(-1)).cpu().float().mean().item()
            
        return accuracy

    def predict_proba(self, x: torch.Tensor) -> pd.DataFrame:
        """
        Predicts probabilities of outcomes.

        @param x: Input data as a torch tensor.
        @return: Probabilities of each outcome as a pandas DataFrame.
        """
        if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
            x = torch.tensor(x.values, dtype=torch.float32) #.to(self.device)
        elif isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        res = self._model(x.to(self.device))
        res = pd.DataFrame(res.cpu().detach().numpy())

        temp = res[0]

        # The probability that it is 0 is 1 - the probability returned by model
        res[0] = 1 - res[0]

        # The probability it is 1 is the probability returned by the model
        res[1] = temp
        return res

    def predict_proba_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predicts probabilities of outcomes using the model.

        @param x: Input data as a torch tensor.
        @return: Probabilities of each outcome as a torch tensor.
        """
        return self._model(x)

    def get_torch_model(self):
        """
        Retrieves the underlying PyTorch model.

        @return: The PyTorch model.
        """
        return self._model
    
    def __repr__(self):
        return str(self._model)
    
    
    @property
    def output_layer(self) -> nn.Linear:
        """
        Return the final nn.Linear layer (the one directly before any activation).
        """
        # nn.Sequential is iterable
        for module in reversed(list(self._model)):
            if isinstance(module, nn.Linear):
                return module
        raise RuntimeError("No nn.Linear layer found in model")
    
    
    def compute_loss(self, X, y) -> float:
        """
        Compute the combined loss: BCELoss + L2 regularization (λ/2 * ||W||²).

        Parameters
        ----------
        X : pd.DataFrame, pd.Series, np.ndarray or torch.Tensor
            Feature matrix.
        y : pd.DataFrame, pd.Series, np.ndarray or torch.Tensor
            Binary labels (0/1).

        Returns
        -------
        float
            The scalar combined loss.
        """
        # put model in eval mode
        self._model.eval()
        with torch.no_grad():
            # --- inputs ---
            if isinstance(X, (pd.DataFrame, pd.Series)):
                inputs = torch.tensor(X.values, dtype=torch.float32).to(self.device)
            elif isinstance(X, np.ndarray):
                inputs = torch.from_numpy(X).float().to(self.device)
            elif isinstance(X, torch.Tensor):
                inputs = X.float().to(self.device)
            else:
                raise TypeError(f"Unsupported X type: {type(X)}")

            # --- labels ---
            if isinstance(y, (pd.DataFrame, pd.Series)):
                labels = torch.tensor(y.values, dtype=torch.float32).view(-1, 1).to(self.device)
            elif isinstance(y, np.ndarray):
                labels = torch.from_numpy(y).float().view(-1, 1).to(self.device)
            elif isinstance(y, torch.Tensor):
                labels = y.float().view(-1, 1).to(self.device)
            else:
                raise TypeError(f"Unsupported y type: {type(y)}")

            # --- forward & loss ---
            outputs = self._model(inputs)      # after sigmoid
            data_loss = self.criterion(outputs, labels)
            
            # Compute L2 regularization term: λ/2 * ||W||²
            l2_reg = 0.0
            for param in self._model.parameters():
                l2_reg += torch.sum(param ** 2)
            l2_reg = (self.mlp_l2_reg / 2.0) * l2_reg
            
            # Combined loss
            combined_loss = data_loss + l2_reg
            combined_loss = combined_loss.cpu().detach()

        # back to train mode
        self._model.train()
        return combined_loss.item()


def remove_dropout(orig_model: SimpleNNModel) -> SimpleNNModel:
    """
    Return a copy of `orig_model` with all nn.Dropout layers removed,
    and with the same weights/biases on every Linear layer.
    """
    # 1) Build a new SimpleNNModel with the same dims but dropout=False
    new = SimpleNNModel(
        input_dim=orig_model.input_dim,
        hidden_dim=orig_model.hidden_dim,
        output_dim=orig_model.output_dim,
        dropout=0.0,                  # turn OFF dropout
        delete_dropout=True,
        seed=None,                      # don't re-seed
        device=orig_model.device,
        early_stopping=orig_model.early_stopping,
        patience=orig_model.patience
    )

    # 2) Copy weights/bias from all Linear layers in orig → new
    #    We assume the same number & order of Linear layers in both.
    old_linears = [m for m in orig_model.get_torch_model() if isinstance(m, nn.Linear)]
    new_linears = [m for m in new.get_torch_model()  if isinstance(m, nn.Linear)]
    if len(old_linears) != len(new_linears):
        raise RuntimeError("Unexpected mismatch in number of Linear layers")

    for old_l, new_l in zip(old_linears, new_linears):
        new_l.weight.data.copy_(old_l.weight.data)
        new_l.bias.data.copy_(old_l.bias.data)

    return new