
from __future__ import annotations

import copy
from typing import Any, List, Sequence, Union, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn

from nam.config import defaults
from nam.models import NAM

TensorLike = Union[pd.DataFrame, pd.Series, np.ndarray, torch.Tensor]


class NamAdapter:
    """
    Adapter for using a Neural Additive Model (NAM) as a PyTorch model.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        *,
        config: Optional[Any] = None,
        hidden_dim: Union[int, Sequence[int], None] = None,
        num_basis_functions: Optional[int] = None,
        activation: Optional[str] = None,
        dropout: Optional[float] = None,
        feature_dropout: Optional[float] = None,
        lr: Optional[float] = None,
        batch_size: Optional[int] = None,
        early_stopping: Optional[bool] = True,
        patience: Optional[int] = 50,
        seed: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        mlp_l2_reg: Optional[float] = 0.001,
        **extra_cfg: Any,
    ):
        # ---- Bookkeeping -------------------------------------------------
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.binary = (self.output_dim == 1)
        self._device = torch.device(device)
        self.mlp_l2_reg = mlp_l2_reg

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # ---- start from NAM defaults() ----------------------------------
        cfg = defaults()

        # ---- merge: config (if provided) -> cfg -------------------------
        if config is not None:
            # copy all public attributes from provided config onto cfg
            for k in dir(config):
                if k.startswith("_"):
                    continue
                v = getattr(config, k)
                try:
                    setattr(cfg, k, v)
                except Exception:
                    pass

        # ---- kwargs override config/defaults ----------------------------
        def ovr(name, val):
            if val is not None:
                setattr(cfg, name, val)

        # mapping/compat:
        # hidden_dim (list/int) is used to set num_units per-feature; num_basis_functions can also be used
        # If both provided, hidden_dim takes precedence; fall back to num_basis_functions; then cfg.num_basis_functions.
        if hidden_dim is not None:
            if isinstance(hidden_dim, (list, tuple)):
                hidden_list = list(hidden_dim)
            else:
                hidden_list = [int(hidden_dim)]
        else:
            hidden_list = []
        

        ovr("activation", activation)
        ovr("num_basis_functions", num_basis_functions)
        ovr("dropout", dropout)
        ovr("feature_dropout", feature_dropout)
        ovr("lr", lr)
        ovr("batch_size", batch_size)

        cfg.regression = False  # classification; we'll set loss accordingly
        cfg.seed = seed if seed is not None else cfg.seed

        # early stopping knobs (kept on adapter, not required by NAM config)
        self.early_stopping = bool(early_stopping)
        self.patience = int(patience) if patience is not None else 0

        # stash any extra user-supplied fields onto cfg
        for k, v in extra_cfg.items():
            try:
                setattr(cfg, k, v)
            except Exception:
                pass

        # ---- Build NAM core --------------------------------------------
        self._cfg = cfg
        self._cfg.device = self._device

        # decide per-feature units
        if hidden_list is not None and len(hidden_list) >= 1:
            per_feature_units = int(hidden_list[0])
        else:
            per_feature_units = int(getattr(cfg, "num_basis_functions", 1000))

        self._model: NAM = NAM(
            cfg,
            name="NamAdapter",
            num_inputs=self.input_dim,
            num_units=[per_feature_units] * self.input_dim,
        ).to(self._device)

        # ---- Loss & criterion ------------------------------------------
        # BCE-with-logits if binary output, CE if multiclass
        self.hidden_dim = hidden_dim if isinstance(hidden_dim, list) else [hidden_dim]
        self._criterion = nn.BCEWithLogitsLoss() if self.binary else nn.CrossEntropyLoss()

    # --------------------------- TRAINING --------------------------------

    def train(
        self,
        X: TensorLike,
        y: TensorLike,
        X_val: Optional[TensorLike] = None,
        y_val: Optional[TensorLike] = None,  # Fixed typo
        epochs: int = 100,
        batch_size: Optional[int] = None,
        to_print: bool = False,
        verbose: Optional[int] = None,
        **_,  # Ignore extra kwargs
    ) -> None:
        """
        Train using PyTorch-Lightning when available; otherwise use a simple PyTorch loop.

        Arguments match your pipeline: `to_print` and/or `verbose` control logs.
        """
        if batch_size is None:
            batch_size = int(getattr(self._cfg, "batch_size", 1024))
        log_ok = bool(to_print) or (verbose is not None and verbose >= 1)

        try:
            # --- Lightning path (matches your previous implementation) ---
            from torch.utils.data import TensorDataset, DataLoader
            from nam.trainer import LitNAM
            import pytorch_lightning as pl

            def _to_tensor(a, reshape: bool = False):
                if isinstance(a, torch.Tensor):
                    t = a.float()
                elif isinstance(a, (pd.DataFrame, pd.Series)):
                    t = torch.tensor(a.values, dtype=torch.float32)
                else:
                    t = torch.tensor(a, dtype=torch.float32)
                return t.view(-1, 1) if reshape else t

            X_t = _to_tensor(X)
            if self.binary:
                y_t = _to_tensor(y, reshape=True)
            else:
                # multiclass: expect class ids
                y_np = np.asarray(y)
                y_t = torch.tensor(y_np, dtype=torch.long)

            train_loader = DataLoader(
                TensorDataset(X_t, y_t),
                batch_size=batch_size,
                shuffle=True,
            )

            if X_val is not None and y_val is not None:
                Xv_t = _to_tensor(X_val)
                if self.binary:
                    yv_t = _to_tensor(y_val, reshape=True)
                else:
                    yv_np = np.asarray(y_val)
                    yv_t = torch.tensor(yv_np, dtype=torch.long)
                val_loader = DataLoader(TensorDataset(Xv_t, yv_t), batch_size=batch_size, shuffle=False)
            else:
                val_loader = None

            lit_module = LitNAM(self._model.config, self._model)
            trainer = pl.Trainer(
                max_epochs=int(epochs),
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=log_ok,
                devices=[self._device.index] if self._device.type == "cuda" else 1,
                accelerator="gpu" if self._device.type == "cuda" else "cpu",
            )
            trainer.fit(lit_module, train_loader, val_loader)
            return

        except Exception:
            # --- Fallback: simple PyTorch training loop -------------------
            pass

        # Fallback loop
        self._simple_train_loop(
            X, y, X_val=X_val, y_val=y_val, epochs=epochs, batch_size=batch_size, log_ok=log_ok
        )

    def _simple_train_loop(self, X, y, X_val=None, y_val=None, epochs=100, batch_size=1024, log_ok=False):
        from torch.utils.data import TensorDataset, DataLoader

        def _to_tensor(a, reshape: bool = False):
            if isinstance(a, torch.Tensor):
                t = a.float()
            elif isinstance(a, (pd.DataFrame, pd.Series)):
                t = torch.tensor(a.values, dtype=torch.float32)
            else:
                t = torch.tensor(a, dtype=torch.float32)
            return t.view(-1, 1) if reshape else t

        X_t = _to_tensor(X).to(self._device)
        if self.binary:
            y_t = _to_tensor(y, reshape=True).to(self._device)
        else:
            y_np = np.asarray(y)
            y_t = torch.tensor(y_np, dtype=torch.long, device=self._device)

        train_loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

        # simple Adam
        optim = torch.optim.Adam(self._model.parameters(), lr=float(getattr(self._cfg, "lr", 3e-4)))  # Removed weight_decay for explicit L2 reg

        best_val = float("inf")
        wait = 0
        best_state = None

        for ep in range(int(epochs)):
            self._model.train()
            for xb, yb in train_loader:
                optim.zero_grad()
                logits, _ = self._model(xb)
                # cross-entropy needs long labels; BCE needs float (already set)
                if not self.binary:
                    data_loss = self._criterion(logits, yb)
                else:
                    # For binary classification, squeeze target to match logits shape
                    data_loss = nn.BCEWithLogitsLoss()(logits, yb.squeeze())
                
                # Compute L2 regularization term: λ/2 * ||W||²
                l2_reg = 0.0
                for param in self._model.parameters():
                    l2_reg += torch.sum(param ** 2)
                l2_reg = (self.mlp_l2_reg / 2.0) * l2_reg
                
                # Combined loss
                loss = data_loss + l2_reg
                loss.backward()
                optim.step()

            val_loss = self.compute_loss(X_val, y_val) if (X_val is not None and y_val is not None) else self.compute_loss(X, y)
            if log_ok:
                print(f"[NAM] epoch {ep+1}/{epochs}  val_loss={val_loss:.4f}")

            if self.early_stopping and (X_val is not None and y_val is not None):
                if val_loss + 1e-7 < best_val:
                    best_val = val_loss
                    wait = 0
                    best_state = {k: v.detach().cpu() for k, v in self._model.state_dict().items()}
                else:
                    wait += 1
                    if wait >= self.patience:
                        if log_ok:
                            print(f"[NAM] early stop at epoch {ep+1}")
                        if best_state is not None:
                            self._model.load_state_dict(best_state)
                            self._model.to(self._device)
                        break

    # --------------------------- EVALUATION -------------------------------

    def compute_loss(self, X: TensorLike, y: TensorLike) -> float:
        """
        Compute the combined loss: BCE/CE loss + L2 regularization (λ/2 * ||W||²).
        """
        if X is None or y is None:
            return float("inf")
        X_t = self._to_tensor(X)
        if self.binary:
            y_t = self._to_tensor(y).view(-1)
            criterion = nn.BCEWithLogitsLoss()
        else:
            y_np = np.asarray(y)
            y_t = torch.tensor(y_np, dtype=torch.long, device=self._device)
            criterion = nn.CrossEntropyLoss()

        self._model.eval()
        with torch.no_grad():
            logits, _ = self._model(X_t)
            data_loss = criterion(logits, y_t)
            
            # Compute L2 regularization term: λ/2 * ||W||²
            l2_reg = 0.0
            for param in self._model.parameters():
                l2_reg += torch.sum(param ** 2)
            l2_reg = (self.mlp_l2_reg / 2.0) * l2_reg
            
            # Combined loss
            combined_loss = data_loss + l2_reg
            return float(combined_loss.item())

    def predict(self, X: TensorLike) -> pd.DataFrame:
        X_t = self._to_tensor(X)
        self._model.eval()
        with torch.no_grad():
            logits, _ = self._model(X_t)
            if self.binary:
                probs = torch.sigmoid(logits).cpu().numpy().reshape(-1, 1)
                return pd.DataFrame(probs, index=self._maybe_index(X), columns=["proba"])
            else:
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                # return DataFrame with columns 0..C-1
                cols = list(range(probs.shape[1]))
                return pd.DataFrame(probs, index=self._maybe_index(X), columns=cols)

    def predict_proba(self, X: TensorLike) -> pd.DataFrame:
        return self.predict(X)
    
    def predict_single(self, x: TensorLike) -> int:
        return int(self.predict(x).iloc[0, 0] >= 0.5)

    # --------------------------- UTILS -----------------------------------

    def get_torch_model(self) -> nn.Module:
        return self._model

    @property
    def device(self):
        return self._device
    
    def remove_dropout(self) -> "NamAdapter":
        """
        Rebuild the wrapped NAM with dropout disabled, load original weights,
        and physically remove any Dropout-like modules.
        """
        import copy
        import torch.nn as nn

        new_adapter = copy.deepcopy(self)

        # 1) Copy and tweak config
        cfg = copy.deepcopy(self._model.config)
        if hasattr(cfg, "dropout"):
            cfg.dropout = 0.0
        for flag in ["delete_dropout"]:  # common alt flags
            if hasattr(cfg, flag):
                setattr(cfg, flag, True)

        # 2) Rebuild a fresh core with same shape but no dropout in config
        name = getattr(self._model, "name", "NamAdapter")
        
        # Get the number of units per feature from the existing model
        if hasattr(self._model, "feature_nns") and len(self._model.feature_nns) > 0:
            # Extract the number of units from the first feature network
            first_feature_nn = self._model.feature_nns[0]
            if hasattr(first_feature_nn, "hidden_units"):
                per_feature_units = first_feature_nn.hidden_units
            elif hasattr(first_feature_nn, "num_units"):
                per_feature_units = first_feature_nn.num_units
            else:
                # Extract the actual number of basis functions from the trained model
                try:
                    first_layer = self._model.feature_nns[0].model[0]
                    if hasattr(first_layer, 'weight'):
                        per_feature_units = first_layer.weight.shape[1]
                    else:
                        params = list(first_layer.parameters())
                        if params:
                            per_feature_units = params[0].shape[1]
                        else:
                            per_feature_units = self._model.num_units[0] if hasattr(self._model, "num_units") else 16
                except (AttributeError, IndexError):
                    per_feature_units = self._model.num_units[0] if hasattr(self._model, "num_units") else 16
        else:
            per_feature_units = self.hidden_dim[0] if self.hidden_dim else 16
            
        new_core = NAM(
            cfg,
            name=name,
            num_inputs=self.input_dim,
            num_units=[per_feature_units] * self.input_dim,
        ).to(self._device)

        # 3) Load weights/buffers (dropout has no params, so strict load should pass)
        new_core.load_state_dict(self._model.state_dict(), strict=True)

        # 4) Physically remove any Dropout-like modules (in case the lib still builds them)
        def _strip(m: nn.Module):
            DROPS = (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout, nn.FeatureAlphaDropout)
            for n, c in list(m.named_children()):
                if isinstance(c, DROPS):
                    setattr(m, n, nn.Identity())
                else:
                    _strip(c)
        _strip(new_core)

        # 5) Preserve train/eval and config; swap into adapter
        new_core.train(self._model.training)
        if hasattr(new_core, "config") and hasattr(new_core.config, "dropout"):
            new_core.config.dropout = 0.0

        new_adapter._model = new_core
        new_adapter.delete_dropout = True
        return new_adapter

    @staticmethod
    def _strip_dropout_modules_(module: torch.nn.Module) -> None:
        DROPOUTS = (
            nn.Dropout, nn.Dropout2d, nn.Dropout3d,
            nn.AlphaDropout, nn.FeatureAlphaDropout
        )
        for name, child in list(module.named_children()):
            if isinstance(child, DROPOUTS):
                setattr(module, name, nn.Identity())
            else:
                NamAdapter._strip_dropout_modules_(child)

    # ---- tensor helpers ----
    def _to_tensor(self, x: TensorLike) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            t = x.float()
        elif isinstance(x, (pd.DataFrame, pd.Series)):
            t = torch.tensor(x.values, dtype=torch.float32)
        else:
            t = torch.tensor(x, dtype=torch.float32)
        return t.to(self._device)

    def _maybe_index(self, x: TensorLike):
        if isinstance(x, pd.DataFrame):
            return x.index
        if isinstance(x, pd.Series):
            return x.index
        return None

    def __repr__(self):
        return f"NamAdapter({self._model})"


# ========================= Extras you already had (kept) ======================

import torch.nn.functional as F

def extract_nam_bias(nam: nn.Module) -> torch.Tensor:
    b = getattr(nam, "bias", None) or getattr(nam, "_bias", None)
    if b is None:
        for name, p in nam.named_parameters():
            if p.shape == torch.Size([1]) and "bias" in name:
                b = p
                break
    if b is None:
        raise AttributeError("NAM bias not found; expected a scalar bias parameter.")
    return b

class FinalLayerOnly(nn.Module):
    """
    Pure linear head over Z=f(x): logits = Z @ 1_F + b.
    No backbone calls. Input must be Z of shape (B, F).
    """
    def __init__(self, num_features: int, bias: torch.Tensor, share_bias: bool = False):
        super().__init__()
        self.input_dim  = num_features
        self.output_dim = 1
        self.hidden_dim = []
        self.layers     = [self.input_dim, self.output_dim]

        lin = nn.Linear(self.input_dim, 1, bias=True)
        with torch.no_grad():
            lin.weight.fill_(1.0)
            if isinstance(bias, nn.Parameter):
                lin.bias.copy_(bias.detach())
            else:
                lin.bias.copy_(torch.as_tensor(float(bias)))
        lin.weight.requires_grad = False

        dev = bias.device if isinstance(bias, torch.Tensor) else None
        dt  = bias.dtype  if isinstance(bias, torch.Tensor) else torch.float32
        if dev is not None:
            lin.to(device=dev, dtype=dt)

        self._core = nn.Sequential(lin)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        assert isinstance(Z, torch.Tensor) and Z.dim() == 2 and Z.size(1) == self.input_dim
        return self._core(Z).squeeze(1)

    @torch.no_grad()
    def predict_logits(self, Z, device=None, return_numpy=False):
        self.eval()
        Zt = torch.as_tensor(Z, dtype=torch.float32, device=device) if not isinstance(Z, torch.Tensor) else Z
        logits = self.forward(Zt)
        return logits.detach().cpu().numpy() if return_numpy else logits

    @torch.no_grad()
    def predict_proba(self, Z, device=None, return_numpy=False):
        self.eval()
        Zt = torch.as_tensor(Z, dtype=torch.float32, device=device) if not isinstance(Z, torch.Tensor) else Z
        probs = torch.sigmoid(self.forward(Zt))
        return probs.detach().cpu().numpy() if return_numpy else probs

    @torch.no_grad()
    def predict(self, Z, device=None, return_numpy=False):
        self.eval()
        Zt = torch.as_tensor(Z, dtype=torch.float32, device=device) if not isinstance(Z, torch.Tensor) else Z
        probs = torch.sigmoid(self.forward(Zt)).detach().cpu().numpy().reshape(-1, 1)
        if return_numpy:
            return probs
        if pd is not None:
            return pd.DataFrame(probs, columns=["proba"])
        return probs

class NAMPenult(nn.Module):
    """Returns per-feature contributions Z (B,F) from a NAM."""
    def __init__(self, nam: nn.Module):
        super().__init__()
        assert hasattr(nam, "feature_nns"), "Expected NAM with .feature_nns"
        self.nam = nam
        self.feature_nns = nam.feature_nns  # ModuleList of FeatureNNs

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        outs = []
        for j, fnn in enumerate(self.feature_nns):
            z_j = fnn(X[:, j].unsqueeze(1))        # (B,1)
            outs.append(z_j.squeeze(1))            # (B,)
        return torch.stack(outs, dim=1)            # (B,F)

class NAMFreeHead(nn.Module):
    """Linear head: logit = Z @ w + b, with convenient pack/unpack of θ=[w;b]."""
    def __init__(self, num_features: int, bias_init: torch.Tensor, device, dtype):
        super().__init__()
        self.lin = nn.Linear(num_features, 1, bias=True, device=device, dtype=dtype)
        with torch.no_grad():
            self.lin.weight.fill_(1.0)
            self.lin.bias.copy_(bias_init.reshape(()))

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        return self.lin(Z).squeeze(1)

    @torch.no_grad()
    def pack_theta(self) -> torch.Tensor:
        return torch.cat([self.lin.weight.view(-1), self.lin.bias.view(-1)], dim=0)

    @torch.no_grad()
    def load_theta(self, theta: torch.Tensor) -> None:
        F = self.lin.in_features
        w, b = theta[:F].view(1, F), theta[F:].view(1)
        self.lin.weight.copy_(w)
        self.lin.bias.copy_(b)