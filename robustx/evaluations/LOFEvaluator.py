import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from robustx.evaluations.CEEvaluator import CEEvaluator


class LOFEvaluator(CEEvaluator):
    """
    Local Outlier Factor (LOF) evaluator implemented per Definition 4.

    Definitions (for a dataset S and query x, with k neighbors and lp distance):
      - L_k(x): k-nearest neighbors of x in S
      - r_dk(x, x') = max{ δ(x, x'), d_k(x') }
        where δ is the lp distance and d_k(x') is x'’s distance to its
        k-th nearest neighbor in S
      - lrd_k(x) = |L_k(x)| / sum_{x' in L_k(x)} r_dk(x, x')
      - LOF_{k,S}(x) = (1/|L_k(x)|) * sum_{x' in L_k(x)} lrd_k(x') / lrd_k(x)

    By default, evaluate() returns the **mean LOF score** over the provided
    counterfactuals (a float). Values near 1 indicate density comparable to
    neighbors; values >1 indicate lower density (more outlier-like), values <1
    indicate higher density (more interior).
    """

    def __init__(self, task, n_neighbors: int = 20, p: int = 2, epsilon: float = 1e-12):
        """
        Parameters
        ----------
        task : object
            Must provide `task.training_data.X` (pd.DataFrame or np.ndarray).
        n_neighbors : int
            k in LOF. Will be clipped to (#train - 1).
        p : int
            lp norm for δ(·,·). p=2 -> Euclidean.
        epsilon : float
            Numerical stability to avoid division by zero and check for self-matches.
        """
        super().__init__(task)
        self.n_neighbors = n_neighbors
        self.p = p
        self.epsilon = epsilon

        # Cached artifacts
        self._X_train = None
        self._feature_columns = None
        self._k = None
        self._nn_train = None          # KNN model on S
        self._k_distance_train = None  # d_k(x') for each x' in S
        self._lrd_train = None         # lrd_k(x') for each x' in S

    # ------------------------------- utils -------------------------------- #
    def _as_array(self, X):
        """Convert to numpy array, preserving training feature order."""
        if isinstance(X, pd.DataFrame):
            if self._feature_columns is None:
                self._feature_columns = list(X.columns)
                return X.values.astype(np.float64)
            cols = [c for c in self._feature_columns if c in X.columns]
            return X[cols].values.astype(np.float64)
        return np.asarray(X, dtype=np.float64)

    def _fit_if_needed(self):
        """Precompute neighbor structures and lrd_k on S."""
        if self._nn_train is not None:
            return

        X_train = self._as_array(self.task.training_data.X)
        n = X_train.shape[0]
        if n < 2:
            raise ValueError("LOF requires at least 2 training samples.")

        k = min(max(1, self.n_neighbors), n - 1)
        self._k = k
        self._X_train = X_train

        # KNN on S; for training points we query k+1 to drop self-neighbor
        self._nn_train = NearestNeighbors(
            n_neighbors=k + 1,
            metric="minkowski",
            p=self.p,
            algorithm="auto",
        ).fit(X_train)

        # Neighbors of each training point (exclude self)
        dists, idxs = self._nn_train.kneighbors(X_train, n_neighbors=k + 1, return_distance=True)
        dists, idxs = dists[:, 1:], idxs[:, 1:]

        # d_k(x') for every x' in S
        self._k_distance_train = dists[:, -1].copy()

        # lrd_k(x') for every x' in S
        kdist_neighbors = self._k_distance_train[idxs]      # (n, k)
        reach_dists = np.maximum(dists, kdist_neighbors)    # (n, k)
        denom = np.maximum(reach_dists.sum(axis=1), self.epsilon)
        self._lrd_train = k / denom                         # (n,)

    # -------------------------------- API --------------------------------- #
    def evaluate(self, counterfactuals, as_labels: bool = False, threshold: float = 1.0, valid_val=1, **kwargs) -> float:
        """
        Parameters
        ----------
        counterfactuals : pd.DataFrame or np.ndarray
            Rows are counterfactual points to score.
        as_labels : bool
            If True, convert LOF to labels using (LOF > threshold) → -1 else +1,
            and return the mean label. If False (default), return the mean LOF.
        threshold : float
            Threshold used only when as_labels=True (default 1.0).

        Returns
        -------
        float
            Mean LOF (if as_labels=False) or mean label in {+1, -1} (if True).
        """
        self._fit_if_needed()
        
        orig_preds = (self.task.model.predict(counterfactuals).iloc[:, 0].to_numpy() >= 0.5).astype(int)
        # Mask out any CEs that weren't even valid on the original model
        mask_valid = (orig_preds == valid_val)
        if not mask_valid.any():
            return 1000.0  # Return a high LOF score if no valid counterfactuals
        
        valid_counterfactuals = counterfactuals[mask_valid]

        # MODIFIED: Delegate core computation to the helper method
        lof_scores = self._compute_lof_array(valid_counterfactuals)

        if as_labels:
            labels = np.where(lof_scores > threshold, -1.0, 1.0)
            return float(labels.mean())

        return float(lof_scores.mean())

    def lof_per_point(self, counterfactuals):
        """
        Convenience method: return the LOF score for each counterfactual point.
        """
        self._fit_if_needed()
        return self._compute_lof_array(counterfactuals)

    # MODIFIED: This is now the primary method for LOF calculation.
    def _compute_lof_array(self, X):
        """
        Computes the LOF score for each point in X, robustly handling
        whether a point is in the training set or not.
        """
        X_cf = self._as_array(X)
        if X_cf.ndim == 1:
            X_cf = X_cf.reshape(1, -1)
        
        if X_cf.shape[0] == 0:
            return np.array([])

        k = self._k
        
        # We now query for k+1 neighbors for all points. This allows us to
        # detect and handle "self-matches" for points already in the training set.
        nn_query = NearestNeighbors(
            n_neighbors=k + 1,
            metric="minkowski",
            p=self.p,
            algorithm="auto",
        ).fit(self._X_train)

        dists, idxs = nn_query.kneighbors(X_cf, return_distance=True)

        # Identify which points are self-matches (distance to nearest neighbor is ~0)
        is_self_match = dists[:, 0] < self.epsilon
        not_self_match = ~is_self_match
        
        # Initialize arrays for the final k neighbors
        m = X_cf.shape[0]
        final_dists = np.empty((m, k))
        final_idxs = np.empty((m, k), dtype=idxs.dtype)

        # For self-matches, discard the first neighbor and take the next k
        if np.any(is_self_match):
            final_dists[is_self_match] = dists[is_self_match, 1:]
            final_idxs[is_self_match] = idxs[is_self_match, 1:]
            
        # For new points, take the first k neighbors
        if np.any(not_self_match):
            final_dists[not_self_match] = dists[not_self_match, :-1]
            final_idxs[not_self_match] = idxs[not_self_match, :-1]

        # --- The rest of the LOF calculation proceeds as before ---

        # lrd_k(x) for each point in X
        kdist_neighbors = self._k_distance_train[final_idxs]
        reach_dists = np.maximum(final_dists, kdist_neighbors)
        denom_cf = np.maximum(reach_dists.sum(axis=1), self.epsilon)
        lrd_cf = k / denom_cf

        # LOF_{k,S}(x) for each point in X
        lrd_neighbors = self._lrd_train[final_idxs]
        lof_scores = (lrd_neighbors / lrd_cf[:, None]).mean(axis=1)
        
        return lof_scores