

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import KDTree
from functools import lru_cache

from robustx.generators.CEGenerator import CEGenerator
from robustx.robustness_evaluations.DeltaRobustnessEvaluator import DeltaRobustnessEvaluator
from robustx.robustness_evaluations.ModelChangesRobustnessEvaluator import ModelChangesRobustnessEvaluator


class TRex(CEGenerator): # called STCE in the RobustX paper
    """
    A counterfactual explanation generator that uses the T-Rex method for finding robust counterfactual explanations.

    Inherits from the CEGenerator class and implements the _generation_method to find counterfactual examples
    with robustness checks using a specified base method and evaluator. The method uses KDTree for efficient
    nearest neighbor search and iterates over the closest positive instances to evaluate their robustness.

    Attributes:
        kdtree: Pre-built KDTree for efficient nearest neighbor search
        positives_data: DataFrame containing the positive instances
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kdtree = None
        self.positives_data = None
        self.robustness_scores_cache = None
        
    def _build_kdtree(self, column_name: str = "target", prob_threshold: float = 0.5):
        """
        Build a KD-tree from training samples that the current model already
        classifies as positive with probability ≥ `prob_threshold`.

        Parameters
        ----------
        column_name : str
            Name of the target column to drop before feeding data to the tree.
        prob_threshold : float, default 0.5
            Minimum predicted probability for the positive class.
        """
        # ── 1.  Separate features ────────────────────────────────────────────────
        print(f"Data support size: {self.task.data_support.data.shape[0]}")
        feats_df = self.task.data_support.data.drop(columns=[column_name])

        # ── 2.  Model predictions (probability of the positive class) ────────────
        preds_raw = self.task.model.predict(feats_df)        # DataFrame | ndarray | Series
        if isinstance(preds_raw, pd.DataFrame):
            pred_probs = preds_raw.values.ravel()
        else:
            pred_probs = np.asarray(preds_raw).ravel()
        #print(pred_probs)

        # ── 3.  Mask for “positive” points according to the model ────────────────
        mask_pos = pred_probs >= 0.5
        self.positives_data = feats_df.loc[mask_pos].reset_index(drop=True)

        # Sanity-check: ensure we actually have positive samples
        if self.positives_data.empty:
            raise ValueError(
                f"No training points with predicted P(positive) ≥ {prob_threshold}"
            )
            
        print("Positive data: ", self.positives_data.shape)
        # ── 4.  Build KD-tree ────────────────────────────────────────────────────
        self.kdtree = KDTree(self.positives_data.values, leaf_size=10)

    def _precompute_robustness_scores(self, k: int = 200):
        """
        Precompute robustness scores for all positive instances and cache them.
        
        Parameters
        ----------
        k : int
            Number of Gaussian samples to use for stability computation
        """
        print(f"Precomputing robustness scores for {len(self.positives_data)} positive instances with k={k}...")
        
        # Store the k value used for caching
        self._cached_k = k
        
        # Initialize cache as a dictionary mapping instance index to robustness score
        self.robustness_scores_cache = {}
        
        for idx, (_, positive) in enumerate(self.positives_data.iterrows()):
            stability_score = self._compute_stability_score(positive, k=k)
            self.robustness_scores_cache[idx] = stability_score
            
        print(f"Robustness scores precomputed and cached for {len(self.robustness_scores_cache)} instances.")

    @lru_cache(maxsize=None)
    def getCandidatesWithRobustnessScores(self, k: int = 200) -> pd.DataFrame:
        """
        Get positive instances with their precomputed robustness scores.
        This method is cached to avoid recomputing scores.
        
        Parameters
        ----------
        k : int
            Number of Gaussian samples to use for stability computation
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing positive instances with their robustness scores
        """
        # Compute robustness scores if not already computed or if k changed
        if (self.robustness_scores_cache is None or 
            not hasattr(self, '_cached_k') or self._cached_k != k):
            print(f"Computing robustness scores for getCandidatesWithRobustnessScores with k={k}...")
            self._cached_k = k
            self._precompute_robustness_scores(k=k)
            
        # Create a copy of positives_data with robustness scores
        result_df = self.positives_data.copy()
        result_df['robustness_score'] = [self.robustness_scores_cache[idx] 
                                       for idx in range(len(self.positives_data))]
        
        return result_df

    def update_robustness_cache(self, k: int = 200):
        """
        Update the robustness scores cache with a different k value.
        This allows users to change the number of Gaussian samples used for stability computation.
        
        Parameters
        ----------
        k : int
            Number of Gaussian samples to use for stability computation
        """
        print(f"Updating robustness scores cache with k={k}...")
        self._precompute_robustness_scores(k=k)
        # Clear the LRU cache to force recomputation with new parameters
        self.getCandidatesWithRobustnessScores.cache_clear()

    @lru_cache(maxsize=None)
    def getCandidates(self, k: int = 200) -> pd.DataFrame:
        """
        Get positive instances that can be used as counterfactual candidates.
        This method is cached and follows the same pattern as RNCE and LastLayerEllipsoidCE.
        
        Parameters
        ----------
        k : int
            Number of Gaussian samples to use for stability computation
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing positive instances that can serve as counterfactual candidates
        """
        # Compute robustness scores if not already computed or if k changed
        if (self.robustness_scores_cache is None or 
            not hasattr(self, '_cached_k') or self._cached_k != k):
            print(f"Computing robustness scores for getCandidates with k={k}...")
            self._cached_k = k
            self._precompute_robustness_scores(k=k)
        
        # Return just the positive instances (without robustness scores column)
        return self.positives_data.copy()

    def _compute_stability_score(self, xp, k: int = 200) -> float:
        """
        Compute the stability score for a given instance.
        This is the core logic extracted from counterfactual_stability method.
        
        Parameters
        ----------
        xp : pd.Series
            The instance for which to compute stability score
        k : int
            Number of Gaussian samples to use
            
        Returns
        -------
        float
            The stability score
        """
        # Predict probability for the given instance
        score_x_raw = self.task.model.predict_proba(pd.DataFrame(xp).T)
        
        # Handle different prediction output formats
        if isinstance(score_x_raw, pd.DataFrame):
            # Handle DataFrame output
            if score_x_raw.shape[1] > 1:
                score_x = score_x_raw.iloc[0, 1]  # Get probability for class 1
            else:
                score_x = score_x_raw.iloc[0, 0]  # Single column case
        elif isinstance(score_x_raw, np.ndarray):
            if score_x_raw.ndim == 2:
                score_x = score_x_raw[0, 1]  # Get probability for class 1
            elif score_x_raw.ndim == 1:
                score_x = score_x_raw[1] if len(score_x_raw) > 1 else score_x_raw[0]
            else:
                score_x = score_x_raw
        else:
            # Handle case where predict_proba returns a scalar
            score_x = score_x_raw
        
        # Convert to float if necessary
        if not isinstance(score_x, (int, float)):
            try:
                score_x = float(score_x)
            except:
                # Last resort: try to extract from object
                score_x = float(np.array(score_x).flat[0])
        
        # Prepare a DataFrame with the predicted score
        score_x_df = pd.DataFrame([score_x] * k)
        score_x_df.reset_index(drop=True, inplace=True)

        # Generate Gaussian samples based on the input instance
        gaussian_samples = np.random.normal(xp.values, 0.1, (k, len(xp)))

        # Get model scores for the Gaussian samples
        model_scores_raw = self.task.model.predict_proba(gaussian_samples)
        
        # Extract probability for class 1 - handle different formats
        if isinstance(model_scores_raw, pd.DataFrame):
            # Handle DataFrame output
            if model_scores_raw.shape[1] > 1:
                model_scores_values = model_scores_raw.iloc[:, 1].values
            else:
                model_scores_values = model_scores_raw.iloc[:, 0].values
        elif isinstance(model_scores_raw, np.ndarray):
            if model_scores_raw.ndim == 2:
                # Shape: (k, n_classes)
                model_scores_values = model_scores_raw[:, 1]
            elif model_scores_raw.ndim == 1:
                # Shape: (k,) - already probabilities for class 1
                model_scores_values = model_scores_raw
            else:
                raise ValueError(f"Unexpected prediction shape: {model_scores_raw.shape}")
        else:
            # Handle case where predict_proba returns a list/tuple
            try:
                model_scores_values = np.array([score[1] for score in model_scores_raw])
            except (TypeError, IndexError):
                # If indexing fails, assume it's already the probability for class 1
                model_scores_values = np.array(model_scores_raw)
        
        # Create DataFrame
        model_scores = pd.DataFrame(model_scores_values)
        model_scores.columns = range(model_scores.shape[1])
        
        # Calculate the stability score using tensor operations
        diff = model_scores.values - score_x_df.values
        stability_score = np.mean(model_scores.values - np.abs(diff))  # Simplified calculation
        
        return float(stability_score)

    def _generation_method(self, instance,
                           robustness_check: ModelChangesRobustnessEvaluator.__class__ = DeltaRobustnessEvaluator,
                           column_name="target",
                           neg_value=0, trex_K=400,
                           trex_threshold=0.4, trex_k=200, **kwargs):
        """
        Generates a counterfactual explanation using the T-Rex method with KDTree optimization.

        @param instance: The instance for which to generate a counterfactual. Can be a DataFrame or Series.
        @param robustness_check: The robustness evaluator to check model changes with respect to input perturbations.
        @param column_name: The name of the target column.
        @param neg_value: The value considered negative in the target variable.
        @param K: The number of nearest neighbor counterfactuals to evaluate.
        @param threshold: The threshold for counterfactual stability.
        @param kwargs: Additional keyword arguments.
        @return: A DataFrame containing the counterfactual explanation if found, otherwise the original instance.
        """
        
        #print(trex_threshold)
        threshold = trex_threshold
        k = trex_k
        K = trex_K
        
        # Build KDTree if not already built
        # print(self.positives_data.shape)
        if self.kdtree is None or self.positives_data is None:
            self._build_kdtree(column_name)
        
        # Compute robustness scores on first request or if k value changed
        if (self.robustness_scores_cache is None or 
            not hasattr(self, '_cached_k') or self._cached_k != k):
            print(f"Computing robustness scores for first time with k={k}...")
            self._cached_k = k
            self._precompute_robustness_scores(k=k)
        
        # Get instance values for KDTree search
        if isinstance(instance, pd.DataFrame):
            instance_values = instance.drop(columns=[column_name]).values.flatten()
        else:
            instance_values = instance.values.flatten()
        
        # Find K nearest neighbors using KDTree
        K = min(K, len(self.positives_data))  # Ensure K does not exceed available positives
        distances, indices = self.kdtree.query(instance_values.reshape(1, -1), k=K)
        
        # Extract the K nearest positive instances
        nearest_positives = self.positives_data.iloc[indices[0]]
    
        # track best stability
        best_score = -float("inf")
        best_positive = None

        # iterate candidates using cached robustness scores
        for idx, (_, positive) in enumerate(nearest_positives.iterrows()):
            # Get the original index in positives_data
            original_idx = indices[0][idx]
            
            # Use cached robustness score instead of computing on-the-fly
            if self.robustness_scores_cache is not None and original_idx in self.robustness_scores_cache:
                stability_score = self.robustness_scores_cache[original_idx]
            else:
                # Fallback to computing on-the-fly if cache is not available
                stability_score = self._compute_stability_score(positive, k=k)

            # immediate return if above threshold
            if stability_score > threshold:
                return pd.DataFrame(positive).T
        
        return pd.DataFrame(instance).T

    def counterfactual_stability(self, xp, k=200):
        """
        Evaluates the stability of a given counterfactual instance.
        This method now uses the cached computation for better performance.

        @param xp: The instance for which to evaluate counterfactual stability.
        @param k: Number of Gaussian samples to use for stability computation.
        @return: A tensor representing the stability score of the counterfactual.
        """
        # Use the new _compute_stability_score method
        stability_score = self._compute_stability_score(xp, k=k)
        
        # Return as tensor for backward compatibility
        return torch.tensor(
            stability_score,
            requires_grad=True,
            dtype=torch.float32
        )