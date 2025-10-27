import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

from robustx.generators.CEGenerator import CEGenerator
from robustx.robustness_evaluations.DeltaRobustnessEvaluator import DeltaRobustnessEvaluator
from robustx.lib.tasks.Task import Task
from functools import lru_cache

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor, as_completed

from multiprocessing import get_context

import torch

import os
import pickle

import copy

import torch
import pandas as pd
from torch import nn
from robustx.datasets.custom_datasets.CsvDatasetLoader import CsvDatasetLoader
from robustx.lib.tasks.ClassificationTask import ClassificationTask

from tqdm import tqdm


# ---------- top-level worker (must be picklable) -------------------
def _rnce_chunk_worker(
        idx_chunk: list[int],
        df_bytes: bytes,
        task_bytes: bytes,
        robust_flag: bool,
        delta: float,
        bias_delta: float,
        column_name: str,
        neg_value: int,
):
    """Process a list of row indices and return accepted rows as ndarray."""
    # delayed imports inside worker
    import pandas as _pd
    from robustx.robustness_evaluations.DeltaRobustnessEvaluator import \
        DeltaRobustnessEvaluator

    # reconstruct heavy objects once
    df_x  = pickle.loads(df_bytes)          # features only
    task  = pickle.loads(task_bytes)        # full Task
    evaluator = DeltaRobustnessEvaluator(task) if robust_flag else None

    accepted = []
    for idx in idx_chunk:
        x_row = df_x.iloc[idx]

        # 1) optional model-prediction filter
        if not robust_flag and not task.model.predict_single(x_row):
            continue

        # 2) robustness filter
        if robust_flag:
            ok = evaluator.evaluate(
                x_row,
                delta=delta,
                bias_delta=bias_delta,
                desired_output=1 - neg_value,
            )
            if not ok:
                continue

        accepted.append(x_row.to_numpy())

    return accepted

def make_embedding_level_task(base_model,
                              original_task,
                              csv_path="embeddings.csv"):
    """
    1) Runs the original task’s train‐set through base_model up to (but excluding)
       its final linear layer, saving penultimate‐layer embeddings + labels to CSV.
    2) Builds a new 1‐layer SimpleNNModel, copies over the old final‐layer weights.
    3) Returns a fresh ClassificationTask on that embeddings CSV.
    """
    device = next(base_model.get_torch_model().parameters()).device
    seq = base_model.get_torch_model()

    # Drop last Linear (+Sigmoid if present)
    n_drop = 2 if isinstance(seq[-1], nn.Sigmoid) else 1
    penult = nn.Sequential(*list(seq.children())[:-n_drop]).to(device)
    penult.eval()

    # Pull off the raw DataFrame and hard-code "target"
    df_train = original_task.training_data.data
    tc = "target"
    X = df_train.drop(columns=[tc]).values
    y = df_train[tc].values

    # Compute embeddings
    with torch.no_grad():
        emb = penult(torch.from_numpy(X).float().to(device)).cpu().numpy()
        
    # Save to CSV
    emb_df = pd.DataFrame(emb, columns=[f"emb_{i}" for i in range(emb.shape[1])])
    emb_df[tc] = y
    emb_df.to_csv(csv_path, index=False)

    # Build new loader & linear model
    emb_loader = CsvDatasetLoader(csv=csv_path, target_column=tc)
    from robustx.lib.models.pytorch_models.SimpleNNModel import SimpleNNModel
    new_linear = SimpleNNModel(input_dim=emb.shape[1],
                               hidden_dim=[],
                               output_dim=1,
                               seed=42)

    # Copy over old final‐layer weights & bias
    old_lin = base_model.output_layer
    new_lin = new_linear.output_layer
    new_lin.weight.data.copy_(old_lin.weight.data)
    new_lin.bias.data  .copy_(old_lin.bias.data)

    return ClassificationTask(new_linear, emb_loader), penult, emb.shape[1]


class RNCE(CEGenerator):
    """
    A counterfactual explanation generator that finds robust nearest counterfactual examples using KDTree.

    Inherits from the CEGenerator class and implements the _generation_method to find counterfactual examples 
    that are robust to perturbations. It leverages KDTree for nearest neighbor search and uses a robustness evaluator 
    to identify robust instances in the training data.

    Attributes:
        intabs (DeltaRobustnessEvaluator): An evaluator for checking the robustness of instances to perturbations.
    """

    def __init__(self, task: Task, **kwargs):
        """
        Initializes the RNCE CE generator with a given task and robustness evaluator.

        @param task: The task to solve, provided as a Task instance.
        """
        super().__init__(task)
        self.intabs = DeltaRobustnessEvaluator(task)

    def _generation_method(self, x, robustInit=True, optimal=True, column_name="target", neg_value=0, delta=0.005,
                        bias_delta=0.005, k=1, **kwargs):
        """
        Generates counterfactual explanations using nearest neighbor search.

        @param x: The instance for which to generate a counterfactual. Can be a DataFrame or Series.
        @param robustInit: If True, only robust instances are considered for counterfactual generation.
        @param column_name: The name of the target column.
        @param neg_value: The value considered negative in the target variable.
        @param delta: The tolerance for robustness in the feature space.
        @param bias_delta: The bias tolerance for robustness in the feature space.
        @param k: The number of counterfactuals to return
        @param kwargs: Additional keyword arguments.
        @return: A DataFrame containing the counterfactual explanation.
        """
        #k = 1
        S = self.getCandidates(robustInit, delta, bias_delta, column_name=column_name, neg_value=neg_value)
    
        # If no candidates found or fewer than requested k, return appropriate format based on k
        if S.empty or S.size < k:
            #print("No instance in the dataset is robust for the given perturbations!")
            x_df = pd.DataFrame(x).T
            if k > 1:
                # Return k copies of the original instance
                return pd.concat([x_df] * k, ignore_index=True)
            else:
                return x_df

        treer = KDTree(S, leaf_size=40)
        
        x_df = pd.DataFrame(x).T
        # print(S.size, x_df.size, k)
        try:
            idxs = np.array(treer.query(x_df, k=k)[1]).flatten()
        except ValueError as e:
            # Handle ValueError with the appropriate format based on k
            if k > 1:
                return pd.concat([x_df] * k, ignore_index=True)
            else:
                return x_df
        
        if k > 1:
            res = pd.DataFrame(S.iloc[idxs])
        else:
            res = pd.DataFrame(S.iloc[idxs[0]]).T
        return res
    
    @lru_cache()
    def getCandidates(
        self,
        robustInit: bool,
        delta: float,
        bias_delta: float,
        column_name: str = "target",
        neg_value: int = 0,
        *,
        n_jobs = None,
        chunk_size: int = 200,        # tune: rows handled per worker call
    ) -> pd.DataFrame:
        """
        Parallel scan with process isolation (each worker has its own
        DeltaRobustnessEvaluator / Gurobi model).
        If n_jobs=1, uses sequential processing instead of ProcessPoolExecutor.
        """
        # if n_jobs is None:
        #     n_jobs = os.cpu_count()
        n_jobs = 1
        
        df_full = self.task.data_support.data
        print(f"[RNCE] data support size: {len(df_full)}")
        df_x    = df_full.drop(columns=[column_name])     # features only

        print(f"[RNCE] rows = {len(df_full)} • processes = {n_jobs}")

        # Sequential processing when n_jobs=1
        #if n_jobs == 1:
        print("[RNCE] Using sequential processing (n_jobs=1)")
        accepted_rows = []
        
        for idx, row in tqdm(df_x.iterrows(), total=len(df_x), desc="RNCE"):
            # 1) optional model-prediction filter
            if not robustInit and not self.task.model.predict_single(row):
                continue

            # 2) robustness filter
            if robustInit:
                ok = self.intabs.evaluate(
                    row,
                    delta=delta,
                    bias_delta=bias_delta,
                    desired_output=1 - neg_value,
                )
                if not ok:
                    continue

            accepted_rows.append(row.to_numpy())
        
        if not accepted_rows:
            return pd.DataFrame()     # empty → handled upstream

        accepted_arr  = np.stack(accepted_rows, axis=0)
        feature_cols  = df_x.columns
        return pd.DataFrame(accepted_arr, columns=feature_cols)

        # # Parallel processing for n_jobs > 1
        # # --- pickle blobs once, broadcast to all workers -------------
        # df_bytes   = pickle.dumps(df_x,   protocol=pickle.HIGHEST_PROTOCOL)
        # task_bytes = pickle.dumps(self.task, protocol=pickle.HIGHEST_PROTOCOL)

        # # --- make index chunks ---------------------------------------
        # indices   = list(range(len(df_x)))
        # chunks    = [indices[i : i + chunk_size]
        #              for i in range(0, len(indices), chunk_size)]

        # ctx = mp.get_context("spawn")
        # accepted_rows = []

        # with ProcessPoolExecutor(max_workers=n_jobs,
        #                          mp_context=ctx) as pool:
        #     futs = [
        #         pool.submit(
        #             _rnce_chunk_worker,
        #             chunk,
        #             df_bytes,
        #             task_bytes,
        #             robustInit,
        #             delta,
        #             bias_delta,
        #             column_name,
        #             neg_value,
        #         )
        #         for chunk in chunks
        #     ]

        #     for f in tqdm(as_completed(futs),
        #                   total=len(futs),
        #                   desc="RNCE"):
        #         accepted_rows.extend(f.result())

        # if not accepted_rows:
        #     return pd.DataFrame()     # empty → handled upstream

        # accepted_arr  = np.stack(accepted_rows, axis=0)
        # feature_cols  = df_x.columns
        # return pd.DataFrame(accepted_arr, columns=feature_cols)

class AmbiguousRNCE(CEGenerator):
    """
    A counterfactual explanation generator that selects a candidate based on two conditions:
      1. The main model's prediction is above 0.5 (if neg_value == 0; or below 0.5 if neg_value == 1).
      2. An additional model (r_model) returns a value greater than a specified level.
      
    It retrieves all points from the training data, filters them using both conditions,
    and then selects the candidate closest (by Euclidean distance) to the input instance using KDTree.
    """
    def __init__(self, task, **kwargd):
        """
        Parameters:
            task: A Task instance containing the model and training data.
        """
        super().__init__(task)

    def _generation_method(self, x, column_name="target", neg_value=0, r_model=None, level=0.5, **kwargs):
        """
        Generates a counterfactual explanation by selecting candidates using cached filtering logic 
        and then choosing the nearest candidate via KDTree.
        
        Parameters:
            x (pd.Series or pd.DataFrame): The input instance.
            column_name (str): Name of the target column to drop from candidate instances.
            neg_value (int): The negative class label.
            r_model: A callable that takes a candidate (as a torch tensor) and returns a scalar tensor.
            level (float): The threshold for the additional (ambiguity) condition.
            **kwargs: Additional keyword arguments.
        
        Returns:
            pd.DataFrame: A DataFrame containing the selected counterfactual explanation.
        """
        # Retrieve filtered candidates using the cached method.
        filtered_candidates = self.getFilteredCandidates(column_name, neg_value, r_model, level)
        
        # Ensure the input instance is a pandas Series.
        if isinstance(x, pd.DataFrame):
            x_instance = x.iloc[0]
        elif isinstance(x, pd.Series):
            x_instance = x
        else:
            raise ValueError("x must be a pandas DataFrame or Series.")
        
        # Use KDTree to select the candidate closest to x_instance.
        if not filtered_candidates.empty:
            tree = KDTree(filtered_candidates.values, leaf_size=40)
            dist, ind = tree.query([x_instance.values], k=1)
            idx = filtered_candidates.index[ind[0][0]]
            result = filtered_candidates.loc[[idx]]
        else:
            print("No candidate satisfies both conditions. Returning the input instance as fallback.")
            result = pd.DataFrame(x_instance).T
        
        # Ensure the result has the same column names as the original instance.
        result.columns = x_instance.index
        return result

    @lru_cache(maxsize=None)
    def getFilteredCandidates(self, column_name="target", neg_value=0, r_model=None, level=0.5):
        """
        Retrieves candidate instances from the training data that satisfy both the main condition 
        (based on the model's prediction) and the additional ambiguity condition (via r_model).
        This method is cached to avoid repeated filtering on unchanged parameters.
        
        Parameters:
            column_name (str): The name of the target column to drop.
            neg_value (int): The negative class label.
            r_model: A callable that takes a candidate (as a torch tensor) and returns a scalar tensor.
            level (float): The threshold for the additional (ambiguity) condition.
        
        Returns:
            pd.DataFrame: A DataFrame of candidate instances that satisfy both conditions.
        """
        # Retrieve raw candidate instances from the training data.
        candidates = []
        for _, instance in self.task.training_data.data.iterrows():
            candidate = instance.drop(column_name)
            candidates.append(candidate)
        S = pd.DataFrame(candidates)
        
        # Define helper function for the main model condition.
        def main_condition(candidate):
            candidate_array = candidate.values.reshape(1, -1)
            p = self.task.model.predict_proba(candidate_array)[1]
            if hasattr(p, 'item'):
                p = p.item()  # Convert to a Python scalar if necessary.
            return p >= 0.5 if neg_value == 0 else p < 0.5

        # Define helper function for the additional ambiguity condition.
        def additional_condition(candidate):
            candidate_tensor = torch.tensor(candidate.values, dtype=torch.float32).unsqueeze(0)
            ambiguity = r_model(candidate_tensor)
            return ambiguity.item() > level
        
        # Filter the candidates based on both conditions.
        filtered_candidates = S[S.apply(lambda row: main_condition(row) and additional_condition(row), axis=1)]
        return filtered_candidates
    
    
   

from functools import lru_cache
from sklearn.neighbors import KDTree
import torch
import numpy as np
import pandas as pd
from torch import nn
from robustx.generators.CEGenerator import CEGenerator
from robustx.robustness_evaluations.DeltaRobustnessEvaluator import DeltaRobustnessEvaluator
from robustx.lib.tasks.Task import Task

class RNCEMod(CEGenerator):
    """
    Embedding‐filter + input‐space nearest‐neighbor CE with caching of candidate filtering.
    """

    def __init__(self, task: Task):
        super().__init__(task)
        self.orig_task = task
        self.base_model = task.model

        # extract penultimate network
        seq = self.base_model.get_torch_model()
        n_drop = 2 if isinstance(seq[-1], nn.Sigmoid) else 1
        self.penult = nn.Sequential(*list(seq.children())[:-n_drop])
        self.device = next(self.base_model.get_torch_model().parameters()).device
        self.penult.to(self.device).eval()

        # build a throwaway embedding‐task so DeltaRobustnessEvaluator works
        emb_task, _, _ = make_embedding_level_task(
            self.base_model, self.orig_task,
            csv_path="tmp_embeddings.csv"
        )
        self.intabs = DeltaRobustnessEvaluator(emb_task)

    @lru_cache(maxsize=None)
    def getCandidates(self,
                      robustInit: bool,
                      delta: float,
                      bias_delta: float,
                      column_name: str = "target",
                      neg_value: int = 0) -> pd.DataFrame:
        """
        Return the ORIGINAL feature‐rows that pass embedding‐level δ‐robustness.
        Cached by (robustInit, delta, bias_delta, column_name, neg_value).
        """
        kept_rows = []
        kept_idxs = []
        print(f"Data support size: {self.orig_task.data_support.data.shape[0]}")
        for idx, row in self.orig_task.data_support.data.iterrows():
            x_raw = row.drop(column_name)
            # compute embedding
            arr  = x_raw.values.reshape(1, -1).astype(np.float32)
            tens = torch.from_numpy(arr).to(self.device)
            with torch.no_grad():
                emb = self.penult(tens).cpu().squeeze(0)

            # test robustness at the embedding
            ok = self.intabs.evaluate(emb,
                                      delta=delta,
                                      bias_delta=bias_delta,
                                      desired_output=1 - neg_value)
            if robustInit and ok:
                kept_rows.append(x_raw)
                kept_idxs.append(idx)
            elif not robustInit:
                pred = self.orig_task.model.predict_single(x_raw)
                if pred != neg_value:
                    kept_rows.append(x_raw)
                    kept_idxs.append(idx)

        if not kept_rows:
            cols = self.orig_task.training_data.data.columns.drop(column_name)
            return pd.DataFrame([], columns=cols)

        return pd.DataFrame(kept_rows, index=kept_idxs)

    def _generation_method(self,
                           x,
                           robustInit: bool = True,
                           optimal: bool = True,
                           column_name: str = "target",
                           neg_value: int = 0,
                           delta_last: float = 0.005,
                           delta_last_bias: float = 0.005,
                           k: int = 1,
                           **kwargs) -> pd.DataFrame:
        """
        1) Filter raw training rows by embedding‐robustness (cached).
        2) KDTree‐NN on raw features to x.
        """
        # prepare raw x
        if isinstance(x, pd.DataFrame):
            x_raw = x.iloc[0]
        elif isinstance(x, pd.Series):
            x_raw = x#.drop(labels=column_name)
        else:
            raise ValueError("x must be Series or single‐row DataFrame")

        # get filtered candidates in raw feature space
        S = self.getCandidates(robustInit, delta_last, delta_last_bias,
                               column_name, neg_value)
        if S.empty:
            return x if isinstance(x, pd.DataFrame) else pd.DataFrame([x_raw])

        # KDTree on raw features
        tree = KDTree(S.values, leaf_size=40)
        query = x_raw.values.reshape(1, -1)
        idxs = tree.query(query, k=k)[1].flatten()

        orig_idxs = S.index.to_numpy()[idxs].tolist()
        raw_df = self.orig_task.training_data.data.drop(columns=[column_name])

        if k > 1:
            return raw_df.loc[orig_idxs].reset_index(drop=True)
        else:
            return raw_df.loc[[orig_idxs[0]]].reset_index(drop=True)
