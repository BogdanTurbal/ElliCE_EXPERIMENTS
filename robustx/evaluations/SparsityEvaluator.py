import numpy as np
import pandas as pd
from robustx.evaluations.CEEvaluator import CEEvaluator

class SparsityEvaluator(CEEvaluator):
    """
    An Evaluator class which calculates the sparsity of counterfactuals (percentage of features changed)

    Attributes / Properties
    -------
    task: Task
        Stores the Task for which we are evaluating the sparsity of CEs

    Methods
    -------
    evaluate() -> float:
        Returns the average sparsity (percentage of features changed) of counterfactuals
    """

    def evaluate(self, counterfactuals, valid_val=1, column_name="target", subset=None, threshold=1e-6, **kwargs):
        """
        Determines the average sparsity (percentage of features changed) of the CEs from their original instances
        @param counterfactuals: pd.DataFrame, dataset containing CEs in same order as negative instances in dataset
        @param valid_val: int, what the target value of a valid counterfactual is defined as, default 1
        @param column_name: name of target column
        @param subset: optional DataFrame, contains instances to generate CEs on
        @param threshold: float, threshold below which a feature change is considered insignificant, default 1e-6
        @param kwargs: other arguments
        @return: float, average sparsity (percentage of features changed) of CEs
        """
        if 'predicted' in counterfactuals.columns and 'Loss' in counterfactuals.columns:
            counterfactuals = counterfactuals.drop(columns=['predicted', 'Loss']).astype(np.float32)
            
        orig_preds = (self.task.model.predict(counterfactuals).iloc[:, 0].to_numpy() >= 0.5).astype(int)
        # Mask out any CEs that weren't even valid on the original model
        mask_valid = (orig_preds == valid_val)
        if not mask_valid.any():
            return 0

        counterfactuals = counterfactuals[mask_valid]

        df1 = counterfactuals

        if subset is None:
            df2 = self.task.get_negative_instances(neg_value=1-valid_val, column_name=column_name)
        else:
            df2 = subset
            
        # Ensure the DataFrames have the same shape
        assert df1.shape == df2.shape, "DataFrames must have the same shape"

        sparsities = []

        # Iterate over each row in the DataFrames
        for i in range(len(df1)):
            row1 = df1.iloc[i:i + 1]  # Get the i-th row as a DataFrame
            row2 = df2.iloc[i:i + 1]  # Get the i-th row as a DataFrame
            
            # Calculate feature-wise differences
            diff = np.abs(row1.values - row2.values) > threshold
            
            # Calculate sparsity as percentage of features changed
            changed_features = np.sum(diff)
            total_features = diff.size
            sparsity = changed_features / total_features
            sparsities.append(sparsity)

        # Calculate and return the average sparsity
        return 1 - np.mean(sparsities)