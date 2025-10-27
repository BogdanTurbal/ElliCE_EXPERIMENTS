import numpy as np

from robustx.evaluations.CEEvaluator import CEEvaluator
from robustx.lib.distance_functions.DistanceFunctions import euclidean, manhattan


class DistanceEvaluator(CEEvaluator):
    """
     An Evaluator class which evaluates the average distance of counterfactuals from their original instance

        ...

    Attributes / Properties
    -------

    task: Task
        Stores the Task for which we are evaluating the distance of CEs

    distance_func: Function
        A function which takes in 2 dataframes and returns an integer representing distance, defaulted to euclidean

    valid_val: int
        Stores what the target value of a valid counterfactual is defined as

    -------

    Methods
    -------

    evaluate() -> int:
        Returns the average distance of each x' from x

    -------
    """

    def evaluate(self, counterfactuals, valid_val=1, distance_func=euclidean, column_name="target", subset=None, **kwargs):
        """
        Determines the average distance of the CEs from their original instances
        @param counterfactuals: pd.DataFrame, dataset containing CEs in same order as negative instances in dataset
        @param valid_val: int, what the target value of a valid counterfactual is defined as, default 1
        @param distance_func: Function, function which takes in 2 dataframes and returns an integer representing
                              distance, defaulted to euclidean
        @param column_name: name of target column
        @param subset: optional DataFrame, contains instances to generate CEs on
        @param kwargs: other arguments
        @return: int, average distance of CEs from their original instances
        """
        if 'predicted' in counterfactuals.columns and 'Loss' in counterfactuals.columns:
            counterfactuals = counterfactuals.drop(columns=['predicted', 'Loss']).astype(np.float32)
            
        orig_preds = (self.task.model.predict(counterfactuals).iloc[:, 0].to_numpy() >= 0.5).astype(int)
        # Mask out any CEs that weren't even valid on the original model
        mask_valid = (orig_preds == valid_val)
        if not mask_valid.any():
            return 1000

        counterfactuals = counterfactuals[mask_valid]

        df1 = counterfactuals

        if subset is None:
            df2 = self.task.get_negative_instances(neg_value= 1-valid_val, column_name=column_name)[mask_valid]
        else:
            df2 = subset[mask_valid]
            
        # Ensure the DataFrames have the same shape
        assert df1.shape == df2.shape, "DataFrames must have the same shape"

        distances = []

        # Iterate over each row in the DataFrames
        for i in range(len(df1)):
            row1 = df1.iloc[i:i + 1]  # Get the i-th row as a DataFrame
            row2 = df2.iloc[i:i + 1]  # Get the i-th row as a DataFrame

            # Calculate distance between corresponding rows
            dist = distance_func(row1, row2)
            if dist <= 1e-6:
                continue
            distances.append(dist)

        # Calculate and return the average distance
        return np.mean(distances)


class DistanceEvaluatorM(DistanceEvaluator):
    """
    Wrapper evaluator that enforces the use of Manhattan distance.
    
    This class reuses the logic in DistanceEvaluator and simply overrides the distance function.
    """

    def evaluate(self, counterfactuals, valid_val=1, distance_func=None, column_name="target", subset=None, **kwargs):
        # Override the distance function to always use the Manhattan metric
        return super().evaluate(
            counterfactuals,
            valid_val=valid_val,
            distance_func=manhattan,
            column_name=column_name,
            subset=subset,
            **kwargs
        )
