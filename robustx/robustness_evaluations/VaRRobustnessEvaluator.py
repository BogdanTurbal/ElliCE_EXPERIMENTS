import pandas as pd
from robustx.robustness_evaluations.ModelChangesRobustnessEvaluator import ModelChangesRobustnessEvaluator
from robustx.lib.tasks.Task import Task

from robustx.evaluations.CEEvaluator import CEEvaluator
from robustx.robustness_evaluations.DeltaRobustnessEvaluator import DeltaRobustnessEvaluator
from robustx.robustness_evaluations.ModelChangesRobustnessEvaluator import ModelChangesRobustnessEvaluator

import numpy as np

class VaRRobustnessEvaluator(ModelChangesRobustnessEvaluator):
    """
    A simple and common robustness evaluation method for evaluating validity of the CE after retraining.
    Used for robustness against model changes.

    Attributes:
        task (Task): The task to solve, inherited from ModelChangesRobustnessEvaluator.
        models (List[BaseModel]): The list of models retrained on the same dataset.
    """

    def __init__(self, ct: Task, models, rfa=False):
        """
        Initializes the VaRRobustnessEvaluator with a given task and trained models.

        @param ct: The task for which robustness evaluations are being made.
                   Provided as a Task instance.
        @param models: The list of models retrained on the same dataset.
        """
        super().__init__(ct)
        self.models = models
        self.rfa = rfa

    def evaluate(self, instance, desired_outcome=1):
        """
        Evaluates whether the instance (the ce) is predicted with the desired outcome by all retrained models.
        The instance is robust if this is true.

        @param instance: The instance (in most cases a ce) to evaluate.
        @param desired_outcome: The value considered positive in the target variable.
        @return: A boolean indicating robust or not.
        """
        instance = pd.DataFrame(instance.values.reshape(1, -1))
        pred_on_orig_model = self.task.model.predict_single(instance)
        # print(pred_on_orig_model)

        # ensure basic validity
        if pred_on_orig_model != desired_outcome:
            print("Original model prediction does not match desired outcome.")
            return False
        # check predictions by new models
        res = 0
        for m in self.models:
            if m.predict_single(instance) == desired_outcome:
                res += 1
        #res /= len(self.models)
        if self.rfa:
            res = int(res == len(self.models))
        else:
            res = res / len(self.models)
        return res#int(res == len(self.models))#res


class VaRRobustnessEvaluatorGlobal(CEEvaluator):
    """
     An Evaluator class which evaluates the proportion of counterfactuals which are robust

        ...

    Attributes / Properties
    -------

    task: Task
        Stores the Task for which we are evaluating the robustness of CEs

    robustness_evaluator: ModelChangesRobustnessEvaluator
        An instance of ModelChangesRobustnessEvaluator to evaluate the robustness of the CEs

    valid_val: int
        Stores what the target value of a valid counterfactual is defined as

    target_col: str
        Stores what the target column name is

    -------

    Methods
    -------

    evaluate() -> int:
        Returns the proportion of CEs which are robust for the given parameters

    -------
    """
    def evaluate(self,
                 counterfactuals: pd.DataFrame,
                 valid_val: int = 1,
                 column_name: str = "target",
                 **kwargs) -> float:
        # 1) Extract only the feature columns
        X_cf = counterfactuals.drop(columns=[column_name, "loss"], errors="ignore")
        if X_cf.shape[0] == 0:
            return 0.0

        # 2) Original model predictions (batch)
        orig_preds = (self.task.model.predict(X_cf).iloc[:, 0].to_numpy() >= 0.5).astype(int)
        # Mask out any CEs that weren't even valid on the original model
        mask_valid = (orig_preds == valid_val)
        if not mask_valid.any():
            return 0.0

        X_valid = X_cf[mask_valid]

        # 3) Get each retrained model’s batch predictions
        #    results in an array of shape (n_valid, n_models)
        pred_matrix = (np.stack([
            m.predict(X_valid)#.iloc[:, 0].to_numpy()
            for m in self.models
        ], axis=1) >= 0.5).astype(int)

        # 4) Build a boolean “correct?” matrix
        correct = (pred_matrix == valid_val)

        # 5) For RFA (require all models), else fraction
        # print("correct")
        # print(pred_matrix.shape)
        # print(valid_val)
        # print(correct.shape)
        # print(correct)
        # print(correct.all(axis=1))
        # print(correct.all(axis=1).shape)
        if self.rfa:
            robust_per_instance = correct.all(axis=1).astype(float)
        else:
            robust_per_instance = correct.mean(axis=1)

        # 6) Return the average robustness over all valid CEs
        return float(robust_per_instance.mean())

    # def evaluate(self, counterfactuals, valid_val=1, column_name="target",
    #              robustness_evaluator: ModelChangesRobustnessEvaluator.__class__ = VaRRobustnessEvaluator, 
    #              **kwargs):
    #     """
    #     Evaluate the proportion of CEs which are robust for the given parameters
    #     @param counterfactuals: pd.DataFrame, the CEs to evaluate
    #     @param delta: int, delta needed for robustness evaluator
    #     @param bias_delta: int, bias delta needed for robustness evaluator
    #     @param M: int, large M needed for robustness evaluator
    #     @param epsilon: int, small epsilon needed for robustness evaluator
    #     @param column_name: str, what the target column name is
    #     @param valid_val: int, what the target value of a valid counterfactual is defined as
    #     @param robustness_evaluator: ModelChangesRobustnessEvaluator.__class__, the CLASS of the evaluator to use
    #     @return: Proportion of CEs which are robust
    #     """
    #     robust = 0
    #     cnt = 0

    #     # Get only the feature variables from the CEs
    #     instances = counterfactuals.drop(columns=[column_name, "loss"], errors='ignore')

    #     robustness_evaluator = robustness_evaluator(self.task, self.models, self.rfa)

    #     for _, instance in instances.iterrows():

    #         # Increment robust if CE is robust under given parameters
    #         if instance is not None:
    #             robust += robustness_evaluator.evaluate(instance, desired_outcome=valid_val)

    #         # Increment total number of CEs encountered
    #         cnt += 1

    #     return robust / cnt



