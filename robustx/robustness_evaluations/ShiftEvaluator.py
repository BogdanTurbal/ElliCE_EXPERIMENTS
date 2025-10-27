import numpy as np
from robustx.lib.tasks.Task import Task
from robustx.generators.CEGenerator import CEGenerator
from robustx.lib.tasks.Task import Task

from robustx.robustness_evaluations.InputChangesRobustnessEvaluator import InputChangesRobustnessEvaluator

def scale_shift(x, factor=1.2, feature_indices=None):
    """
    Feature-wise scaling shift: multiplies all or selected dimensions of x by a constant factor.

    Args:
        x (np.ndarray): Original input instance.
        factor (float): Scaling constant >1 amplifies, <1 attenuates.
        feature_indices (List[int], optional): Indices of dimensions to scale. If None, scales all.

    Returns:
        np.ndarray: Shifted input.
    """
    shifted = x.copy()
    if feature_indices is None:
        shifted = shifted * factor
    else:
        shifted[feature_indices] = shifted[feature_indices] * factor
    return shifted


def reweight_shift(x, weight_vector):
    """
    Marginal reweighting shift: element-wise multiplies x by a weight vector.

    Args:
        x (np.ndarray): Original input instance.
        weight_vector (np.ndarray): Same shape as x; up/down-weight features.

    Returns:
        np.ndarray: Shifted input.
    """
    return x * weight_vector

class ShiftEvaluator(InputChangesRobustnessEvaluator):
    """
    Evaluates whether a given CF remains valid when the CF itself is shifted
    by a user-provided function.
    """

    def __init__(self, ct: Task, shift_fn = scale_shift):
        super().__init__(ct)
        self.shift_fn = shift_fn

    def evaluate(self, instance: np.ndarray, counterfactual: np.ndarray,
                 generator: CEGenerator, neg_value=0, column_name=None, 
                 models=None, 
                 awp_models=None, 
                 lit_models=None, 
                 delta_models=None,
                 delta=None,#proplace_delta,
                 bias_delta=None,
                 r_model=None,
                 d_model=None, level=None) -> bool:
        """
        Apply shift_fn to the CF and test if the perturbed CF still flips the model's prediction.
        """
        # Original target class of CF
        y_target = 1 - generator.model.predict(instance.reshape(1, -1))[0]
        # Perturb the CF
        cf_shifted = self.shift_fn(counterfactual)
        # Check validity after shift
        pred = generator.model.predict(cf_shifted.reshape(1, -1))[0]
        return bool(pred == y_target)