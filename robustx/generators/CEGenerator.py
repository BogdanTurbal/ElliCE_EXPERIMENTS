from abc import ABC, abstractmethod
import pandas as pd
from robustx.lib.tasks.Task import Task

from tqdm import tqdm

from tqdm.auto import tqdm


class CEGenerator(ABC):
    """
    Abstract class for generating counterfactual explanations for a given task.

    This class provides a framework for generating counterfactuals based on a distance function
    and a given task. It supports default distance functions such as Euclidean and Manhattan,
    and allows for custom distance functions.

    Attributes:
        _task (Task): The task to solve.
        __customFunc (callable, optional): A custom distance function.
    """

    def __init__(self, ct: Task, custom_distance_func=None, *args, **kwargs):
        """
        Initializes the CEGenerator with a task and an optional custom distance function.

        @param ct: The Task instance to solve.
        @param custom_distance_func: An optional custom distance function.
        """
        self._task = ct
        self.__customFunc = custom_distance_func

    @property
    def task(self):
        return self._task

    def generate(
        self,
        instances: pd.DataFrame,
        *,
        neg_value=0,
        column_name: str = "target",
        n_jobs=None,
        eps: float = None,
        method_name: str = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Sequential generator - parallelization handled at higher level.

        Parameters
        ----------
        instances : pd.DataFrame
            Rows you want counterfactuals for (target column may or may not be
            present – we ignore/drop it).
        neg_value : int, default 0
            What counts as a negative label (passed through).
        column_name : str, default "target"
            Name of the target column (passed through).
        n_jobs : int, optional
            Ignored - parallelization handled at higher level.
        eps : float, optional
            The epsilon value for counterfactual generation.
        method_name : str, optional
            Name of the method being used (for method-specific parameter setting).
        kwargs :
            Any extra arguments forwarded to `generate_for_instance`.

        Returns
        -------
        pd.DataFrame
            Counterfactuals, with the same index order as `instances`.
        """
        # Set task parameters
        if eps is not None:
            self.task.eps = eps
        if method_name == "RNCE":
            self.task.delta = eps
        
        # Apply any additional parameters from kwargs to the task
        method_params = kwargs.pop('task_params', {})
        for param_name, param_value in method_params.items():
            setattr(self.task, param_name, param_value)
        
        # Sequential processing only - parallelization handled at higher level
        dfs = [
            self.generate_for_instance(row, neg_value=neg_value,
                                    column_name=column_name, **kwargs)
            for _, row in tqdm(instances.iterrows(), total=len(instances),
                            desc="CF-sequential", disable=False, file=None)
        ]
        return pd.concat(dfs) if dfs else pd.DataFrame()

    def _get_task_init_args(self):
        """
        Extract initialization arguments from the task.
        Override this method in subclasses if needed.
        """
        # This is a placeholder - you may need to customize this based on your Task class
        # For example, if Task takes a model and data:
        # return (self.task.model, self.task.data)
        # For now, we'll try to get common attributes
        try:
            # Common pattern: if Task has model and training_data attributes
            if hasattr(self.task, 'model') and hasattr(self.task, 'training_data'):
                return (self.task.model, self.task.training_data)
            elif hasattr(self.task, 'model'):
                return (self.task.model,)
            else:
                # Fallback: return empty args and let subclass handle it
                return ()
        except:
            return ()

    def generate_for_instance(self, instance, neg_value=0,
                              column_name="target", **kwargs) -> pd.DataFrame:
        """
        Generates a counterfactual for a provided instance.

        @param instance: The instance for which you would like to generate a counterfactual.
        @param column_name: The name of the target column.
        @param neg_value: The value considered negative in the target variable.
        @return: A DataFrame containing the counterfactual explanations for the instance.
        """
        return self._generation_method(instance, neg_value=neg_value, column_name=column_name, **kwargs)

    def generate_for_all(self, neg_value=0, column_name="target", **kwargs) -> pd.DataFrame:
        """
        Generates counterfactuals for all instances with a given negative value in their target column.

        @param neg_value: The value in the target column which counts as a negative instance.
        @param column_name: The name of the target variable.
        @return: A DataFrame of the counterfactuals for all negative values.
        """
        negatives = self.task.get_negative_instances(neg_value, column_name=column_name)
        
        # Only pass eps and method_name if they're not already in kwargs
        if 'eps' not in kwargs:
            kwargs['eps'] = getattr(self.task, 'eps', None)
        if 'method_name' not in kwargs:
            kwargs['method_name'] = self.__class__.__name__
        
        counterfactuals = self.generate(
            negatives,
            column_name=column_name,
            neg_value=neg_value,
            **kwargs
        )

        counterfactuals.index = negatives.index
        return counterfactuals

    @abstractmethod
    def _generation_method(self, instance,
                           column_name="target", neg_value=0, **kwargs):
        """
        Abstract method to be implemented by subclasses for generating counterfactuals.

        @param instance: The instance for which to generate a counterfactual.
        @param column_name: The name of the target column.
        @param neg_value: The value considered negative in the target variable.
        @return: A DataFrame containing the generated counterfactuals.
        """
        pass

    @property
    def custom_distance_func(self):
        """
        Returns custom distance function passed at instantiation
        @return: distance Function, (DataFrame, DataFrame) -> Int
        """
        return self.__customFunc