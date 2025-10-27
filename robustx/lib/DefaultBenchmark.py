import pandas as pd
from robustx.evaluations.DistanceEvaluator import DistanceEvaluator, DistanceEvaluatorM
from robustx.evaluations.ValidityEvaluator import ValidityEvaluator
from robustx.evaluations.ManifoldEvaluator import ManifoldEvaluator
from robustx.generators.CE_methods.BinaryLinearSearch import BinaryLinearSearch
from robustx.generators.CE_methods.GuidedBinaryLinearSearch import GuidedBinaryLinearSearch
from robustx.generators.CE_methods.NNCE import NNCE
from robustx.generators.CE_methods.KDTreeNNCE import KDTreeNNCE
from robustx.generators.CE_methods.MCE import MCE
from robustx.generators.CE_methods.Wachter import Wachter
from robustx.generators.robust_CE_methods.APAS import APAS
from robustx.generators.robust_CE_methods.ArgEnsembling import ArgEnsembling
from robustx.generators.robust_CE_methods.DiverseRobustCE import DiverseRobustCE
from robustx.generators.robust_CE_methods.MCER import MCER
from robustx.generators.robust_CE_methods.ModelMultiplicityMILP import ModelMultiplicityMILP
from robustx.generators.robust_CE_methods.PROPLACE import PROPLACE
from robustx.generators.robust_CE_methods.RNCE import RNCE, AmbiguousRNCE, RNCEMod
from robustx.generators.robust_CE_methods.ROAR import ROAR

from robustx.generators.robust_CE_methods.STCE import TRex#LastLayerEllipsoidCEOH
from robustx.generators.robust_CE_methods.SEV import SEV

from robustx.generators.robust_CE_methods.LastLayerEllipsoidCE import LastLayerEllipsoidCEOHC, LastLayerEllipsoidCEOHCNT, LastLayerEllipsoidCEOHCBall
from robustx.generators.robust_CE_methods.LastLayerEllipsoidCENam import LastLayerEllipsoidCEOHCNam, LastLayerEllipsoidCEOHCNTNam
from robustx.generators.robust_CE_methods.TRexI import TRexI

from robustx.evaluations.SparsityEvaluator import SparsityEvaluator 
#HypercubeCFGenerator, LastLayerHypercubeCEOH

from robustx.lib.tasks.ClassificationTask import ClassificationTask
import time
from tabulate import tabulate

from tqdm import tqdm

# Updated METHODS dictionary now includes a new entry for LIT-based counterfactuals.
# (Assuming you add a corresponding CE generator class for LIT—here named "LITGenerator".)
METHODS = {
    "APAS": APAS, 
    "ArgEnsembling": ArgEnsembling, 
    "DiverseRobustCE": DiverseRobustCE, 
    "MCER": MCER,
    "ModelMultiplicityMILP": ModelMultiplicityMILP, 
    "PROPLACE": PROPLACE, 
    "RNCE": RNCE, 
    "ROAR": ROAR,
    "STCE": TRex, 
    "BinaryLinearSearch": BinaryLinearSearch, 
    "GuidedBinaryLinearSearch": GuidedBinaryLinearSearch,
    "NNCE": NNCE, 
    "KDTreeNNCE": KDTreeNNCE, 
    "MCE": MCE, 
    "Wachter": Wachter, 
    "AmbiguousRNCE": AmbiguousRNCE,
    "RNCEMod": RNCEMod,
    # "LastLayerEllipsoid": LastLayerEllipsoidCE,#lambda task: LastLayerEllipsoidCE(task, eps=0.05)
    "SEV":SEV,
    "LastLayerEllipsoidCEOHC": LastLayerEllipsoidCEOHC,
    "LastLayerEllipsoidCEOHCNT": LastLayerEllipsoidCEOHCNT,
    "TRexI": TRexI,
    "LastLayerEllipsoidCEOHCBall": LastLayerEllipsoidCEOHCBall,
    # NAM-specific methods
    "LastLayerEllipsoidCEOHCNam": LastLayerEllipsoidCEOHCNam,
    "LastLayerEllipsoidCEOHCNTNam": LastLayerEllipsoidCEOHCNTNam,
    #"BootstrappedLastLayerEllipsoidCE": BootstrappedLastLayerEllipsoidCE
    # "HypercubeCFGenerator": HypercubeCFGenerator,
    # "LastLayerHypercubeCEOH": LastLayerHypercubeCEOH
    #"LIT": LITGenerator  # NEW: a counterfactual generator that uses models trained via LIT.
}
from robustx.evaluations.LOFEvaluator import LOFEvaluator
EVALUATIONS = {
    "Distance": DistanceEvaluator, 
    "DistanceM": DistanceEvaluatorM, 
    "Validity": ValidityEvaluator, 
    "Manifold": ManifoldEvaluator,
    #"Delta-robustness": RobustnessProportionEvaluator,
    "Sparsity": SparsityEvaluator,
    "LOF": LOFEvaluator
}

def default_benchmark(ct: ClassificationTask, methods, evaluations,
                      subset: pd.DataFrame = None, **params):
    """
    Generates and prints a table summarizing the performance of different counterfactual
    explanation generation methods.
    
    Args:
      ct: ClassificationTask.
      methods: A list (or set) of method names.
      evaluations: A list (or set) of evaluator names.
      subset: optional DataFrame, subset of instances to generate CEs on.
      **params: Additional parameters passed to the CE generation methods and evaluators.
    
    Returns:
      None
    """
    results = []
    
    # Extract model ensembles - now with support for target_eps suffixes
    # Create dictionaries to map model types to their lists
    retrained_models = {}
    awp_models = {}
    lit_models = {}
    delta_models = {}
    diff_ensamble = {}
    elips_ensamble = {}
    tree_rset = {}
    
    # Parse all model parameters and group by type and target_eps
    for param_name, model_list in params.items():
        if not isinstance(model_list, list):
            continue
            
        # Parse parameter name to extract model type and target_eps
        if param_name.startswith("models_"):
            target_eps = param_name[7:]  # Remove "models_" prefix
            retrained_models[target_eps] = model_list
        elif param_name.startswith("awp_models_"):
            target_eps = param_name[11:]  # Remove "awp_models_" prefix
            awp_models[target_eps] = model_list
        elif param_name.startswith("lit_models_"):
            target_eps = param_name[11:]  # Remove "lit_models_" prefix
            lit_models[target_eps] = model_list
        elif param_name.startswith("delta_models_"):
            target_eps = param_name[13:]  # Remove "delta_models_" prefix
            delta_models[target_eps] = model_list
        elif param_name.startswith("diff_ensamble_"):
            target_eps = param_name[14:]  # Remove "diff_ensamble_" prefix
            diff_ensamble[target_eps] = model_list
        elif param_name.startswith("elips_ensamble_"):
            target_eps = param_name[15:]  # Remove "elips_ensamble_" prefix
            elips_ensamble[target_eps] = model_list
        elif param_name.startswith("tree_rset_"):
            target_eps = param_name[10:]  # Remove "tree_rset_" prefix
            tree_rset[target_eps] = model_list
        # Backward compatibility - handle old style without suffixes
        elif param_name == "models":
            retrained_models["default"] = model_list
        elif param_name == "awp_models":
            awp_models["default"] = model_list
        elif param_name == "lit_models":
            lit_models["default"] = model_list
        elif param_name == "delta_models":
            delta_models["default"] = model_list
        elif param_name == "diff_ensamble":
            diff_ensamble["default"] = model_list
        elif param_name == "elips_ensamble":
            elips_ensamble["default"] = model_list
        elif param_name == "tree_rset":
            tree_rset["default"] = model_list
    
    for method_name in methods:
        #print(f"Running {method_name}...")
        # Instantiate the CE generator method
        method = None
        if isinstance(method_name, str):
            method = METHODS[method_name]
        else:
            method, method_name = method_name
            
        start_time = time.perf_counter()
        
        ce_generator = method(ct)
        
        # Generate counterfactuals ONCE
        if subset is None:
            ces = ce_generator.generate_for_all(**params)
        else:
            ces = ce_generator.generate(subset, **params)
        
        # End timer
        end_time = time.perf_counter()
        
        # Start evaluation
        eval_results = [method_name, end_time - start_time]
        rl_headers = []
        
        # print(f"Evaluating {method_name}...")
        # print(ces)
        
        for eval_name in evaluations:
            ce_evaluator = None
            rfa = False
            if isinstance(eval_name, str):
                evaluator = EVALUATIONS[eval_name]
                rfa = 'RFA' in eval_name
                ce_evaluator = evaluator(ct)
                ce_evaluator.rfa = rfa
                rl_headers.append(eval_name)
            else:
                evaluator, rb_eval_name = eval_name
                rfa = 'RFA' in rb_eval_name
                rl_headers.append(rb_eval_name)
                ce_evaluator = evaluator(ct)
                ce_evaluator.rfa = rfa
                
                # Enhanced model assignment logic with target_eps support
                # Extract target_eps from evaluator name if present
                target_eps = "default"
                if "-" in rb_eval_name:
                    parts = rb_eval_name.split("-")
                    # Look for numeric target_eps in the parts
                    for part in parts:
                        if part.replace(".", "").isdigit():
                            target_eps = part
                            break
                
                # Assign the appropriate model ensemble based on the prefix
                if rb_eval_name.startswith("AWP"):
                    ce_evaluator.models = awp_models.get(target_eps, awp_models.get("default", []))
                elif rb_eval_name.startswith("LIT"):
                    ce_evaluator.models = lit_models.get(target_eps, lit_models.get("default", []))
                elif rb_eval_name.startswith("Ret"):
                    ce_evaluator.models = retrained_models.get(target_eps, retrained_models.get("default", []))
                elif "Delta" in rb_eval_name:
                    ce_evaluator.models = delta_models.get(target_eps, delta_models.get("default", []))
                elif "ROB" in rb_eval_name:
                    ce_evaluator.models = diff_ensamble.get(target_eps, diff_ensamble.get("default", []))
                elif "ELIPSOID" in rb_eval_name:
                    ce_evaluator.models = elips_ensamble.get(target_eps, elips_ensamble.get("default", []))
                elif "TreeFARMS" in rb_eval_name:
                    ce_evaluator.models = tree_rset.get(target_eps, tree_rset.get("default", []))
                    
            eval_results.append(ce_evaluator.evaluate(ces, **params))
        results.append(eval_results)
    
    # Set headers
    headers = ["Method", "Execution Time (s)"]
    for eval_name in rl_headers:
        headers.append(eval_name)
    
    # Print results table
    print(tabulate(results, headers, tablefmt="grid"))
    return results