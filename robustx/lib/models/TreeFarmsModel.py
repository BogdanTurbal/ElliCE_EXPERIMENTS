import numpy as np
import pandas as pd
import json
import shap
import torch
from sklearn.metrics import accuracy_score, f1_score

def parse_tree(node, node_index=0, node_list=None):
    if node_list is None:
        node_list = []

    current_index = node_index
    node_data = {
        'index': current_index,
        'feature': -2,
        'threshold': -2.0,
        'children_left': -1,
        'children_right': -1,
        'children_default': -1,
        'is_leaf': False,
    }

    if 'prediction' in node:
        # Leaf node
        node_data['is_leaf'] = True
    else:
        # Internal node
        node_data['feature'] = node['feature']
        node_data['threshold'] = 0.5  # or node['threshold'] if available
    node_list.append(node_data)

    if not node_data['is_leaf']:
        # Left child (false branch)
        left_child_index = len(node_list)
        parse_tree(node['false'], node_index=left_child_index, node_list=node_list)
        # Right child (true branch)
        right_child_index = len(node_list)
        parse_tree(node['true'], node_index=right_child_index, node_list=node_list)

        node_data['children_left'] = left_child_index
        node_data['children_right'] = right_child_index
        node_data['children_default'] = right_child_index

    return node_list

def compute_node_samples(X, y, node_list):
    y = np.asarray(y)
    if y.ndim > 1:
        y = y.ravel()
    y = y.astype(int)

    n_nodes = len(node_list)
    n_classes = len(np.unique(y))
    node_sample_weight = np.zeros(n_nodes, dtype=np.float64)
    node_values = np.zeros((n_nodes, n_classes), dtype=np.float64)
    sample_indices = {}

    # Initialize all samples starting at the root node
    sample_indices[0] = np.arange(len(X))

    for node in node_list:
        node_index = node['index']
        indices = sample_indices.get(node_index, [])
        node_sample_weight[node_index] = len(indices)

        # Count samples per class
        if len(indices) > 0:
            class_counts = np.bincount(y[indices], minlength=n_classes)
            node_values[node_index, :] = class_counts

        if not node['is_leaf']:
            feature = node['feature']
            threshold = node['threshold']

            if feature >= 0:  # Valid feature check
                left_indices = indices[X[indices, feature] <= threshold]
                right_indices = indices[X[indices, feature] > threshold]

                sample_indices[node['children_left']] = left_indices
                sample_indices[node['children_right']] = right_indices

    return node_sample_weight, node_values

def create_custom_rash_tree(rash_raw_model, X, y):
    custom_tree = json.loads(rash_raw_model.__repr__())
    node_list = parse_tree(custom_tree)

    n_classes = len(np.unique(y))
    n_nodes = len(node_list)
    children_left = np.full(n_nodes, -1, dtype=np.int32)
    children_right = np.full(n_nodes, -1, dtype=np.int32)
    children_default = np.full(n_nodes, -1, dtype=np.int32)
    features = np.full(n_nodes, -2, dtype=np.int32)
    thresholds = np.full(n_nodes, -2.0, dtype=np.float64)

    # Compute node sample weight and values based on X and y
    node_sample_weight, node_values = compute_node_samples(X, y, node_list)

    for node in node_list:
        idx = node['index']
        children_left[idx] = node['children_left']
        children_right[idx] = node['children_right']
        children_default[idx] = node['children_default']
        features[idx] = node['feature']
        thresholds[idx] = node['threshold']

    tree_dict = {
        "children_left": children_left,
        "children_right": children_right,
        "children_default": children_default,
        "features": features,
        "thresholds": thresholds,
        "values": (node_values[:, 1] / np.sum(node_values, axis=1))[..., np.newaxis],  # shape (n_nodes, 1, n_classes)
        "node_sample_weight": node_sample_weight,
    }
    #print(tree_dict['values'].shape)
    #print(tree_dict)
    model = {"trees": [tree_dict]}
    return model

class TreeFarmsModel:
    def __init__(self, rash_model, X, y, preprocessor = None):
        self.rash_model = rash_model
        model = create_custom_rash_tree(rash_model, np.array(X), np.array(y))
        explainer = shap.TreeExplainer(model, np.array(X))
        self.model = explainer.model
        self.preprocessor = preprocessor

    def train(self, **kwargs):
        raise NotImplementedError("Training is not supported for TreeFarmsModel.")

    def predict_proba(self, input_data, inverse=True):
        if isinstance(input_data, pd.DataFrame):
            pass
        else:
            input_data = pd.DataFrame(input_data, columns=self.preprocessor.feature_names_in_)

        if self.preprocessor is not None and inverse:
            input_data_orig = self.preprocessor.transform(input_data) # into integer data into binary data
        else:
            input_data_orig = input_data

        if not isinstance(input_data_orig, np.ndarray):
            input_data_orig = input_data_orig.to_numpy()

        res = self.model.predict(input_data_orig)
        probs = np.zeros((len(res), 2))
        probs[:, 1] = res
        probs[:, 0] = 1 - res

        return probs

    def predict(self, input_data, inverse=True):
        proba = self.predict_proba(input_data, inverse=inverse)[:, 1]
        return (proba >= 0.5).astype(int)
    
    def predict_single(self, input_data, inverse=True):
        proba = self.predict_proba(input_data, inverse=inverse)[:, 1]
        return int(proba >= 0.5)
    
    def predict_proba_tensor(self, input_data, inverse=True):
        return torch.tensor(self.predict_proba(input_data, inverse=inverse), dtype=torch.float32)
    
    def evaluate(self, X, y, inverse=True):
        predictions = self.predict(X, inverse=inverse)
        accuracy = accuracy_score(y, predictions)
        f1 = f1_score(y, predictions, average='weighted')

        return {
            "accuracy": accuracy,
            "f1_score": f1
        }    
