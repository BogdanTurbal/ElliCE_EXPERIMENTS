import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Union

from sklearn.preprocessing import StandardScaler
from robustx.datasets.provided_datasets.AdultDatasetLoader import AdultDatasetLoader


def load_parkinsons(csv_path: str = "datasets/parkinsons_updrs.csv", target_col: str = "motor_UPDRS") -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    if target_col not in ("motor_UPDRS", "total_UPDRS"):
        raise ValueError("target_col must be 'motor_UPDRS' or 'total_UPDRS'")
    med = df[target_col].median()
    df["__y__"] = (df[target_col] > med).astype(int)
    drop_cols = ["subject#", "test_time", "motor_UPDRS", "total_UPDRS"]
    X_all = df.drop(columns=drop_cols + ["__y__"]).values.astype(np.float32)
    y_all = df["__y__"].values
    classes, counts = np.unique(y_all, return_counts=True)
    min_cnt = int(counts.min())
    X_parts, y_parts = [], []
    for c in classes:
        idx = np.random.choice(np.where(y_all == c)[0], size=min_cnt, replace=False)
        X_parts.append(X_all[idx])
        y_parts.append(y_all[idx])
    X_bal = np.vstack(X_parts)
    y_bal = np.hstack(y_parts)
    perm = np.random.permutation(len(y_bal))
    X = X_bal[perm]
    y = y_bal[perm]
    return X, y


def load_wine_quality(csv_path: str = "datasets/winequality.csv", target_col: str = "quality") -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path, sep=",")
    med = df[target_col].median()
    df["__y__"] = (df[target_col] > med).astype(int)
    X_all = df.drop(columns=[target_col, "__y__", "type"]).values.astype(np.float32)
    y_all = df["__y__"].values
    classes, counts = np.unique(y_all, return_counts=True)
    min_cnt = int(counts.min())
    X_parts, y_parts = [], []
    for c in classes:
        idx = np.random.choice(np.where(y_all == c)[0], size=min_cnt, replace=False)
        X_parts.append(X_all[idx])
        y_parts.append(y_all[idx])
    X_bal = np.vstack(X_parts)
    y_bal = np.concatenate(y_parts)
    perm = np.random.permutation(len(y_bal))
    X = X_bal[perm]
    y = y_bal[perm]
    return X, y


def load_iris(csv_path: str = "datasets/iris.csv", positive_class: str = "setosa") -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    if "Iris-" + positive_class not in df["species"].unique():
        raise ValueError(f"{positive_class} not found in species column")
    df["__y__"] = (df["species"] == "Iris-" + positive_class).astype(int)
    feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    X_all = df[feature_cols].values.astype(np.float32)
    y_all = df["__y__"].values
    classes, counts = np.unique(y_all, return_counts=True)
    min_cnt = int(counts.min())
    X_parts, y_parts = [], []
    for c in classes:
        idx = np.random.choice(np.where(y_all == c)[0], size=min_cnt, replace=False)
        X_parts.append(X_all[idx])
        y_parts.append(y_all[idx])
    X_bal = np.vstack(X_parts)
    y_bal = np.concatenate(y_parts)
    perm = np.random.permutation(len(y_bal))
    X = X_bal[perm]
    y = y_bal[perm]
    return X, y


def load_iris_ext(csv_path: str = "datasets/iris_extended.csv", positive_class: str = "setosa") -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    if positive_class not in df["species"].unique():
        raise ValueError(f"{positive_class} not found in species column")
    df["__y__"] = (df["species"] == positive_class).astype(int)
    feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    X_all = df[feature_cols].values.astype(np.float32)
    y_all = df["__y__"].values
    classes, counts = np.unique(y_all, return_counts=True)
    min_cnt = int(counts.min())
    X_parts, y_parts = [], []
    for c in classes:
        idx = np.random.choice(np.where(y_all == c)[0], size=min_cnt, replace=False)
        X_parts.append(X_all[idx])
        y_parts.append(y_all[idx])
    X_bal = np.vstack(X_parts)
    y_bal = np.concatenate(y_parts)
    perm = np.random.permutation(len(y_bal))
    X = X_bal[perm]
    y = y_bal[perm]
    return X, y


def load_banknote(csv_path: Union[str, Path] = "datasets/banknote_authentication.csv") -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    feature_cols = ["variance", "skewness", "curtosis", "entropy"]
    X_all = df[feature_cols].values.astype(np.float32)
    y_all = df["class"].astype(int).values
    classes, counts = np.unique(y_all, return_counts=True)
    min_cnt = int(counts.min())
    X_parts, y_parts = [], []
    for c in classes:
        idx = np.random.choice(np.where(y_all == c)[0], size=min_cnt, replace=False)
        X_parts.append(X_all[idx])
        y_parts.append(y_all[idx])
    X_bal = np.vstack(X_parts)
    y_bal = np.concatenate(y_parts)
    perm = np.random.permutation(len(y_bal))
    X = X_bal[perm]
    y = y_bal[perm]
    return X, y


from pathlib import Path
from typing import Union
from collections import defaultdict
import re

def load_germanc(
    csv_path: Union[str, Path] = "datasets/germanc.csv",
    random_state: int = 42,
):
    df = (
        pd.read_csv(csv_path)
        .drop(columns=[c for c in ["Unnamed: 0"] if c in pd.read_csv(csv_path).columns])
        .apply(pd.to_numeric, errors="coerce")
        .dropna(axis=0, how="any")
    )

    dummy_groups = defaultdict(list)
    pattern = re.compile(r"(.+)_\w+$")

    for col in df.columns:
        m = pattern.match(col)
        if m:
            dummy_groups[m.group(1)].append(col)

    # cols_to_drop = []
    # for group_cols in dummy_groups.values():
    #     if len(group_cols) > 1 and (df[group_cols].sum(axis=1) == 1).all():
    #         cols_to_drop.append(group_cols[0])  # drop the first dummy

    y_all = df["y"].astype(int)
    X_all = df.drop(columns=["y"])

    rng = np.random.RandomState(random_state)
    min_cnt = y_all.value_counts().min()
    balanced_idx = np.concatenate(
        [rng.choice(np.where(y_all == cls)[0], size=min_cnt, replace=False)
         for cls in y_all.unique()]
    )
    rng.shuffle(balanced_idx)

    return (
        X_all.iloc[balanced_idx].to_numpy(dtype=np.float32),
        y_all.iloc[balanced_idx].to_numpy(dtype=int),
    )
    
def load_amsterdam(csv_path: Union[str, Path] = "datasets/amsterdam.csv", random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path).apply(pd.to_numeric, errors="coerce").dropna()
    y_all, X_all = df["rec4"].astype(int), df.drop(columns=["rec4"]) 
    rng = np.random.RandomState(random_state)
    min_cnt = np.unique(y_all, return_counts=True)[1].min()
    parts = [rng.choice(np.where(y_all == c)[0], size=min_cnt, replace=False) for c in np.unique(y_all)]
    idx = rng.permutation(np.concatenate(parts))
    return X_all.iloc[idx].values.astype(np.float32), y_all.iloc[idx].values.astype(int)


def scale_fold_selective(X_trn: np.ndarray, X_val: np.ndarray, X_tst: np.ndarray, one_hot_threshold: int = 2):
    uniq = np.apply_along_axis(lambda col: len(np.unique(col)), 0, X_trn)
    cont_cols = np.where(uniq > one_hot_threshold)[0]
    if cont_cols.size:
        sc = StandardScaler()
        X_trn[:, cont_cols] = sc.fit_transform(X_trn[:, cont_cols])
        X_val[:, cont_cols] = sc.transform(X_val[:, cont_cols])
        X_tst[:, cont_cols] = sc.transform(X_tst[:, cont_cols])
    return X_trn, X_val, X_tst


def load_polish() -> Tuple[np.ndarray, np.ndarray]:
    nparr = pd.read_csv("datasets/polish-companies_clean_uncut.csv").values
    np.random.shuffle(nparr)
    X_all, y_all = nparr[:, 1:65].astype(np.float32), nparr[:, 65]
    classes, min_cnt = np.unique(y_all, return_counts=True)
    min_cnt = int(min_cnt.min())
    X, y = [], []
    for c in classes:
        idx = np.random.choice(np.where(y_all == c)[0], size=min_cnt, replace=False)
        X.append(X_all[idx]); y.append(y_all[idx])
    X, y = np.vstack(X), np.concatenate(y)
    rng = np.random.permutation(len(y))
    X, y = X[rng], y[rng]
    return X, y


def load_austrc() -> Tuple[np.ndarray, np.ndarray]:
    nparr = pd.read_csv("datasets/austrc.csv").values
    np.random.shuffle(nparr)
    n = nparr.shape[1]
    
    X_all, y_all = nparr[:, 1:n - 1].astype(np.float32), nparr[:, n - 1]

    classes, min_cnt = np.unique(y_all, return_counts=True)
    min_cnt = int(min_cnt.min())
    X, y = [], []
    for c in classes:
        idx = np.random.choice(np.where(y_all == c)[0], size=min_cnt, replace=False)
        X.append(X_all[idx]); y.append(y_all[idx])
    X, y = np.vstack(X), np.concatenate(y)
    rng = np.random.permutation(len(y))
    X, y = X[rng], y[rng]
    return X, y


def load_phonemes() -> Tuple[np.ndarray, np.ndarray]:
    arr = pd.read_csv("datasets/phoneme.csv").values
    np.random.shuffle(arr)
    return arr[:5000, :5].astype(np.float32), arr[:5000, 5]


def load_adult_dataset_balanced(subsample_size: int = 30_000, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    ad = AdultDatasetLoader(); ad.load_data()
    X_all, y_all = ad.get_default_preprocessed_features().values, ad.y.values
    rng = np.random.RandomState(random_state)
    min_cnt = np.unique(y_all, return_counts=True)[1].min()
    idx_bal = np.concatenate([rng.choice(np.where(y_all == c)[0], min_cnt, replace=False) for c in np.unique(y_all)])
    if len(idx_bal) > subsample_size:
        idx_bal = rng.choice(idx_bal, subsample_size, replace=False)
    return X_all[idx_bal], y_all[idx_bal]


def load_compas() -> Tuple[np.ndarray, np.ndarray]:
    nparr = pd.read_csv("datasets/compas.csv").values
    np.random.shuffle(nparr)
    n = nparr.shape[1]
    X_all, y_all = nparr[:, 1:n].astype(np.float32), nparr[:, 0]
    classes, min_cnt = np.unique(y_all, return_counts=True)
    min_cnt = int(min_cnt.min())
    X, y = [], []
    for c in classes:
        idx = np.random.choice(np.where(y_all == c)[0], size=min_cnt, replace=False)
        X.append(X_all[idx]); y.append(y_all[idx])
    X, y = np.vstack(X), np.concatenate(y)
    rng = np.random.permutation(len(y))
    X, y = X[rng], y[rng]
    return X, y


def load_diabetes() -> Tuple[np.ndarray, np.ndarray]:
    nparr = pd.read_csv("datasets/diabetes.csv").values
    np.random.shuffle(nparr)
    n = nparr.shape[1]
    X_all, y_all = nparr[:, :n - 1].astype(np.float32), nparr[:, n - 1]
    classes, min_cnt = np.unique(y_all, return_counts=True)
    min_cnt = int(min_cnt.min())
    X, y = [], []
    for c in classes:
        idx = np.random.choice(np.where(y_all == c)[0], size=min_cnt, replace=False)
        X.append(X_all[idx]); y.append(y_all[idx])
    X, y = np.vstack(X), np.concatenate(y)
    rng = np.random.permutation(len(y))
    X, y = X[rng], y[rng]
    return X, y


def load_diabetes_cnt() -> Tuple[np.ndarray, np.ndarray]:
    nparr = pd.read_csv("datasets/diabetes.csv").values
    np.random.shuffle(nparr)
    n = nparr.shape[1]
    X_all, y_all = nparr[:, :n - 1].astype(np.float32), nparr[:, n - 1]
    classes, min_cnt = np.unique(y_all, return_counts=True)
    min_cnt = int(min_cnt.min())
    X, y = [], []
    for c in classes:
        idx = np.random.choice(np.where(y_all == c)[0], size=min_cnt, replace=False)
        X.append(X_all[idx]); y.append(y_all[idx])
    X, y = np.vstack(X), np.concatenate(y)
    rng = np.random.permutation(len(y))
    X, y = X[rng], y[rng]
    return X, y


# def load_fico() -> Tuple[np.ndarray, np.ndarray]:
#     nparr = pd.read_csv("datasets/fico.csv").values
#     np.random.shuffle(nparr)
#     n = nparr.shape[1]
#     X_all, y_all = nparr[:, :n - 1].astype(np.float32), nparr[:, n - 1]
#     classes, min_cnt = np.unique(y_all, return_counts=True)
#     min_cnt = int(min_cnt.min())
#     X, y = [], []
#     for c in classes:
#         idx = np.random.choice(np.where(y_all == c)[0], size=min_cnt, replace=False)
#         X.append(X_all[idx]); y.append(y_all[idx])
#     X, y = np.vstack(X), np.concatenate(y)
#     rng = np.random.permutation(len(y))
#     X, y = X[rng], y[rng]
#     return X, y

# def load_fico(
#     random_state: int | None = None,
# ):
#     df = pd.read_csv("datasets/fico.csv")

#     sentinel_vals = [-7, -8, -9]
#     df.replace(sentinel_vals, 0, inplace=True)

#     #feat_cols = df.columns[:-1]          # everything except the target
#     #df.dropna(inplace=True) #axis=0, how="any", subset=feat_cols, 

#     X_all = df.iloc[:, :-1].to_numpy(dtype=np.float32)
#     y_all = df.iloc[:, -1].to_numpy()

#     rng = np.random.default_rng(random_state)

#     classes, counts = np.unique(y_all, return_counts=True)
#     min_cnt = counts.min()
#     idx_balanced = np.concatenate([
#         rng.choice(np.where(y_all == c)[0], size=min_cnt, replace=False)
#         for c in classes
#     ])
#     rng.shuffle(idx_balanced)
#     X_all, y_all = X_all[idx_balanced], y_all[idx_balanced]


#     return X_all, y_all

def load_fico(
    random_state: int | None = None,
):
    df = pd.read_csv("datasets/fico.csv")

    sentinel_vals = [-7, -8, -9]
    df.replace(sentinel_vals, np.nan, inplace=True)

    feat_cols = df.columns[:-1] 
    k = min(3, len(feat_cols))
    topk = df[feat_cols].isna().mean().nlargest(k).index.tolist()
    if topk:
        df.drop(columns=topk, inplace=True)
        feat_cols = df.columns[:-1]  

    row_missing_pct = df[feat_cols].isna().mean(axis=1)
    df = df.loc[row_missing_pct <= 0.40].copy()

    df[feat_cols] = df[feat_cols].fillna(-1)

    X_all = df.loc[:, feat_cols].to_numpy(dtype=np.float32)
    y_all = df.iloc[:, -1].to_numpy()

    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(y_all, return_counts=True)
    min_cnt = counts.min()
    idx_balanced = np.concatenate([
        rng.choice(np.where(y_all == c)[0], size=min_cnt, replace=False)
        for c in classes
    ])
    rng.shuffle(idx_balanced)
    X_all, y_all = X_all[idx_balanced], y_all[idx_balanced]

    return X_all, y_all



def load_dataset(name: str, dataset_size = 10_000) -> Tuple[np.ndarray, np.ndarray]:
    if name == "AdultBalanced":
        return load_adult_dataset_balanced(subsample_size=dataset_size)
    if name == "Polish":
        return load_polish()
    if name == "Phonemes":
        return load_phonemes()
    if name.lower() in {"amsterdam", "ansterdam"}:
        return load_amsterdam()
    if name.lower() in {"germanc"}:
        return load_germanc()
    if name.lower() in {"austrc", "australian"}:
        return load_austrc()
    if name.lower() in {"compas", "compass"}:
        return load_compas()
    if name.lower() in {"diabetescnt", "diabetes_cnt"}:
        return load_diabetes_cnt()
    if name.lower() in {"diabetes"}:
        return load_diabetes()
    if name.lower() in {"fico"}:
        return load_fico()
    if name.lower() in {"iris"}:
        return load_iris_ext() #use extended iris dataset
    if name.lower() in {"banknote", "banknote_authentication"}:
        return load_banknote()
    if name.lower() in {"winequality"}:
        return load_wine_quality()
    if name.lower() in {"parkinsons"}:
        return load_parkinsons()
    raise ValueError(f"Unknown dataset {name}")


