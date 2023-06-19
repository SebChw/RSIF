import itertools
import pickle
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from isolation_simforest import IsolationSimilarityForest
from pyod.models.ecod import ECOD
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm.auto import trange

from data.data_getter import (
    get_categorical_dataset,
    get_glocalkd_dataset,
    get_npz_dataset,
    get_timeseries,
)
from risf.distance import (
    SelectiveDistance,
    TestDistanceMixin,
    TrainDistanceMixin,
    split_distance_mixin,
)
from risf.distance_functions import dice_projection, jaccard_projection
from risf.forest import RandomIsolationSimilarityForest
from risf.risf_data import RisfData

PRECOMPUTED_DISTANCES_PATH = Path("../precomputed_distances")
PRECOMPUTED_DISTANCES_PATH.mkdir(exist_ok=True)
SEED = 23
N_REPEATED_HOLDOUT = 3  #! 10 in the final experiments
TEST_HOLDOUT_SIZE = 0.3

MIN_N_SELECTED = 10


def get_dataset(
    type_, data_folder: Path, name: str, numerical_features: bool = False
) -> dict:
    """Return given dataset.

    Parameters
    ----------
    type_ : str
        data type of dataset can be numerical, categorical, graph or timeseries
    data_folder : Path
        folder with data
    name : str
        name of dataset
    numerical_features : bool, optional
        Used for complex datatypes to convert them into some numerical representation that can be feed to IF, by default False

    Returns
    -------
    dict
        Dictonary with format defined in every dataset getter
    """
    if type_ == "numerical":
        return get_npz_dataset(data_folder / (name + ".npz"))

    if type_ == "categorical":
        return get_categorical_dataset(data_folder / (name + ".csv"))

    if type_ == "graph":
        return get_glocalkd_dataset(data_folder, name, numerical_features)

    if type_ == "timeseries":
        return get_timeseries(data_folder, name)


def new_clf(name, SEED, clf_kwargs={}):
    if name == "ECOD":
        return ECOD()
    if name == "LOF":
        return LOF()
    if name == "IForest":
        return IForest(random_state=SEED, **clf_kwargs)
    if name == "HBOS":
        return HBOS()
    if name == "RISF":
        return RandomIsolationSimilarityForest(random_state=SEED, **clf_kwargs)
    if name == "ISF":
        return IsolationSimilarityForest(random_state=SEED, **clf_kwargs)
    else:
        raise NotImplementedError()


def init_results(clf_name, dataset_name, dataset_type, aucs, clf_kwargs={}):
    results = []
    for auc in aucs:
        results.append(
            {
                "clf": clf_name,
                "dataset_name": dataset_name,
                "dataset_type": dataset_type,
                "auc": auc,
                **clf_kwargs,
            }
        )

    return results


def get_binary_distances_choice(distances: list) -> np.ndarray:
    """Function used to create all combinations of used distances.

    Parameters
    ----------
    distances : list

    Returns
    -------
    np.ndarray
    """
    return np.array(
        [
            list(option)
            for option in itertools.product([True, False], repeat=len(distances))
            if list(option) != [False] * len(distances)
        ]
    )


def get_distance_path(dataset_name, distance):
    return (
        PRECOMPUTED_DISTANCES_PATH
        / f"{dataset_name}_{distance.__class__.__name__}.pickle"
    )


def precompute_distances(
    data: np.ndarray,
    distances: List[TrainDistanceMixin],
    selected_objects: Optional[np.ndarray] = None,
    n_jobs: int = -3,
):
    """Precomputes all distances and saves them to file."""
    X = data["X"]
    for distance in distances:
        entire_distance = TrainDistanceMixin(
            distance, selected_objects=selected_objects
        )
        entire_distance.precompute_distances(X, n_jobs=n_jobs)
        with open(get_distance_path(data["name"], distance), "wb") as f:
            pickle.dump(entire_distance, f)


def split_all_distances(
    distances: list[TrainDistanceMixin], train_indices: np.ndarray
) -> Tuple[TrainDistanceMixin, TestDistanceMixin]:
    """Used for Cross Validation. Splits all distances into train and test parts."""
    train_distances = []
    test_distances = [[]]

    for whole_distance in distances:
        train_distance, test_distance = split_distance_mixin(
            whole_distance, train_indices
        )
        train_distances.append(train_distance)
        test_distances[0].append(test_distance)

    return train_distances, test_distances


def selection_choice(train_index, n_selected_obj):
    """Selects n_selected_obj objects from train_index."""
    return np.random.choice(train_index, n_selected_obj, replace=False)


class ObjectsSelector:
    """To make distances comparison fair. We force the same CV splits for each ratio."""

    def __init__(
        self,
    ):
        self.splits = []

    def __call__(self, train_index, n_selected_obj, fold_id):
        if len(self.splits) < N_REPEATED_HOLDOUT:
            #! TO provide same first n selected objects for each ratio
            train_idx = train_index.copy()
            np.random.shuffle(train_idx)
            self.splits.append(train_idx)

        return self.splits[fold_id][:n_selected_obj]


def experiment_risf_complex(
    data: RisfData,
    distances: List[TrainDistanceMixin],
    selected_obj_ratio: float,
    selection_func: Union[Callable, ObjectsSelector] = selection_choice,
    clf_kwargs={},
):
    """Perform experiments for RISF on complex or mixed data types.

    1. Precompute distances or loads them if already precalculated
    2. Selects n_selected_obj objects from train_index (Between how many objects it is allowed to use distances
    3. Perform CV

    """
    distances_path = [
        get_distance_path(data["name"], distance) for distance in distances
    ]

    distances_calculated = []
    for i, distance_path in enumerate(distances_path):
        if not distance_path.exists():
            precompute_distances(data, distances)

        with open(distance_path, "rb") as f:
            distances_calculated.append(pickle.load(f))

    X, y = data["X"], data["y"]
    all_indices = np.arange(X.shape[0])

    n_selected_obj = max(
        [
            int(selected_obj_ratio * len(all_indices) * (1 - TEST_HOLDOUT_SIZE)),
            MIN_N_SELECTED,
        ]
    )

    auc = []

    for fold_id in trange(N_REPEATED_HOLDOUT, desc="repeated holdout"):
        # This split should be deterministic and the same for every dataset experiment
        train_index, test_index = train_test_split(
            all_indices,
            test_size=TEST_HOLDOUT_SIZE,
            random_state=SEED + fold_id,
            stratify=y,
        )

        # Pomyslec o madrzejszym probkowaniu a nie tylko losowym.
        # Wybierac punkty ktory sa ze soba rozroznialne.
        # Sprawdzic czy np. sa niezerowe odleglosci miedzy nimi. dla kazdego feature.
        selected_objects = selection_func(train_index, n_selected_obj, fold_id)

        train_distances, test_distances = split_all_distances(
            distances_calculated, train_index
        )
        X_train, X_test, y_test = X[train_index], X[test_index], y[test_index]

        X_risf = RisfData(random_state=SEED)
        X_risf.add_data(X_train, dist=[train_distances])
        X_risf.precompute_distances(selected_objects=selected_objects)

        n_jobs = 1

        clf = RandomIsolationSimilarityForest(
            random_state=SEED,
            distances=X_risf.distances,
            n_jobs=n_jobs,
            **clf_kwargs,
        ).fit(X_risf)

        X_test_risf = X_risf.transform(
            [X_test],
            forest=clf,
            n_jobs=-2,
            precomputed_distances=[test_distances],
        )

        y_test_pred = (-1) * clf.predict(X_test_risf, return_raw_scores=True)

        auc.append(np.round(roc_auc_score(y_test, y_test_pred), decimals=4))

    return np.array(auc)


def perform_experiment_simple(
    clf_name: str,
    data: np.ndarray,
    distances: List[SelectiveDistance],
    clf_kwargs: dict,
    dtype_="numerical",
):
    """
    For much simpler experiments performed on numerical data and plain numpy arrays
    """
    X, y = data["X"], data["y"]
    all_indices = np.arange(X.shape[0])
    auc = []

    for fold_id in range(N_REPEATED_HOLDOUT):
        # This split should be deterministic and the same for every dataset experiment
        train_index, test_index = train_test_split(
            all_indices,
            test_size=TEST_HOLDOUT_SIZE,
            random_state=SEED + fold_id,
            stratify=y,
        )

        X_train, X_test, y_test = X[train_index], X[test_index], y[test_index]

        if clf_name == "RISF":
            # Check categorical features using jaccard distance!
            clf = RandomIsolationSimilarityForest(
                random_state=SEED,
                distances=distances,
                n_jobs=-1,
                **clf_kwargs,
            ).fit(X_train)

            y_test_pred = (-1) * clf.predict(X_test, return_raw_scores=True)

        else:
            clf = new_clf(clf_name, SEED, clf_kwargs)  #! Give kwargs to clf
            clf.fit(X_train)
            y_test_pred = clf.decision_function(X_test)

        auc.append(np.round(roc_auc_score(y_test, y_test_pred), decimals=4))

    return np.array(auc)
