import itertools
import pickle
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pyod.models.ecod import ECOD
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from data.data_getter import (
    get_categorical_dataset,
    get_glocalkd_dataset,
    get_multiomics_data,
    get_npz_dataset,
    get_sets_data,
    get_timeseries,
)
from risf.distance import (
    DistanceMixin,
    SelectiveDistance,
    TestDistanceMixin,
    TrainDistanceMixin,
    split_distance_mixin,
)
from risf.distance_functions import *
from risf.forest import RandomIsolationSimilarityForest
from risf.risf_data import RisfData

PRECOMPUTED_DISTANCES_PATH = Path("../precomputed_distances")
PRECOMPUTED_DISTANCES_PATH.mkdir(exist_ok=True)
BEST_DISTANCES_PATH = Path("../best_distances")
BEST_DISTANCES_PATH.mkdir(exist_ok=True)
SEED = 23
N_REPEATED_HOLDOUT = 10
N_REPEATED_HOLDOUT_BEST_DIST = 3
TEST_HOLDOUT_SIZE = 0.3

MIN_N_SELECTED = 10
SELECTED_OBJ_RATIO = 0.5


def get_dataset(type_, data_folder: Path, name: str, clf: str) -> dict:
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
    if type_ in ["numerical", "nlp", "cv"]:
        return get_npz_dataset(data_folder / (name + ".npz"))

    if type_ in ["binary", "nominal"]:
        return get_categorical_dataset(data_folder / (name + ".csv"), clf)

    if type_ == "graph":
        return get_glocalkd_dataset(data_folder, name)

    if type_ == "timeseries":
        return get_timeseries(data_folder, name)

    if type_ == "multiomics":
        return get_multiomics_data(data_folder, name, clf == "RISF")

    if type_ == "seq_of_sets":
        return get_sets_data(data_folder, name)


def new_clf(name, SEED, clf_kwargs={}):
    if name == "ECOD":
        return ECOD()
    if name == "LOF":
        return LOF(**clf_kwargs)
    if name == "IForest":
        return IForest(random_state=SEED, **clf_kwargs)
    if name == "HBOS":
        return HBOS()
    if name == "RISF":
        return RandomIsolationSimilarityForest(random_state=SEED, **clf_kwargs)
    if name == "ISF":
        return RandomIsolationSimilarityForest(random_state=SEED, **clf_kwargs)
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


def get_binary_distances_choice(distances: np.ndarray) -> np.ndarray:
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


def get_distance_path(dataset_name, distance, id_=0):
    if distance.__class__.__name__ == "GraphDist":
        dist_name = distance.distance.__class__.__name__
    else:
        dist_name = distance.__class__.__name__

    return PRECOMPUTED_DISTANCES_PATH / f"{dataset_name}_{dist_name}_{id_}.pickle"


def precompute_distances(
    data: dict,
    distances: List,
    selected_objects: Optional[np.ndarray] = None,
    n_jobs: int = -3,
    id_=0,
):
    """Precomputes all distances and saves them to file."""
    if not isinstance(distances, list):
        distances = [distances]

    X = data["X"]
    for distance in distances:
        entire_distance = TrainDistanceMixin(
            distance, selected_objects=selected_objects
        )
        entire_distance.precompute_distances(X, n_jobs=n_jobs)
        with open(get_distance_path(data["name"], distance, id_), "wb") as f:
            pickle.dump(entire_distance, f)


def split_all_distances(
    distances: list[TrainDistanceMixin], train_indices: np.ndarray, test_index
) -> Tuple[List[TrainDistanceMixin], List[TestDistanceMixin]]:
    """Used for Cross Validation. Splits all distances into train and test parts."""
    train_distances = []
    test_distances = []

    for whole_distance in distances:
        if isinstance(whole_distance, DistanceMixin):
            train_distance, test_distance = split_distance_mixin(
                whole_distance, train_indices, test_index
            )
        else:
            train_distance, test_distance = whole_distance, whole_distance
        train_distances.append(train_distance)
        test_distances.append(test_distance)

    return train_distances, test_distances


def selection_choice(train_index, n_selected_obj, fold_id=None):
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


def get_risf_distances(
    data: dict, distances: List = None, selected_objects=None, id_=0
):
    """Returns distances used in RISF."""
    new_distances = []
    for distance in distances:
        if not isinstance(distance, SelectiveDistance):
            distance_path = get_distance_path(data["name"], distance, id_=id_)

            if not distance_path.exists():
                precompute_distances(
                    data, [distance], selected_objects=selected_objects, id_=id_
                )

            with open(distance_path, "rb") as f:
                new_distances.append(pickle.load(f))
        else:
            new_distances.append(distance)

    return new_distances


def get_n_selected_obj(selected_obj_ratio, all_indices):
    return max(
        [
            int(selected_obj_ratio * len(all_indices) * (1 - TEST_HOLDOUT_SIZE)),
            MIN_N_SELECTED,
        ]
    )


def get_splits(all_indices, fold_id, y):
    # This split should be deterministic and the same for every dataset experiment
    train_index, test_index = train_test_split(
        all_indices,
        test_size=TEST_HOLDOUT_SIZE,
        random_state=SEED + fold_id,
        stratify=y,
    )

    train_index = np.sort(train_index)
    test_index = np.sort(test_index)

    return train_index, test_index


def get_risf_auc(X_risf, X_test, test_distances, y_test, clf_kwargs):
    n_jobs = -2

    clf = RandomIsolationSimilarityForest(
        random_state=SEED,
        distances=X_risf.distances,
        n_jobs=n_jobs,
        **clf_kwargs,
    ).fit(X_risf)

    X_test_risf = X_risf.transform(
        X_test,
        forest=clf,
        n_jobs=-2,
        precomputed_distances=test_distances,
    )

    y_test_pred = (-1) * clf.predict(X_test_risf, return_raw_scores=True)
    return np.round(roc_auc_score(y_test, y_test_pred), decimals=4)


def find_best_distances(
    X,
    y,
    distances,
    data_type,
    data_name,
    fold_id,
    clf_name,
    max_size=3,
    old_train_index=None,
    id_=0,
):
    PICKLE_PATH = (
        BEST_DISTANCES_PATH / f"{clf_name}_{data_name}{id_ if id_ != 0 else ''}.pickle"
    )
    best_distances = {}
    if Path(PICKLE_PATH).exists():
        best_distances = pickle.load(open(PICKLE_PATH, "rb"))

    if clf_name == "ISF":
        max_size = 1

    if fold_id not in best_distances:
        results_distance = []
        data = {"X": X, "y": y, "name": data_name}
        distances_to_use = get_binary_distances_choice(distances)
        for dist_to_use in distances_to_use:
            if dist_to_use.sum() > max_size or dist_to_use.sum() == 0:
                continue

            dist = list(distances[dist_to_use])
            if (
                data_type
                in [
                    "timeseries",
                    "graph",
                    "binary",
                    "nominal",
                    "seq_of_sets",
                    "multiomics",
                    "histogram",
                ]
                or clf_name == "ISF"
            ):
                aucs = experiment_risf_complex(clf_name, data, dist, n_holdouts=N_REPEATED_HOLDOUT_BEST_DIST, 
                                                optimize_distances=False, clf_kwargs={}, old_train_index=old_train_index, id_ = id_)  # fmt: skip

            elif data_type in ["numerical", "nlp", "cv"]:
                aucs = perform_experiment_simple(
                    clf_name, data, n_holdouts=N_REPEATED_HOLDOUT_BEST_DIST, distances=dist, optimize_distances=False, clf_kwargs={}
                )  # fmt: skip

            results_distance.append((np.mean(aucs), dist))

        results_distance.sort(key=lambda x: x[0], reverse=True)

        best_distances[fold_id] = results_distance
        pickle.dump(best_distances, open(PICKLE_PATH, "wb"))

    return best_distances[fold_id][0][1]


def experiment_risf_mixed(
    data: dict,
    distances: List[List],
    selected_obj_ratio: float = SELECTED_OBJ_RATIO,
    selection_func: Union[Callable, ObjectsSelector] = ObjectsSelector(),
    clf_kwargs={},
    optimize_distances=False,
):
    """In this case in our data we assume that X is actually a list of objects and distances are list of lists"""
    y = data["y"]
    all_indices = np.arange(len(y))

    n_selected_obj = get_n_selected_obj(selected_obj_ratio, all_indices)

    auc = []
    for fold_id in range(N_REPEATED_HOLDOUT):
        train_index, test_index = get_splits(all_indices, fold_id, y)

        # I wonder if every feature should have different selected objects
        selected_objects = selection_func(
            np.arange(len(train_index)), n_selected_obj, fold_id
        )

        X_risf = RisfData(random_state=SEED)
        test_features = []
        all_test_distances = []
        y_train, y_test = y[train_index], y[test_index]
        for f_id in range(len(data["X"])):
            feature, data_type = data["X"][f_id], data["features_types"][f_id]
            distance = distances[f_id]
            # To make sure distances are precalculated on entire data
            _ = get_risf_distances(
                {"X": feature, "name": data["name"]}, distance, id_=f_id
            )

            X_train, X_test = feature[train_index], feature[test_index]

            if optimize_distances:
                distance = find_best_distances(X_train, y_train, distance, data_type, data['name'], fold_id, clf_name="RISF", old_train_index=train_index, id_=f_id)  # fmt: skip

            distance = get_risf_distances(
                {"X": feature, "name": data["name"]}, distance, id_=f_id
            )

            if isinstance(distance[0], TrainDistanceMixin):
                train_distances, test_distances = split_all_distances(distance, train_index, test_index)  # fmt: skip
            else:
                train_distances, test_distances = distance, distance

            X_risf.add_data(X_train, dist=train_distances)
            X_risf.precompute_distances(selected_objects=selected_objects)

            test_features.append(X_test)
            all_test_distances.append(test_distances)

        auc.append(get_risf_auc(X_risf, test_features, all_test_distances, y_test, clf_kwargs))  # fmt: skip

    return np.array(auc)


def experiment_risf_complex(
    clf_name: str,
    data: dict,
    distances: List,
    selected_obj_ratio: float = SELECTED_OBJ_RATIO,
    selection_func: Union[Callable, ObjectsSelector] = ObjectsSelector(),
    n_holdouts=N_REPEATED_HOLDOUT,
    clf_kwargs={},
    optimize_distances: bool = False,
    old_train_index=None,
    id_=0,
):
    """Perform experiments for RISF on complex or mixed data types.

    2. Selects n_selected_obj objects from train_index (Between how many objects it is allowed to use distances
    3. Perform CV

    """
    X, y = data["X"], data["y"]
    all_indices = np.arange(len(X))

    n_selected_obj = get_n_selected_obj(selected_obj_ratio, all_indices)

    auc = []
    for fold_id in range(n_holdouts):
        train_index, test_index = get_splits(all_indices, fold_id, y)

        selected_objects = selection_func(
            np.arange(len(train_index)), n_selected_obj, fold_id
        )

        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]  # fmt: skip

        #! IF we do this for Precalculated distances we need at firs to precompute them, so that we don't do it on smaller subset of the data!
        distances_for_train = get_risf_distances(data, distances, id_=id_)
        if optimize_distances:
            best_distances = find_best_distances(X_train, y_train, distances, data["type"], data['name'], fold_id, clf_name, old_train_index=train_index)  # fmt: skip
            distances_for_train = get_risf_distances(data, best_distances, id_=id_)

        if old_train_index is not None:
            #! We must have correct indices for precomputed distances
            train_index = old_train_index[train_index]
            test_index = old_train_index[test_index]

        train_distances, test_distances = split_all_distances(
            distances_for_train, train_index, test_index
        )

        X_risf = RisfData(random_state=SEED)
        X_risf.add_data(X_train, dist=train_distances)
        X_risf.precompute_distances(selected_objects=selected_objects)

        auc.append(get_risf_auc(X_risf, [X_test], [test_distances], y_test, clf_kwargs))

    return np.array(auc)


def optimize_lof(X_train, y_train, data_name, fold_id):
    df = pd.read_csv("../best_distances/LOF.csv")

    candidates = df[(df["dataset"] == data_name) & (df["fold_id"] == fold_id)]
    if candidates.empty:
        for metric in ["cosine", "euclidean", "manhattan"]:
            aucs = perform_experiment_simple(
                "LOF",
                {"X": X_train, "y": y_train},
                clf_kwargs={"metric": metric},
                n_holdouts=N_REPEATED_HOLDOUT_BEST_DIST,
                distances=None,
                optimize_distances=False,
            )
            new_row = {"metric": metric, "auc": np.mean(aucs), "dataset": data_name, "fold_id": fold_id}  # fmt: skip
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        df.to_csv("../best_distances/LOF.csv", index=False)

    candidates = df.sort_values("auc", ascending=False)
    return candidates.iloc[0, 0]


def perform_experiment_simple(
    clf_name: str,
    data: dict,
    clf_kwargs: dict,
    n_holdouts: int = N_REPEATED_HOLDOUT,
    distances: Optional[List[List]] = None,
    optimize_distances: bool = False,
):
    """
    For much simpler experiments performed on numerical data and plain numpy arrays
    """
    X, y = data["X"], data["y"]
    all_indices = np.arange(X.shape[0])
    auc = []

    for fold_id in range(n_holdouts):
        train_index, test_index = get_splits(all_indices, fold_id, y)

        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]  # fmt: skip

        if clf_name in "RISF":
            best_distances = distances
            if optimize_distances:
                best_distances = find_best_distances(X_train, y_train, distances, data["type"], data['name'], fold_id, clf_name)  # fmt: skip

            clf = RandomIsolationSimilarityForest(
                random_state=SEED,
                distances=[best_distances],
                n_jobs=-2,
                **clf_kwargs,
            ).fit(X_train)
            y_test_pred = (-1) * clf.predict(X_test, return_raw_scores=True)

        else:
            if clf_name == "LOF" and optimize_distances:
                clf_kwargs = {
                    "metric": optimize_lof(X_train, y_train, data["name"], fold_id)
                }

            clf = new_clf(clf_name, SEED, clf_kwargs)
            clf.fit(X_train)
            y_test_pred = clf.decision_function(X_test)

        auc.append(np.round(roc_auc_score(y_test, y_test_pred), decimals=4))

    return np.array(auc)
