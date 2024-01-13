import itertools
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from data.data_getter import (
    get_categorical_dataset,
    get_glocalkd_dataset,
    get_multiomics_data,
    get_npz_dataset,
    get_sets_data,
    get_timeseries,
)
from pyod.models.ecod import ECOD
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from rsif.distance import (
    DistanceMixin,
    SelectiveDistance,
    TestDistanceMixin,
    TrainDistanceMixin,
    split_distance_mixin,
)
from rsif.distance_functions import *
from rsif.forest import RandomSimilarityIsolationForest
from rsif.rsif_data import RsifData
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

PRECOMPUTED_DISTANCES_PATH = Path("../precomputed_distances")
BEST_DISTANCES_PATH = Path("../best_distances")


SEED = 23
np.random.seed(SEED)
N_REPEATED_HOLDOUT = 10
N_REPEATED_HOLDOUT_BEST_DIST = 3
TEST_HOLDOUT_SIZE = 0.3

MIN_N_SELECTED = 50  # Minimal number of objects selected for calculating projections. Necessary for very small datasets
SELECTED_OBJ_RATIO = 0.5

LOG_PATH = Path("../logs")
LOG_PATH.mkdir(exist_ok=True, parents=True)

SCORES_PATH = Path("../scores")
SCORES_PATH.mkdir(exist_ok=True, parents=True)

def append_scores(dataset_name: str, alg_name: str, fold_id:int, scores: List[float]):
    path = SCORES_PATH / f"{dataset_name}_{fold_id}.csv"

    if not path.exists():
        df= pd.DataFrame({alg_name: scores})
    else:
        df = pd.read_csv(path)
        df[alg_name] = scores
    
    df.to_csv(path, index=False)


class NJobs:
    n_jobs = -1

    @classmethod
    def set_n_jobs(cls, n_jobs):
        if not isinstance(n_jobs, int):
            raise ValueError("n_jobs must be integer")
        cls.n_jobs = n_jobs


def get_logger(name: str) -> logging.Logger:
    """Returns logger with given name."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    ch = logging.FileHandler(LOG_PATH / f"{name}.log")
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p"
    )
    ch.setFormatter(formatter)

    logger.addHandler(ch)

    return logger


def check_precomputed_notebook():
    """Check if precomputed distances are downloaded."""
    if not Path("../precomputed_distances").exists():
        raise ValueError(
            "Please download precomputed distances matrices from Zenodo first"
        )


def ensure_existance(paths: List[Path]):
    for path in paths:
        path.mkdir(exist_ok=True, parents=True)


ensure_existance([BEST_DISTANCES_PATH])


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
        return get_categorical_dataset(data_folder / (name + ".csv"))

    if type_ == "graph":
        return get_glocalkd_dataset(data_folder, name)

    if type_ == "timeseries":
        return get_timeseries(data_folder, name)

    if type_ == "multiomics":
        return get_multiomics_data(data_folder, name, clf == "RSIF")

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
    if name == "RSIF":
        return RandomSimilarityIsolationForest(random_state=SEED, **clf_kwargs)
    if name == "ISF":
        return RandomSimilarityIsolationForest(random_state=SEED, **clf_kwargs)
    else:
        raise NotImplementedError()


def init_results(
    clf_name: str, dataset_name: str, dataset_type: str, aucs: np.ndarray, clf_kwargs={}
) -> List[Dict]:
    """Initializes results for given experiment. For every entry in aucs one dict is created. This format is easily converted to pandas DataFrame.

    Returns
    -------
    List[Dict]
        list of dicts with one entry per auc.
    """
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
    distances : np.ndarray

    Returns
    -------
    np.ndarray
        All combinations of used distances. True/False indicate whether distance is used or not.
    """
    return np.array(
        [
            list(option)
            for option in itertools.product([True, False], repeat=len(distances))
            if list(option) != [False] * len(distances)
        ]
    )


def get_distance_path(dataset_name: str, distance, id_=0) -> Path:
    """Distances paths are created based on dataset name and used function.
    It may happen that For some dataset you will have few features with same distance function.
    Then you can use id_ to distinguish them."""

    if distance.__class__.__name__ == "GraphDist":
        dist_name = distance.distance.__class__.__name__
    else:
        dist_name = distance.__class__.__name__

    return PRECOMPUTED_DISTANCES_PATH / f"{dataset_name}_{dist_name}_{id_}.npy"


def precompute_distances(
    data: dict,
    distances: List,
    selected_objects: Optional[np.ndarray] = None,
    id_=0,
):
    """Precomputes all distances and saves them to file."""
    if not isinstance(distances, list):
        distances = [distances]

    X = data["X"]
    returned_dists = []
    for distance in distances:
        entire_distance = TrainDistanceMixin(
            distance, selected_objects=selected_objects
        )
        distance_path = get_distance_path(data["name"], distance, id_)
        if distance_path.exists():
            distance_matrix = np.load(distance_path)
            entire_distance.distance_matrix = distance_matrix
        else:
            entire_distance.precompute_distances(X, n_jobs=NJobs.n_jobs)
            np.save(distance_path, entire_distance.distance_matrix)

        returned_dists.append(entire_distance)
    return returned_dists


def split_all_distances(
    distances: list[TrainDistanceMixin], train_indices: np.ndarray, test_index
) -> Tuple[List[TrainDistanceMixin], List[TestDistanceMixin]]:
    """Used for Cross Validation. Splits distance matrix calculated for entire dataset into train and test distance matrices.
    Split are only necessary for DistanceMixin which are based on distance matrix. For other, distances are calculated on the fly
    """
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
    """This object is used during sensitivity analysis for testing influence of SELECTED_OBJECTS_RATIO parameter
    To make comparison fair, we force set of objects obtained bigger n_selected_obj to be supersets of smaller.
    """

    def __init__(self, n_holdouts: int):
        self.splits = []
        self.n_holdouts = n_holdouts

    def __call__(
        self, train_index: np.ndarray, n_selected_obj: int, fold_id: int
    ) -> np.ndarray:
        """Returns indices of objects that can be used for calculating projections.

        Parameters
        ----------
        train_index : np.ndarray
            Indices of objects for training
        n_selected_obj : int
            number of objects that will be used for calculating projections
        fold_id : int

        Returns
        -------
        np.ndarray
            Indices of objects that can be used for calculating projections.
        """
        if len(self.splits) < self.n_holdouts:
            train_idx = train_index.copy()
            np.random.shuffle(train_idx)
            self.splits.append(train_idx)

        return self.splits[fold_id][:n_selected_obj]


def get_rsif_distances(
    data: dict, distances: List = None, selected_objects=None, id_=0
) -> List[Union[DistanceMixin, SelectiveDistance]]:
    """Returns List of actual distance objects that can work with RSIF. If possible reloads precomputed distances from file."""
    new_distances = []
    for distance in distances:
        if not isinstance(distance, SelectiveDistance):
            distance = precompute_distances(
                data, [distance], selected_objects=selected_objects, id_=id_
            )
            new_distances.append(distance[0])
        else:
            new_distances.append(distance)

    return new_distances


def get_n_selected_obj(selected_obj_ratio: float, all_indices: np.ndarray) -> int:
    """Calculates number of objects that will be used for calculating projections.
    Some datasets are very small and if object_ratio is 0.1 we got only few objects."""
    return max(
        [
            int(selected_obj_ratio * len(all_indices) * (1 - TEST_HOLDOUT_SIZE)),
            MIN_N_SELECTED,
        ]
    )


def get_splits(
    all_indices: np.ndarray, fold_id: int, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns deterministic train and test splits stratified by y."""
    train_index, test_index = train_test_split(
        all_indices,
        test_size=TEST_HOLDOUT_SIZE,
        random_state=SEED + fold_id,
        stratify=y,
    )

    train_index = np.sort(train_index)
    test_index = np.sort(test_index)

    return train_index, test_index


def get_rsif_auc(
    X_rsif: RsifData,
    X_test: List[np.ndarray],
    test_distances: List[List[Union[DistanceMixin, SelectiveDistance]]],
    y_test: np.ndarray,
    clf_kwargs: Dict,
    clf_name: str,
    data_name: str,
    fold_id:int
) -> float:
    """Fit rsif -> transform test data -> predict -> calculate auc.

    Parameters
    ----------
    X_rsif : RsifData
        Train dataset wrapped in Rsif representation
    X_test : List[np.ndarray]
        Given list of numpy arrays RSIF will transform it into RSIF representation
    test_distances : List[List[Union[DistanceMixin, SelectiveDistance]]]
        List of distances for every feature.

    Returns
    -------
    float
        AUC score obtained on test data
    """
    clf = RandomSimilarityIsolationForest(
        random_state=SEED,
        distances=X_rsif.distances,
        n_jobs=NJobs.n_jobs,
        **clf_kwargs,
    ).fit(X_rsif)

    X_test_rsif = X_rsif.transform(
        X_test,
        forest=clf,
        n_jobs=NJobs.n_jobs,
        precomputed_distances=test_distances,
    )

    y_test_pred = (-1) * clf.predict(X_test_rsif, return_raw_scores=True)

    append_scores(data_name, clf_name, fold_id, y_test_pred)
    return np.round(roc_auc_score(y_test, y_test_pred), decimals=4)


def find_best_distances(
    X: np.ndarray,
    y: np.ndarray,
    distances: np.ndarray,
    data_type: str,
    data_name: str,
    fold_id: int,
    clf_name: str,
    max_size: int = 3,
    old_train_index: np.ndarray = None,
    id_: int = 0,
) -> List[Union[DistanceMixin, SelectiveDistance]]:
    """Performs search over all possible combinations of distances up to length of `max_size` and returns the best one.

    Parameters
    ----------
    X : np.ndarray
    y : np.ndarray
    distances : np.ndarray
        numpy array of distances from which we will sample.
    data_type : str
        based on data type complex/numerical we will run different functions
    data_name : str
    fold_id : int
        We perform it within nested cross validation so we need to know which fold we are in.
    clf_name : str
        based on classifier RSIF/ISF we will run different functions
    max_size : int, optional
        To make this search faster we can restrict maximum length of distances, by default 3
    old_train_index : np.ndarray, optional
        This is to use correct distance matrices, by default None
    id_ : int, optional
        Needed to distinguish features with the same distances, by default 0

    Returns
    -------
    List[Union[DistanceMixin, SelectiveDistance]]
        Distances with highest AUC score
    """
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
            if data_type in ["timeseries", "graph", "binary", "nominal", "seq_of_sets", "multiomics", "histogram"] or clf_name == "ISF":  # fmt: skip
                aucs = experiment_rsif_complex(clf_name, data, dist, n_holdouts=N_REPEATED_HOLDOUT_BEST_DIST, 
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


def optimize_lof(
    X_train: np.ndarray, y_train: np.ndarray, data_name: str, fold_id: int
) -> str:
    """Since LOF is also distance based we can optimize it in the same way as RSIF or ISF.

    Returns
    -------
    str
        One of "cosine", "euclidean", "manhattan"
    """
    lof_dist_path = BEST_DISTANCES_PATH / "LOF.csv"
    if not lof_dist_path.exists():
        with open(lof_dist_path, "w") as f:
            f.write("metric,auc,dataset,fold_id\n")

    df = pd.read_csv(lof_dist_path)

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

        df.to_csv(lof_dist_path, index=False)

    candidates = df.sort_values("auc", ascending=False)
    return candidates.iloc[0, 0]


def perform_experiment_simple(
    clf_name: str,
    data: dict,
    clf_kwargs: dict,
    n_holdouts: int = N_REPEATED_HOLDOUT,
    distances: Optional[List[List]] = None,
    optimize_distances: bool = False,
) -> np.ndarray:
    """When we operate on simple numerical data we can use this function to perform experiments.

    Parameters
    ----------
    clf_name : str
        name of classifier to be used
    data : dict
        dictionary with three keys X, y, name
    clf_kwargs : dict
        additional arguments for classifier
    n_holdouts : int, optional
        How many repeated holdout will be done, by default N_REPEATED_HOLDOUT
    distances : Optional[List[List]], optional
        distances for RSIF to be used, by default None
    optimize_distances : bool, optional
        Whether we should find best distances, by default False

    Returns
    -------
    np.ndarray
        AUC score obtained for every split from repeated holdout
    """
    X, y = data["X"], data["y"]
    all_indices = np.arange(X.shape[0])
    auc = []

    for fold_id in range(n_holdouts):
        train_index, test_index = get_splits(all_indices, fold_id, y)

        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]  # fmt: skip

        if clf_name in "RSIF":
            best_distances = distances
            if optimize_distances:
                best_distances = find_best_distances(X_train, y_train, distances, data["type"], data['name'], fold_id, clf_name)  # fmt: skip

            clf = RandomSimilarityIsolationForest(
                random_state=SEED,
                distances=[best_distances],
                n_jobs=NJobs.n_jobs,
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

        append_scores(data["name"], clf_name, fold_id, y_test_pred)
        auc.append(np.round(roc_auc_score(y_test, y_test_pred), decimals=4))

    return np.array(auc)


def experiment_rsif_complex(
    clf_name: str,
    data: dict,
    distances: List,
    selected_obj_ratio: float = SELECTED_OBJ_RATIO,
    n_holdouts=N_REPEATED_HOLDOUT,
    clf_kwargs={},
    optimize_distances: bool = False,
    old_train_index: np.ndarray = None,
    id_=0,
) -> np.ndarray:
    """Perform experiments for RSIF on complex or mixed data types.

    2. Selects n_selected_obj objects from train_index (Between how many objects it is allowed to use distances
    3. Perform CV

    Parameters
    ----------
    clf_name : str
        name of classifier to be used
    data : dict
        dictionary with three keys X, y, name
    distances : List
        distances for RSIF to be used
    selected_obj_ratio : float, optional
        how many object from training split will be actually used during training phase, by default SELECTED_OBJ_RATIO
    n_holdouts : _type_, optional
        how many repetitions of repeated holdout will be performed, by default N_REPEATED_HOLDOUT
    clf_kwargs : dict, optional
        additional parameters that will be pased to clasifiers, by default {}
    optimize_distances : bool, optional
        Whether one should perform distance optimization, by default False
    old_train_index : np.ndarray, optional
        This is necessary not to mess up distance matrices, by default None
    id_ : int, optional
        Not used here just to follow API, by default 0

    Returns
    -------
    np.ndarray
        AUC score obtained for every split from repeated holdout
    """
    selection_func = ObjectsSelector(n_holdouts=n_holdouts)

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

        distances_for_train = get_rsif_distances(data, distances, id_=id_)
        if optimize_distances:
            best_distances = find_best_distances(X_train, y_train, distances, data["type"], data['name'], fold_id, clf_name, old_train_index=train_index)  # fmt: skip
            distances_for_train = get_rsif_distances(data, best_distances, id_=id_)

        if old_train_index is not None:
            # We must have correct indices for precomputed distances
            train_index = old_train_index[train_index]
            test_index = old_train_index[test_index]

        train_distances, test_distances = split_all_distances(
            distances_for_train, train_index, test_index
        )

        X_rsif = RsifData(random_state=SEED)
        X_rsif.add_data(X_train, dist=train_distances)
        X_rsif.precompute_distances(selected_objects=selected_objects)
        
        auc.append(get_rsif_auc(X_rsif, [X_test], [test_distances], y_test, clf_kwargs, clf_name, data['name'], fold_id=fold_id))

    return np.array(auc)


def experiment_rsif_mixed(
    data: dict,
    distances: List[List],
    selected_obj_ratio: float = SELECTED_OBJ_RATIO,
    clf_kwargs={},
    optimize_distances=False,
):
    """This is very similar to complex case but now data["X"] is a list of numpy arrays. Every array is a different feature.
    So wrapping everything into RsifData object is a little bit more complicated.

    Parameters
    ----------
    data : dict
        dictionary with three keys X, y, name. X key is now a list of numpy arrays
    distances : List[List]
        distances for RSIF to be used
    selected_obj_ratio : float, optional
        how many object from training split will be actually used during training phase, by default SELECTED_OBJ_RATIO
    clf_kwargs : dict, optional
        additional parameters that will be pased to clasifiers, by default {}
    optimize_distances : bool, optional
        Whether one should perform distance optimization, by default False

    Returns
    -------
    _type_
        _description_
    """
    selection_func = ObjectsSelector(n_holdouts=N_REPEATED_HOLDOUT)

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

        X_rsif = RsifData(random_state=SEED)
        test_features = []
        all_test_distances = []
        y_train, y_test = y[train_index], y[test_index]
        for f_id in range(len(data["X"])):
            feature, data_type = data["X"][f_id], data["features_types"][f_id]
            distance = distances[f_id]
            # To make sure distances are precalculated on entire data
            _ = get_rsif_distances(
                {"X": feature, "name": data["name"]}, distance, id_=f_id
            )

            X_train, X_test = feature[train_index], feature[test_index]

            if optimize_distances:
                distance = find_best_distances(X_train, y_train, distance, data_type, data['name'], fold_id, clf_name="RSIF", old_train_index=train_index, id_=f_id)  # fmt: skip

            distance = get_rsif_distances(
                {"X": feature, "name": data["name"]}, distance, id_=f_id
            )

            if isinstance(distance[0], TrainDistanceMixin):
                train_distances, test_distances = split_all_distances(distance, train_index, test_index)  # fmt: skip
            else:
                train_distances, test_distances = distance, distance

            X_rsif.add_data(X_train, dist=train_distances)
            X_rsif.precompute_distances(selected_objects=selected_objects)

            test_features.append(X_test)
            all_test_distances.append(test_distances)


        auc.append(get_rsif_auc(X_rsif, test_features, all_test_distances, y_test, clf_kwargs, clf_name="RSIF", data_name=data['name'], fold_id=f_id))  # fmt: skip

    return np.array(auc)
