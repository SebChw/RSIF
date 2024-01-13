import math
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple

import numpy as np
from joblib import Parallel, cpu_count, delayed


class SelectiveDistance:
    """The goal of this distance is to take a np array select some objects and calculate distance only for them"""

    def __init__(self, projection_func: Callable, min_n: int, max_n: int) -> None:
        """
        Parameters
        ----------
        projection_func : Callable
            function used to perform projections
        min_n : int
            minimum number of features to be selected
        max_n : int
            maximum number of features to be selected
        """
        self.projection_func = projection_func
        self.min_n = min_n
        self.max_n = max_n

    def project(
        self,
        X: np.ndarray,
        Op: np.ndarray,
        Oq: np.ndarray,
        tree,  # can't use typing due to circural imports
        random_instance: np.random.RandomState,
    ) -> np.ndarray:
        """performs projection on selected features

        Parameters
        ----------
        X : np.ndarray
            Data of one particular feature
        Op : np.ndarray
            First object to create projection plane
        Oq : np.ndarray
            Second object that creates projection plane
        tree : RandomSimilarityIsolationTree
            Tree to which projection parameters are saved
        random_instance : np.random.RandomState
            random instance for reproducibility

        Returns
        -------
        np.ndarray
            projection
        """
        if not hasattr(tree, "selected_features"):
            n_selected = random_instance.randint(self.min_n, self.max_n + 1)
            selected_features = random_instance.choice(
                X.shape[1], n_selected, replace=False
            )
            tree.selected_features = selected_features

            similarity = -np.matmul(
                X[:, selected_features], X[:, selected_features].T, dtype=np.float64
            )
            best_pair = np.unravel_index(np.argmax(similarity), similarity.shape)

            Op = X[best_pair[0]]
            Oq = X[best_pair[1]]
            tree.Oi = Op
            tree.Oj = Oq
        else:
            selected_features = tree.selected_features

        return self.projection_func(
            X[:, selected_features],
            Op[selected_features],
            Oq[selected_features],
        )


class DistanceMixin(ABC):
    """Abstract class that defines interface that distance functions based on similarity matrices must use"""

    def __init__(
        self, distance: Callable, selected_objects: Optional[np.ndarray] = None
    ) -> None:
        self.selected_objects = selected_objects
        self.distance_func = distance

        self.precomputed = False

    # TODO: This change is to be added. For now as we have only one precoumpted matrix which is fully precomputed
    # @property
    # def selected_objects(self):
    #     return self._selected_objects

    # @selected_objects.setter
    # def selected_objects(self, value):
    #     if self.precomputed:
    #         print("Distance is already precomputed selected objects won't be changed")
    #     else:
    #         self._selected_objects = value

    def project(
        self,
        id_x: np.ndarray,
        id_p: np.ndarray,
        id_q: np.ndarray,
        tree=None,
        random_instance=None,
    ) -> np.ndarray:
        """Projects simply by reading values from distance matrix

        Parameters
        ----------
        id_x : np.ndarray
            indices of objects to be projected
        id_p : int
            index of first object to create projection plane
        id_q : int
            index of second object to create projection plane
        tree : _type_, optional
            Not used here, by default None
        random_instance : _type_, optional
            Not used here, by default None

        Returns
        -------
        np.ndarray
            projection values
        """
        if id_x.dtype == np.float64:  # workaround for now
            id_x = id_x.astype(np.int32)
            id_p = id_p.astype(np.int32)
            id_q = id_q.astype(np.int32)

        return self.distance_matrix[id_p, id_x] - self.distance_matrix[id_q, id_x]

    def _generate_indices_splits(
        self, pairs_of_indices: np.ndarray, n_jobs: int
    ) -> List[Tuple[int, int]]:
        """Divides indices into splits for parallelization

        Parameters
        ----------
        pairs_of_indices : np.ndarray
            pairss of objects between which we have to compute distance
        n_jobs : int
            number of processes that will be used

        Returns
        -------
        List[Tuple[int, int]]
            each tuple is a first and last index of split for every process
        """
        n_jobs = n_jobs if n_jobs > 0 else (cpu_count() + 1 + n_jobs)
        split_size = math.ceil(len(pairs_of_indices) / n_jobs)
        return [(i * split_size, (i + 1) * split_size) for i in range(n_jobs)]

    def precompute_distances(
        self,
        X: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        n_jobs: int = 1,
        prefer: Optional[str] = None,
    ):
        """Creates distance matrix for given data

        Parameters
        ----------
        X : np.ndarray
            Array of object between which we will calculate distance
        X_test : Optional[np.ndarray], optional
            For test set we must compute distance between members of train set (X) and test set (X_test), by default None
        n_jobs : int, optional
            number of processes to use, by default 1
        prefer : Optional[str], optional
            parameter passed to joblib parallel, by default None
        """
        if self.precomputed:
            return

        num_train_objects = len(X)

        if X_test is None:
            num_test_objects = num_train_objects
            X_test = X
        else:
            num_test_objects = len(X_test)

        pairs_of_indices = self._generate_indices(num_train_objects, num_test_objects)

        splits_intervals = self._generate_indices_splits(pairs_of_indices, n_jobs)

        distances = Parallel(n_jobs=n_jobs, prefer=prefer, require="sharedmem")(
            delayed(_parallel_on_array)(
                pairs_of_indices[split_beg:split_end], X, X_test, self.distance_func
            )
            for split_beg, split_end in splits_intervals
        )

        self.distance_matrix = np.zeros((num_train_objects, num_test_objects))

        self._assign_to_distance_matrix(
            pairs_of_indices[:, 0], pairs_of_indices[:, 1], np.concatenate(distances)
        )

        self.precomputed = True

    @abstractmethod
    def _generate_indices(
        self, num_train_objects: int, num_test_objects: int
    ) -> np.ndarray:
        """To allow parallelization we must generate indices of objects that we will calculate distance between.

        Parameters
        ----------
        num_train_objects : int
            number of train objects
        num_test_objects : _type_, optional
            number of test objects, by default None

        Returns
        -------
        np.ndarray
        """
        pass

    @abstractmethod
    def _assign_to_distance_matrix(
        self,
        rows_id: np.ndarray,
        cols_id: np.ndarray,
        concatenated_distances: np.ndarray,
    ):
        """After distance is calculated we have long 1d numpy array we must reformat it to distance matrix. Remember that matrix is symmetric

        Parameters
        ----------
        rows_id : np.ndarray
        cols_id : np.ndarray
        concatenated_distances : np.ndarray
        """
        pass


def _parallel_on_array(
    indices: np.ndarray, X1: np.ndarray, X2: np.ndarray, function: Callable
) -> List[float]:
    """Calculates distance between objects in X1 and X2 for given indices

    Parameters
    ----------
    indices : np.ndarray
        pairs of indices between which we will calculate distance
    X1 : np.ndarray
    X2 : np.ndarray
    function : Callable
        distance function

    Returns
    -------
    List[float]
        list of distances between objects

    Notes
    -----
    If distance function returns NaN it must be later imputed.
    """
    nan_encountered = False
    distances = []
    for i, j in indices:
        distance = function(X1[i], X2[j])
        if (not nan_encountered) and np.isnan(distance):
            print(
                "Encountered NaN during distance calculations. Imputing it later will be necessary."
            )
            nan_encountered = True
        distances.append(distance)
    return distances


class TrainDistanceMixin(DistanceMixin):
    def _generate_indices(
        self, num_train_objects: int, num_test_objects=None
    ) -> np.ndarray:
        if self.selected_objects is None:
            self.selected_objects = np.arange(num_train_objects)

        indices = np.zeros((num_train_objects, num_train_objects), dtype=bool)
        indices[self.selected_objects] = 1

        # Mask some indices so that we don't calculate same distance twice
        for i, s_o in enumerate(self.selected_objects):
            indices[s_o, self.selected_objects[: i + 1]] = 0

        return np.vstack(np.where(indices)).T

    def create_top_k_projection_pairs(self, k=10):
        k *= 2  # our matrix is symmetric we will have double as many stuff
        top_k_indices = np.argpartition(-self.distance_matrix.flatten(), k)[:k]
        top_k_pairs = np.unravel_index(top_k_indices, self.distance_matrix.shape)
        self.top_k_pairs = []
        for i, j in zip(*top_k_pairs):
            if i < j:
                self.top_k_pairs.append((i, j))

    def _assign_to_distance_matrix(
        self,
        row_ids: np.ndarray,
        col_ids: np.ndarray,
        concatenated_distances: np.ndarray,
    ):
        self.distance_matrix[row_ids, col_ids] = concatenated_distances
        self.distance_matrix[col_ids, row_ids] = concatenated_distances


class TestDistanceMixin(DistanceMixin):
    # This is for pytest so it doesn't try to import this as a test
    __test__ = False

    def __init__(self, distance: Callable, selected_objects):
        super().__init__(distance, selected_objects)

    def _generate_indices(
        self, num_train_objects: int, num_test_objects: int
    ) -> np.ndarray:
        indices = np.zeros((num_train_objects, num_test_objects))
        indices[self.selected_objects] = 1

        return np.vstack(np.where(indices)).T

    def _assign_to_distance_matrix(
        self,
        row_ids: np.ndarray,
        col_ids: np.ndarray,
        concatenated_distances: np.ndarray,
    ):
        """In this case we will only look on upper triangle of matrix as on rows we have train and column test objects"""
        self.distance_matrix[row_ids, col_ids] = concatenated_distances


def split_distance_mixin(
    distance_mixin: TrainDistanceMixin, train_indices: np.ndarray, test_indices=None
) -> Tuple[TrainDistanceMixin, TestDistanceMixin]:
    """Given distance_mixin with all distances precalculated it return train and test distance mixin object
    useful when performing cross validation

    Args:
        distance_mixin (TrainDistanceMixin): Distances precalculated for entire dataset
        train_indices (np.ndarray): Object that should be treated as train set. Test set is just remaining indices

    Returns:
        Tuple[TrainDistanceMixin, TestDistanceMixin]:
    """

    # During training we assume 0th object is at 0th index. Indices must be sorted!
    train_indices = np.sort(train_indices)
    distance_matrix = distance_mixin.distance_matrix
    distance_function = distance_mixin.distance_func

    if test_indices is None:
        test_indices = np.setdiff1d(np.arange(distance_matrix.shape[0]), train_indices)
    else:
        test_indices = np.sort(test_indices)

    train_distance_mixin = TrainDistanceMixin(
        distance_function, selected_objects=np.arange(len(train_indices))
    )
    train_distance_mixin.precomputed = True

    train_distance_matrix = distance_matrix[train_indices]
    train_distance_mixin.distance_matrix = train_distance_matrix[:, train_indices]

    test_distance_mixin = TestDistanceMixin(
        distance_function, selected_objects=np.arange(len(train_indices))
    )
    test_distance_mixin.precomputed = True

    test_distance_matrix = distance_matrix[train_indices]
    test_distance_mixin.distance_matrix = test_distance_matrix[:, test_indices]

    return train_distance_mixin, test_distance_mixin
