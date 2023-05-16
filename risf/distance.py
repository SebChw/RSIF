import math
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from joblib import Parallel, cpu_count, delayed


class DistanceMixin(ABC):
    def __init__(self, distance: callable, selected_objects=None) -> None:
        self.selected_objects = selected_objects
        self.distance_func = distance

        self.precomputed = False

    # @property
    # def selected_objects(self):
    #     return self._selected_objects
    #! This change is to be added. For now as we have only one precoumpted matrix which is fully precomputed
    #! We can neglect it.
    # @selected_objects.setter
    # def selected_objects(self, value):
    #     if self.precomputed:
    #         print("Distance is already precomputed selected objects won't be changed")
    #     else:
    #         self._selected_objects = value

    def project(self, id_x, id_p, id_q):
        return self.distance_matrix[id_p, id_x] - self.distance_matrix[id_q, id_x]

    def _generate_indices_splits(self, pairs_of_indices, n_jobs):
        n_jobs = n_jobs if n_jobs > 0 else (cpu_count() + 1 + n_jobs)
        split_size = math.ceil(len(pairs_of_indices) / n_jobs)
        return [(i * split_size, (i + 1) * split_size) for i in range(n_jobs)]

    def precompute_distances(self, X, X_test=None, n_jobs=1, prefer=None):
        """
        pair_selected_objects. If you want to calculate distances between only N
        nodes and all the rest pass their indices as this argument
        e.g if you pass [3,10,12] having 20 graphs in dataset distance
        matrix will be filled only for 3 rows

        selected_objects - indices of objects that can constitute a pair
        """
        if self.precomputed:
            print("Distances already calculated. Skipping...")
            return

        num_train_objects = len(X)

        if X_test is None:
            num_test_objects = num_train_objects
            X_test = X
        else:
            num_test_objects = len(X_test)

        pairs_of_indices = self._generate_indices(num_train_objects, num_test_objects)

        splits_intervals = self._generate_indices_splits(pairs_of_indices, n_jobs)

        distances = Parallel(n_jobs=n_jobs, prefer=prefer)(
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
    def _generate_indices(self, num_train_objects, num_test_objects):
        pass

    @abstractmethod
    def _assign_to_distance_matrix(self, rows_id, cols_id, concatenated_distances):
        pass


def _parallel_on_array(indices, X1, X2, function):
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
    def _generate_indices(self, num_train_objects, num_test_objects=None):
        if self.selected_objects is None:
            self.selected_objects = np.arange(num_train_objects)

        indices = np.zeros((num_train_objects, num_train_objects), dtype=bool)
        indices[self.selected_objects] = 1

        # Mask some indices so that we don't calculate same distance twice
        # !in case of dot product the disstance of vector with itself will be 0 (but this improves performance by 2%), i vs i+1
        for i, s_o in enumerate(self.selected_objects):
            indices[s_o, self.selected_objects[: i + 1]] = 0

        return np.vstack(np.where(indices)).T

    def _assign_to_distance_matrix(self, row_ids, col_ids, concatenated_distances):
        self.distance_matrix[row_ids, col_ids] = concatenated_distances
        self.distance_matrix[col_ids, row_ids] = concatenated_distances


class TestDistanceMixin(DistanceMixin):
    # This is for pytest so it doesn't try to import this as a test
    __test__ = False

    def __init__(self, distance: callable, selected_objects):
        super().__init__(distance, selected_objects)

    def _generate_indices(self, num_train_objects, num_test_objects):
        indices = np.zeros((num_train_objects, num_test_objects))
        indices[self.selected_objects] = 1

        return np.vstack(np.where(indices)).T

    def _assign_to_distance_matrix(self, row_ids, col_ids, concatenated_distances):
        self.distance_matrix[row_ids, col_ids] = concatenated_distances


def split_distance_mixin(
    distance_mixin: TrainDistanceMixin, train_indices: np.ndarray
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
    test_indices = np.setdiff1d(np.arange(distance_matrix.shape[0]), train_indices)

    train_distance_mixin = TrainDistanceMixin(
        distance_function, selected_objects=train_indices
    )
    train_distance_mixin.precomputed = True

    train_distance_matrix = distance_matrix.copy()
    train_distance_matrix = distance_matrix[train_indices]
    train_distance_mixin.distance_matrix = train_distance_matrix[:, train_indices]

    test_distance_mixin = TestDistanceMixin(
        distance_function, selected_objects=test_indices
    )
    test_distance_mixin.precomputed = True

    test_distance_matrix = distance_matrix.copy()
    test_distance_matrix = distance_matrix[train_indices]
    test_distance_mixin.distance_matrix = test_distance_matrix[:, test_indices]

    return train_distance_mixin, test_distance_mixin


class OnTheFlyDistanceMixin:
    """This is rather a POC and a skeleton than some serious implementation"""

    def __init__(self, distance: callable, memorize=False):
        self.distance_func = distance
        self.memorize = memorize

        self.project = self.project_factory()

    def project_factory(self):
        # Also custom project may be returned for dot product?
        def project(id_x, id_p, id_q):
            return self.distance_func(
                self.X[id_x], self.X_test[id_p]
            ) - self.distance_func(self.X[id_x], self.X_test[id_p])

        def project_memorize(id_x, id_p, id_q):
            # TODO: allow memoization
            pass

        if self.memorize:
            return project_memorize
        return project

    def precompute_distances(self, X, X_test=None):
        self.X = X
        if X_test is None:
            self.X_test = X
        else:
            self.X_test = X_test
