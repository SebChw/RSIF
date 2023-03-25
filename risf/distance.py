import numpy as np
from abc import ABC, abstractmethod
from joblib import Parallel, delayed, cpu_count
import math


class DistanceMixin(ABC):
    def __init__(self, distance: callable, selected_objects=None) -> None:
        self.selected_objects = selected_objects
        self.distance = distance

    def project(self, id_x, id_p, id_q):
        return self.distance_matrix[id_p, id_x] - self.distance_matrix[id_q, id_x]

    def _generate_indices_splits(self, pairs_of_indices, n_jobs):
        n_jobs = n_jobs if n_jobs > 0 else (cpu_count() + 1 + n_jobs)
        split_size = math.ceil(len(pairs_of_indices) / n_jobs)
        return [(i*split_size, (i+1)*split_size) for i in range(n_jobs)]

    def precompute_distances(self, X, X_test=None, n_jobs=1, prefer=None):
        """
        pair_selected_objects. If you want to calculate distances between only N 
        nodes and all the rest pass their indices as this argument
        e.g if you pass [3,10,12] having 20 graphs in dataset distance 
        matrix will be filled only for 3 rows

        selected_objects - indices of objects that can constitute a pair
        """
        num_train_objects = len(X)

        if X_test is None:
            num_test_objects = num_train_objects
            X_test = X
        else:
            num_test_objects = len(X_test)

        pairs_of_indices = self._generate_indices(
            num_train_objects, num_test_objects)

        splits_intervals = self._generate_indices_splits(
            pairs_of_indices, n_jobs)

        distances = Parallel(n_jobs=n_jobs, prefer=prefer)(delayed(_parallel_on_array)(
            pairs_of_indices[split_beg:split_end], X, X_test, self.distance) for split_beg, split_end in splits_intervals)

        self.distance_matrix = np.zeros((num_train_objects, num_test_objects))

        self._assign_to_distance_matrix(
            pairs_of_indices[:, 0], pairs_of_indices[:, 1], np.concatenate(distances))

    @abstractmethod
    def _generate_indices(self, num_train_objects, num_test_objects):
        pass

    @abstractmethod
    def _assign_to_distance_matrix(self, rows_id, cols_id, concatenated_distances):
        pass


def _parallel_on_array(indices, X1, X2, function):
    return [function(X1[i], X2[j]) for i, j in indices]


class TrainDistanceMixin(DistanceMixin):
    def _generate_indices(self, num_train_objects, num_test_objects=None):
        if self.selected_objects is None:
            self.selected_objects = np.arange(num_train_objects)

        indices = np.zeros((num_train_objects, num_train_objects), dtype=bool)
        indices[self.selected_objects] = 1

        # Mask some indices so that we don't calculate same distance twice
        # !in case of dot product the disstance of vector with itself will be 0 (but this improves performance by 2%), i vs i+1
        for i, s_o in enumerate(self.selected_objects):
            indices[s_o, self.selected_objects[:i+1]] = 0

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


class OnTheFlyDistanceMixin():
    """This is rather a POC and a skeleton than some serious implementation"""

    def __init__(self, distance: callable, memorize=False):
        self.distance = distance
        self.memorize = memorize

        self.project = self.project_factory()

    def project_factory(self):
        # Also custom project may be returned for dot product?
        def project(id_x, id_p, id_q):
            return self.distance(self.X[id_x], self.X_test[id_p]) - self.distance(self.X[id_x], self.X_test[id_p])

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
