import itertools
import numpy as np
from pqdm.processes import pqdm
from itertools import permutations
from abc import ABC, abstractmethod


class DistanceMixin(ABC):
    def project(self, id_x, id_p, id_q):
        return self.distance_matrix[id_p, id_x] - self.distance_matrix[id_q, id_x]

    @abstractmethod
    def precompute_distances(self,):
        pass


class TrainDistanceMixin(DistanceMixin):
    def __init__(self, distance: callable) -> None:
        self.distance = distance
        self.used_points = set()

    def get_all_objects_ids_with_selected_at_front(self, m, selected_objects):
        """Puts selected object indices at the front and rest at the back.
        Such order is important if we want to iterate from i+1 in the second loop.
        With such an order we won't skip any pair.

        Args:
            m (int): number of all objects
            selected_objects (list): indices of selected object

        Returns:
            list: array with selected indices at the front.
        """
        not_selected_objects = list(
            set(list(range(m))) - set(selected_objects))
        all_objects = selected_objects + not_selected_objects

        return all_objects

    def precompute_distances(self, X, selected_objects=None):
        """
        selected_objects. If you want to calculate distances between only N 
        nodes and all the rest pass their indices as this argument
        e.g if you pass [3,10,12] having 20 graphs in dataset distance 
        matrix will be filled only for 3 rows

        selected_objects - indices of objects that can constitute a pair
        """
        m = len(X)  # number of all objects
        if selected_objects is None:  # ! all objects are "selected" if nothing given
            selected_objects = list(range(0, m))

        # number of objects that can constitute a pair
        n = len(selected_objects)

        # this matrix is symmetrix. We will only as many rows as n
        #! Not sure if this should be n by m to have only selected objects at rows positions
        #! this would introduce problems with indexing
        self.distance_matrix = np.zeros((m, m))
        all_objects = self.get_all_objects_ids_with_selected_at_front(
            m, selected_objects)

        # #! making parallelization
        # indices = np.argwhere(np.tri(n, M=m, k=-1) == 0)
        # print(indices)
        # eventually use pqdm to have it everything nicely done in parallel

        for i in range(n):  # Iterations over objects that can constitute a pair
            # Iterations over all object so that there are no overlapping calculations
            for j in range(i+1, m):
                # i and first_obj corresponds to one another only if selected_objects is None
                first_obj_idx = selected_objects[i]
                second_obj_idx = all_objects[j]
                # print(first_obj_idx, second_obj_idx)
                objp, objq = X[first_obj_idx], X[second_obj_idx]
                distance = self.distance(objp, objq)

                self.distance_matrix[first_obj_idx, second_obj_idx] = distance
                self.distance_matrix[second_obj_idx, first_obj_idx] = distance


class TestDistanceMixin(DistanceMixin):
    # This is for pytest so it doesn't try to import this as a test
    __test__ = False

    def __init__(self, train_distance_mixin: TrainDistanceMixin):
        #! Actually if I pass different distance object I do not need offset indices will match anyway!
        self.distance = train_distance_mixin.distance
        self.train_points_to_use = train_distance_mixin.used_points

    def precompute_distances(self, X_train: np.ndarray, X_test: np.ndarray):
        # ! We create bigger array than needed but this is still fine.
        self.distance_matrix = np.zeros((X_train.shape[0], X_test.shape[0]))
        for used_point in self.train_points_to_use:
            for i, test_obj in enumerate(X_test):
                self.distance_matrix[used_point][i] = self.distance(
                    X_train[used_point], test_obj)
