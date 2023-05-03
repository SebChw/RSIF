import pickle
from typing import Callable

import numpy as np
import pandas as pd

from risf.distance import DistanceMixin, TrainDistanceMixin


def list_to_numpy(transformed):
    data = np.empty(len(transformed), dtype=object)
    data[:] = transformed
    return data


SUPPORTED_TYPES = ((np.ndarray, lambda x: x), (list,
                                            list_to_numpy), (pd.Series, lambda x: x.to_numpy()))


class RisfData(list):

    @classmethod
    def validate_column(cls, X):
        for dtype_, transform in SUPPORTED_TYPES:
            if isinstance(X, dtype_):
                return transform(X)

        raise TypeError(
            f"If you don't provide data_transform function, given data must be an instance of {[x[0] for x in SUPPORTED_TYPES]}")

    @staticmethod
    def distance_check(X, dist):
        if isinstance(dist, DistanceMixin):
            dist = dist.distance_func

        Oi, Oj = X[0], X[1]
        try:
            dist(Oi, Oj)
        except Exception as e:  # In that case thiss really can be any kind of exception
            raise ValueError(
                "Cannot' calculate distance between two instances of a given column!") from e

    @staticmethod
    def calculate_data_transform(X, data_transform):
        if data_transform is not None:
            try:
                X = [data_transform(x) for x in X]
            except Exception as e:
                raise ValueError("Cannot' calculate data transform!") from e

        return X

    def __init__(self, num_of_selected_objects: int = None, random_state=23):
        self.distances = []
        self.names = []
        self.transforms = []
        self.shape = None
        self.num_of_selected_objects = num_of_selected_objects

        self.random_gen = np.random.RandomState(random_state)

    def update_metadata(self, data_transform, name):
        self.transforms.append(data_transform)
        self.names.append(name if name is not None else f"attr{len(self)}")

    def shape_check(self, X):
        if self.shape is None:
            self.shape = (X.shape[0], 1)
        else:
            n_objects, n_columns = self.shape
            if n_objects != X.shape[0]:
                raise ValueError(
                    "You newly added column must have same number of object as previous ones")
            self.shape = (n_objects, n_columns+1)

    def add_distances(self, X, distances):
        distances_parsed = []

        for dist in distances:
            if isinstance(dist, str):
                with open(dist, 'rb') as f:
                    dist = pickle.load(f)

            self.distance_check(X, dist)

            if not isinstance(dist, DistanceMixin):
                dist = TrainDistanceMixin(dist)

            distances_parsed.append(dist)

        self.distances.append(distances_parsed)

    def add_data(self, X, dist: list, data_transform: Callable = None, name=None):
        if not isinstance(dist, list):
            dist = [dist]

        X = self.calculate_data_transform(X, data_transform)
        X = self.validate_column(X)
        self.shape_check(X)
        self.add_distances(X, dist)

        super().append(X)

        self.update_metadata(data_transform, name)

    def precompute_distances(self, n_jobs=1, train_data : list = None):
        for i in range(len(self)):
            data, distances = self[i], self.distances[i]
            for distance in distances:
                if train_data is None:
                    if self.num_of_selected_objects is not None:
                        distance.selected_objects = self.random_gen.choice(len(self[i]),
                                                    self.num_of_selected_objects,
                                                    replace=False) # ! Use property here
                    distance.precompute_distances(X=data, X_test=None, n_jobs=n_jobs)
                else:
                    distance.precompute_distances(X=train_data[i], X_test=data, n_jobs=n_jobs)

                self.impute_missing_values(distance)

    #This function should be inside distance, but we have a lot of pickle of older version so that's why it's here
    def impute_missing_values(self, distance, strategy="max"):
        where_nan = np.isnan(distance.distance_matrix)
        distance.distance_matrix[where_nan] = np.nanmax(distance.distance_matrix)
