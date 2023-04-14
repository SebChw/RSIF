from risf.distance import TrainDistanceMixin, DistanceMixin
import numpy as np
import pandas as pd


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

    def __init__(self,):
        self.distances = []
        self.names = []
        self.transforms = []
        self.shape = None

    def update_metadata(self, dist, data_transform, name):
        self.transforms.append(data_transform)
        self.names.append(name if name is not None else f"attr{len(self)}")

        # I can give DistanceMixing that already knows everything
        # !I wonder if we should use duck typing instead
        if isinstance(dist, DistanceMixin):
            self.distances.append(dist)
        else:  # Or function that is wrapped into DistanceMixin
            self.distances.append(TrainDistanceMixin(dist))

    def shape_check(self, X):
        if self.shape is None:
            self.shape = (X.shape[0], 1)
        else:
            n_objects, n_columns = self.shape
            if n_objects != X.shape[0]:
                raise ValueError(
                    "You newly added column must have same number of object as previous ones")
            self.shape = (n_objects, n_columns+1)

    def add_data(self, X, dist: callable, data_transform: callable = None, name=None):
        X = self.calculate_data_transform(X, data_transform)
        X = self.validate_column(X)
        self.shape_check(X)
        self.distance_check(X, dist)

        super().append(X)

        self.update_metadata(dist, data_transform, name)

    def precompute_distances(self, n_jobs=1):
        for data, distance in zip(self, self.distances):
            distance.precompute_distances(data, n_jobs=n_jobs)
