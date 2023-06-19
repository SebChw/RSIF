from __future__ import annotations

import pickle
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd

from risf.distance import (
    DistanceMixin,
    SelectiveDistance,
    TestDistanceMixin,
    TrainDistanceMixin,
)


def list_to_numpy(transformed):
    data = np.empty(len(transformed), dtype=object)
    data[:] = transformed
    return data


SUPPORTED_TYPES = (
    (np.ndarray, lambda x: x),
    (list, list_to_numpy),
    (pd.Series, lambda x: x.to_numpy()),
)


class RisfData(list):
    """Class used to store data for RISF algorithm. It's basically a list of columns with some additional metadata"""

    @classmethod
    def validate_column(cls, X: Union[np.ndarray, list, pd.Series]) -> np.ndarray:
        """Just check if we can work with given data type.

        Parameters
        ----------
        X : Union[np.ndarray, list, pd.Series]
            data to be validated

        Returns
        -------
        np.ndarray
            validated data as np array
        Raises
        ------
        TypeError
            if given data is not an instance of any of supported types
        """
        for dtype_, transform in SUPPORTED_TYPES:
            if isinstance(X, dtype_):
                return transform(X)

        raise TypeError(
            f"Given data must be an instance of {[x[0] for x in SUPPORTED_TYPES]}"
        )

    @staticmethod
    def distance_check(X: np.ndarray, dist: Union[DistanceMixin, SelectiveDistance]):
        """Check whether distance can be calculated between two instances of a given column.

        Parameters
        ----------
        X : np.ndarray
            column
        dist : Union[DistanceMixin, SelectiveDistance]
            distance that will be checked

        Raises
        ------
        ValueError
            if distance cannot be calculated
        """
        if isinstance(dist, DistanceMixin):
            dist = dist.distance_func
        elif isinstance(dist, SelectiveDistance):
            dist = dist.projection_func

        Oi, Oj = X[0], X[1]
        try:
            dist(Oi, Oj)
        except Exception as e:  # In that case thiss really can be any kind of exception
            raise ValueError(
                "Cannot' calculate distance between two instances of a given column!"
            ) from e

    def __init__(self, num_of_selected_objects: int = None, random_state=23):
        # TODO: I wonder if we should pass num_of_selected_objects here or user should take care about it. It creates another layer of complexity
        self.distances = []
        self.names = []
        self.shape = None
        self.num_of_selected_objects = num_of_selected_objects

        self.random_gen = np.random.RandomState(random_state)

    def update_metadata(self, name: Optional[str] = None):
        """Updates metadata of a given column. For now it's just a column name

        Parameters
        ----------
        name : str
            name of the column
        """
        self.names.append(name if name is not None else f"attr{len(self)}")

    def shape_check(self, X: np.ndarray):
        """Check if given column has the same number of objects as previous ones.

        Parameters
        ----------
        X : np.ndarray

        Raises
        ------
        ValueError
            Raised if number of objects doesn't match
        """
        if self.shape is None:
            self.shape = (X.shape[0], 1)
        else:
            n_objects, n_columns = self.shape
            if n_objects != X.shape[0]:
                raise ValueError(
                    "You newly added column must have same number of object as previous ones"
                )
            self.shape = (n_objects, n_columns + 1)

    def add_distances(
        self,
        X: np.ndarray,
        distances: List[Union[TrainDistanceMixin, SelectiveDistance, str]],
    ):
        """Validate and add distances to a given column. If distance is a string then it will be loaded from a file.

        REMEMBER THAT YOU CAN'T MIX DISTANCE MIXIN AND SELECTIVE DISTANCE IN ONE COLUMN

        Parameters
        ----------
        X : np.ndarray
        distances : Union[TrainDistanceMixin, SelectiveDistance, str]
        """
        distances_parsed = []

        for dist in distances:
            if isinstance(dist, str):
                with open(dist, "rb") as f:
                    dist = pickle.load(f)

            self.distance_check(X, dist)

            distances_parsed.append(dist)

        self.distances.append(distances_parsed)

    def add_data(
        self,
        X: np.ndarray,
        dist: List[Union[SelectiveDistance, DistanceMixin, str]],
        name: Optional[str] = None,
    ):
        """Add new column to the data. Firsly performing all necessary checks"""
        if not isinstance(dist, list):
            dist = [dist]

        X = self.validate_column(X)
        self.shape_check(X)
        self.add_distances(X, dist)

        super().append(X)

        self.update_metadata(name)

    def precompute_distances(
        self,
        n_jobs=1,
        train_data: Optional[RisfData] = None,
        selected_objects: Optional[np.ndarray] = None,
    ):
        """This function is propably too complex. It precomputes all distances between objects in all columns.
        The work of it differst between training and testing phase. In training phase we need to precompute all distances

        Parameters
        ----------
        n_jobs : int, optional
            , by default 1
        train_data : Optional[RisfData], optional
            if this is test set you must provide train_data, by default None
        selected_objects : Optional[np.ndarray], optional
            objects that were selected during training phase, by default None
        """
        for i in range(len(self)):
            data, distances = self[i], self.distances[i]
            for distance in distances:
                if train_data is None:
                    if selected_objects is not None:
                        distance.selected_objects = selected_objects
                    elif self.num_of_selected_objects is not None:
                        distance.selected_objects = self.random_gen.choice(
                            len(self[i]), self.num_of_selected_objects, replace=False
                        )  # ! Use property here

                    distance.precompute_distances(X=data, X_test=None, n_jobs=n_jobs)
                else:
                    distance.precompute_distances(
                        X=train_data[i], X_test=data, n_jobs=n_jobs
                    )

                self.impute_missing_values(distance)

    def impute_missing_values(self, distance, strategy="max"):
        where_nan = np.isnan(distance.distance_matrix)
        distance.distance_matrix[where_nan] = np.nanmax(distance.distance_matrix)

    def transform(
        self,
        list_of_X: List[np.ndarray],
        forest,
        n_jobs=1,
        precomputed_distances: Optional[List[TestDistanceMixin]] = None,
    ) -> RisfData:
        """This function is probably also too complex. It performs transformation of a given test data to Risf Data based on train data.

        Parameters
        ----------
        list_of_X : List[np.ndarray]
            List of features, it must match those in the training set
        forest : RandomIsolationSimilarityForest
            Forest used in the training phase
        n_jobs : int, optional
            _description_, by default 1
        precomputed_distances : Optional[List[TrainDistanceMixin]], optional
            If you have already precomputed distances for test set use it here, by default None

        Returns
        -------
        RisfData
            Parsed RisfData with all distances precomputed
        """
        test_data = RisfData()
        for i, X in enumerate(list_of_X):
            test_distances_of_attribute = []

            if precomputed_distances is None:
                for distance in self.distances[i]:
                    test_distance = TestDistanceMixin(
                        distance.distance_func, list(forest.get_used_points())
                    )

                    test_distances_of_attribute.append(test_distance)
            else:
                test_distances_of_attribute = precomputed_distances[i]

            test_data.add_data(X, test_distances_of_attribute, self.names[i])

        test_data.precompute_distances(train_data=self, n_jobs=n_jobs)

        return test_data
