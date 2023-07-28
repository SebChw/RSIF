from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import sklearn.utils.validation as sklearn_validation
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, OutlierMixin

import risf.utils.measures as measures
from risf.distance import SelectiveDistance, TrainDistanceMixin
from risf.distance_functions import euclidean_projection
from risf.risf_data import RisfData
from risf.tree import RandomIsolationSimilarityTree
from risf.utils.validation import (
    check_distance,
    check_max_samples,
    check_random_state,
    prepare_X,
)


class RandomIsolationSimilarityForest(BaseEstimator, OutlierMixin):
    """
    An algorithm for outlier detection based on ideas from Isolation Forest
    and Random Similarity Forest.It borrows the idea of isolating data-points
    by performing random splits, as in Isolation Forest,but they are not
    performed on features, but on the projections on the line created by randomly sampled objects
    as in Random Similarity Forest.
    """

    def __init__(
        self,
        distances: List[List[Union[SelectiveDistance, TrainDistanceMixin]]] = [
            [SelectiveDistance(euclidean_projection, 1, 1)]
        ],
        n_estimators: int = 100,
        max_samples: Union[int, float, str] = "auto",
        contamination: Union[str, float] = "auto",
        bootstrap: bool = False,
        n_jobs: Optional[int] = None,
        random_state: int = 23,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        distances : List[Union[SelectiveDistance, TrainDistanceMixin]], optional
            For every attribute you should provide a list of distance functions, by default [ SelectiveDistance(euclidean_projection, 1, 1) ].
            Unfortunately you can't mix types if you want to use euclidean distance and DTW for time series you need to add two attributes.
        n_estimators : int, optional
            number of trees in forest, by default 100
        max_samples : Union[int, float, str], optional
            number of samples used to fit every forest. If float it's n_samples*max_samples. If auto its 256, by default "auto"
        contamination : Union[str, float], optional
            percentage of samples considered to be outliers in the data, by default "auto"
        max_depth : int, optional
            maximum depth of every tree, by default 8
        bootstrap : bool, optional
            Whether to sample data with or withouth replacement, by default False
        n_jobs : Optional[int], optional
            number of threads used durign fit phase, by default None
        random_state : int, optional
            seed used to create random instance, by default 23
        verbose : bool, optional
            to print additional information during fit, by default False
        """
        self.distances = distances
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def fit(
        self, X: Union[np.ndarray, RisfData], y: Optional[np.ndarray] = None
    ) -> RandomIsolationSimilarityForest:
        """fit forest of Generalized Isolation Trees to data X. If you pass y it will be used to set contamination

        Parameters
        ----------
        X : Union[np.ndarray, RisfData]
            data to fit
        y : Optional[np.ndarray], optional
            labels, by default None

        Returns
        -------
        RandomIsolationSimilarityForest
            fitted forest
        """
        self.prepare_to_fit(X)
        self.trees_ = self.create_trees()

        #! In case when we have this gigantic distance matrix sharedmem is even faster + we don't get memory errors.
        self.trees_ = Parallel(n_jobs=self.n_jobs, require="sharedmem")(
            delayed(_build_tree)(
                tree,
                self.X,
                i,
                self.n_estimators,
                self.subsample_size,
                verbose=self.verbose,
            )
            for i, tree in enumerate(self.trees_)
        )

        self.set_offset(y)

        return self

    def prepare_to_fit(self, X: Union[np.ndarray, RisfData]):
        """Steps done here:
        1. parsing X to obtain numpy array and features_span (indices of particular features in X)
        2. Creating random instance
        3. Calculating subsample_size (size of bootstraped dataset used to fit every tree)
        4. Calculating max depth
        4. Validating distances

        Parameters
        ----------
        X : Union[np.ndarray, RisfData]
            data to fit
        """
        self.X, self.features_span = prepare_X(X)
        self.random_state = check_random_state(self.random_state)
        self.subsample_size = check_max_samples(self.max_samples, self.X)
        self.max_depth = np.ceil(np.log2(self.subsample_size))
        self.distances = check_distance(self.distances, len(self.features_span))

    def create_trees(self) -> List[RandomIsolationSimilarityTree]:
        """Creates trees used in forest.

        Notes
        -----
        It is very important that every tree has different random_state, otherwise they will be identical
        """
        return [
            RandomIsolationSimilarityTree(
                distances=self.distances,
                max_depth=self.max_depth,
                random_state=i,
                features_span=self.features_span,
            )
            for i in range(self.n_estimators)
        ]

    def set_offset(self, y: Optional[np.ndarray] = None):
        """set offset for decision function. If y is passed it will be used to set contamination. If auto it is set to -0.5. If float it is set
        to 100*contamination percentile of scores so that this percentile of samples will be considered outliers.

        Parameters
        ----------
        y : Optional[np.ndarray], optional
            labels, by default None
        """
        if y is not None:
            self.contamination = sum(y) / len(y)  # 0/1 = in/out lier

        if self.contamination == "auto":
            self.decision_threshold_ = -0.5
        else:
            self.decision_threshold_ = np.percentile(
                self.score_samples(self.X), 100.0 * self.contamination
            )

    def calculate_mean_path_lengths(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates mean path_length of every sample in X matrix
        Parameters
        ----------
            X : array-like of shape (n_samples, n_features)
                The input samples.
        """
        all_path_lengths = [t.path_lengths_(X) for t in self.trees_]
        return np.mean(all_path_lengths, axis=0)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Opposite of the anomaly score defined in the original paper.
        The anomaly score of an input sample is computed as
        the mean anomaly score of the trees in the forest.
        The measure of normality of an observation given a tree is the depth
        of the leaf containing this observation, which is equivalent to
        the number of splittings required to isolate this point. In case of
        several observations n_left in the leaf, the average path length of
        a n_left samples isolation tree is added.

        Parameters
        ----------
            X : array-like of shape (n_samples, n_features)
                The input samples.
        Returns
        -------
            scores : ndarray of shape (n_samples,)
                The anomaly score of the input samples.
                The lower, the more abnormal.
        """
        mean_path_lengths = self.calculate_mean_path_lengths(X)

        c = measures._average_path_length(self.subsample_size)

        scores = np.array([-(2 ** (-pl / c)) for pl in mean_path_lengths])

        return scores

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Performs whole pipeline of calculating abnormality score and then
        offsetting it so that negative values denote outliers
        """
        sklearn_validation.check_is_fitted(self)
        scores = self.score_samples(X)

        return scores - self.decision_threshold_

    def predict_train(self, return_raw_scores: bool = False) -> np.ndarray:
        """Predict if a particular sample from train set is an outlier or not"""
        return self.predict(self.X, return_raw_scores)

    def predict(
        self, X: Union[np.ndarray, RisfData], return_raw_scores: bool = False
    ) -> np.ndarray:
        """Predict if a particular sample is an outlier or not.

        Paramteres
        ----------
            X : array-like, shape (n_samples, n_features)
                The input samples.
            return_raw_scores : bool, optional
                If True, returns raw anomaly score (useful for AUC calculation), by default False
        Returns
        -------
            is_outlier : array, shape (n_samples,) For each observation, tell
            whether or not (1 or 0) it should be considered as an outlier according to the fitted model.
            If return_raw_scores is True, returns anomaly scores instead (the lower, the more abnormal).
        """
        if isinstance(X, RisfData):
            for tree in self.trees_:
                tree.set_test_distances(X.distances)

        X, features_span = prepare_X(X)
        decision_function = self.decision_function(X)

        if return_raw_scores:
            return decision_function

        is_outlier = np.zeros(X.shape[0], dtype=int)
        is_outlier[decision_function < 0] = 1
        return is_outlier

    def get_used_points(self) -> set:
        """Returns set of points used to fit forest

        Returns
        -------
        set
            indices of points used to fit forest. It is used to calculate distances to test points
        """
        used_points = set()
        for tree in self.trees_:
            used_points.update(tree.get_used_points())

        return used_points


def _build_tree(
    tree: RandomIsolationSimilarityTree,
    X: np.ndarray,
    tree_idx: int,
    n_trees: int,
    subsample_size: int = 256,
    verbose: int = 0,
    bootstrap: bool = False,
):
    """Performs subsampling of data X and feeds a tree with subsampled data"""
    if verbose > 1:
        print("building tree %d of %d" % (tree_idx + 1, n_trees))

    examples = tree.random_state.choice(
        X.shape[0], size=subsample_size, replace=bootstrap
    )
    tree.fit(X[examples])
    return tree
