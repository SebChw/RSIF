from sklearn.base import BaseEstimator, OutlierMixin
import numpy as np
from joblib import Parallel, delayed

from risf.tree import RandomIsolationSimilarityTree

import risf.utils.measures as measures
import sklearn.utils.validation as sklearn_validation
from risf.utils.validation import (prepare_X,
                                   check_random_state,
                                   check_max_samples)


class RandomIsolationSimilarityForest(BaseEstimator, OutlierMixin):
    """An algorithm for outlier detection based on ideas from Isolation Forest
    and Random Similarity Forest.It borrows the idea of isolating data-points
    by performing random splits, as in Isolation Forest,but they are not
    performed on features, but in the same way as in Random Similarity Forest.
    Parameters
    ----------
        random_state : int optional (default=1)
            If int, random_state is the seed used by the
                    random number generator;
        n_estimators : integer, optional (default=100)
            The number of trees in the forest.
        distance : str, risf.distance.Distance object or
                    list of them (default='euclidean')
            If str or risf.distance.Distance all features uses the same func
            if list then each feature is assosiated with corresponding distance
        max_depth : integer or None, optional (default=None)
            The maximum depth of the tree. If None, then nodes are
            expanded until all leaves are pure.
        max_samples : int or float
            size of subsamples used for fitting trees, if int then use number
            of objects provided, if float then use fraction of whole sample
        contamination : string or float (default='auto'), fraction of expected
        outliers in the data. If auto, then use algorithm criterion described in
        Isolation Forest paper. Float means fraction of objects that
        should be considered outliers.

        Attributes
        ----------
            trees_ : list of SimilarityTreeClassifiers
                The collection of fitted sub-estimators.
        Notes
        -----
            To obtain a deterministic behaviour during
            fitting, ``random_state`` has to be fixed.
    """

    def __init__(
        self,
        distance="euclidean",
        n_estimators=100,
        max_samples="auto",
        contamination="auto",
        max_features=1.0,
        max_depth=8,
        bootstrap=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
    ):
        self.distance = distance
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.max_depth = max_depth
        self.bootsrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X: np.array, y=None):
        """Build a forest of trees from the training set X.
        Parameters
        ----------
            X : array-like matrix of shape = [n_samples, n_features]
                Every feature must be either numeric value or integer
                index to the object.
            y : None, added to follow Scikit-Learn convention
        Returns
        -------
            self : object.
        """
        self.X = prepare_X(X)
        # This will be a random instance now and the same will be passed to every tree and subtree
        self.random_state = check_random_state(self.random_state)
        self.subsample_size = check_max_samples(self.max_samples, self.X)
        if y is not None:
            self.contamination = sum(y)/len(y) #0/1 = in/out lier

        self.trees_ = [
            RandomIsolationSimilarityTree(
                distance=self.distance,
                max_depth=self.max_depth,
                random_state=self.random_state,
            )
            for i in range(self.n_estimators)
        ]

        self.trees_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_build_tree)(
                tree, self.X, i, self.n_estimators, self.subsample_size, verbose=self.verbose
            )
            for i, tree in enumerate(self.trees_)
        )

        self.set_offset()

        return self

    def set_offset(self):
        """sets offset based on contamination setting"""
        if self.contamination == "auto":
            self.offset_ = -0.5

        else:
            self.offset_ = np.percentile(
                self.score_samples(self.X), 100.0 * self.contamination)

    def calculate_mean_path_lengths(self, X: np.array):
        """
        Calculates mean path_length of every sample in X matrix
        Parameters
        ----------
            X : array-like of shape (n_samples, n_features)
                The input samples.
        """
        all_path_lengths = [t.path_lengths_(X) for t in self.trees_]
        return np.mean(all_path_lengths, axis=0)

    def score_samples(self, X: np.array):
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
        # Average depth at which a sample lies over all trees
        mean_path_lengths = self.calculate_mean_path_lengths(X)

        # Depths are normalized in the same fashion as in Isolation Forest
        c = measures._average_path_length(self.subsample_size)

        scores = np.array([-(2 ** (-pl / c)) for pl in mean_path_lengths])

        return scores

    def _validate_X_predict(self, X: np.array):
        """
        Takes first tree and checks if all distances can be calculated,
        not sure how to implement this. Making these projections h
        """
        # tree = self.trees_[0]
        # try:
        #     for feature_idx, distance_type in enumerate(tree.distances_):
        #         projection.make_projection(X[:, feature_idx], tree.Oi,
        #           tree.Oj, distance_type)
        # except:
        #     raise TypeError(f"Cannot compute distance \
        #                       using feature {feature_idx}")

        return X

    def decision_function(self, X: np.array):
        """
        Performs whole pipeline of calculating abnormality score and then
        offsetting it so that negative values denote outliers
        """
        sklearn_validation.check_is_fitted(self)
        X = sklearn_validation.check_array(X)
        X = self._validate_X_predict(X)

        scores = self.score_samples(X)

        return scores - self.offset_

    def predict(self, X: np.array):
        """Predict if a particular sample is an outlier or not.
        Paramteres
        ----------
            X : array-like, shape (n_samples, n_features)
                The input samples.
        Returns
        -------
            is_inlier : array, shape (n_samples,) For each observation, tell
            whether or not (0 or 1) it should be
            considered as an outlier according to the fitted model.
        """
        if isinstance(X, list):
            for tree in self.trees_:
                tree.set_distances(X.distances)

        X = prepare_X(X)
        decision_function = self.decision_function(X)

        if isinstance(X, list):
            for tree in self.trees_:
                tree.set_distances(self.distances_)

        is_outlier = np.zeros(X.shape[0], dtype=int)
        is_outlier[decision_function < 0] = 1
        return is_outlier

    def set_used_points():
        #TODO
        pass

def _build_tree(
    tree: RandomIsolationSimilarityTree,
    X: np.array,
    tree_idx: int,
    n_trees: int,
    subsample_size: int = 256,
    verbose: int = 0,
    bootstrap: bool = False,
):
    """Performs subsampling of data X and feeds a tree with subsampled data"""
    if verbose > 1:
        print("building tree %d of %d" % (tree_idx + 1, n_trees))

    # forest.bootstrap = True:
    # Randomly select samples with replacement for each tree
    # forest.bootstrap = False
    # Randomly select samples withouth replacement
    examples = tree.random_state.choice(
        X.shape[0], size=subsample_size, replace=bootstrap
    )
    tree.fit(X[examples])
    return tree
