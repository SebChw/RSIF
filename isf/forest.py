from random import random
from sklearn.base import BaseEstimator, OutlierMixin
import numpy as np
from joblib import Parallel, delayed

from isf.tree import RandomIsolationSimilarityTree

import isf.utils.measures as measures
import sklearn.utils.validation as sklearn_validation
from isf.utils.validation import prepare_X, check_random_state_, check_max_samples
import isf.projection as projection



class RandomIsolationSimilarityForest(BaseEstimator, OutlierMixin):
    def __init__(
        self,
        distance='euclidean',
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


    def fit(self, X, y=None):
        self.X = prepare_X(X)

        self.random_state = check_random_state_(self.random_state)
        self.random_state=0
        self.subsample_size = check_max_samples(self.max_samples, self.X)

        if self.contamination == 'auto':
            self.offset_ = -0.5

        self.trees_ = [RandomIsolationSimilarityTree(distance=self.distance, max_depth=self.max_depth, 
                                                        random_state=self.random_state + i)
                       for i in range(self.n_estimators)]

        self.trees_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_build_tree)(tree, self, self.X, i, self.n_estimators, verbose=self.verbose)
            for i, tree in enumerate(self.trees_)
        )

        return self

    def calculate_mean_path_lengths(self, X):
        return np.mean([t.path_lengths_(X, check_input=False) for t in self.trees_], axis=0)

    def score_samples(self, X, n_estimators=None):
        # Average depth at which a sample lies over all trees
        mean_path_lengths = self.calculate_mean_path_lengths(X)

        # Depths are normalized in the same fashion as in Isolation Forest
        c = measures._average_path_length(self.subsample_size)
        
        scores = np.array([- 2 ** (-pl / c) for pl in mean_path_lengths]) 

        return scores

    def _validate_X_predict(self, X):
        """
            Takes first tree and checks if all distances can be calculated,
            not sure how to implement this. Making these projections h
        """
        # tree = self.trees_[0]
        # try:
        #     for feature_idx, distance_type in enumerate(tree.distances_):
        #         projection.make_projection(X[:, feature_idx], tree.Oi, tree.Oj, distance_type)
        # except:
        #     raise TypeError(f"Cannot compute distance using feature {feature_idx}")

        return X

    def decision_function(self, X, n_estimators=None):
        sklearn_validation.check_is_fitted(self)
        X = sklearn_validation.check_array(X)
        X = self._validate_X_predict(X)

        scores = self.score_samples(X, n_estimators=None)

        return scores - self.offset_

    def predict(self, X):
        decision_function = self.decision_function(X)

        is_inlier = np.ones(X.shape[0], dtype=int)
        is_inlier[decision_function < 0] = -1
        return is_inlier


def _build_tree(tree, forest, X, tree_idx, n_trees, verbose=0, bootstrap=False):
    if verbose > 1:
        print("building tree %d of %d" % (tree_idx + 1, n_trees))

    #forest.bootstrap = True:
        #Randomly select samples with replacement for each tree
    #forest.bootstrap = False
        #Randomly select samples withouth replacement
    examples = tree.random_instance.choice(X.shape[0], size=X.shape[0], replace=bootstrap)
    tree.fit(X[examples])
    return tree