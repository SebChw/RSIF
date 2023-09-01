from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np

import risf.splitting as splitting
import risf.utils.measures as measures
from risf.distance import SelectiveDistance, TestDistanceMixin, TrainDistanceMixin
from risf.utils.validation import check_random_state


class RandomIsolationSimilarityTree:
    """Unsupervised Similarity Tree measuring outlyingness score.
    Random Isolation Similarity Trees are base models used as building blocks
    for Random Isolation Similarity Forest ensemble.
    """

    def __init__(
        self,
        distances: List[List[Union[SelectiveDistance, TrainDistanceMixin]]],
        features_span: List[Tuple[int, int]],
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        max_depth: int = 8,
        depth: int = 0,
        pair_strategy="local",
    ):
        """
        Parameters
        ----------
        distances : List[List[Union[SelectiveDistance, TrainDistanceMixin]]]
            for every parameter we have a list of distances
        features_span : List[Tuple[int, int]]
            Very important parameter it tells boundaries between different features
        random_state : int
            EACH TREE MUSH HAVE UNIQUE RANDOM STATE INTEGER. DON'T PASS SAME RANDOM STATE INSTANCE TO ALL TREES
        max_depth : int, optional
            max depth that tree can reach before setting node to leaf, by default 8
        depth : int, optional
            depth of current node, by default 0
        """
        self.features_span = features_span
        self.distances = distances
        self.max_depth = max_depth
        self.depth = depth
        self.left_node = None
        self.right_node = None
        self.is_leaf = False
        self.pair_strategy = pair_strategy
        self.random_state = check_random_state(random_state)

    def fit(self, X: np.ndarray, y=None) -> RandomIsolationSimilarityTree:
        """
        Build a Isolation Similarity Tree from the training set X.

        Steps performed for every node:
        - Seek for non unique features (if feature has only one value for all instances it is useless)
        - Check if we reached max depth or if we have only one instance left or if we have no non unique features. If yes set node to leaf
        - Choose random feature from non unique features
        - Choose random distance from distances for that feature
        - filter selected objects. This step is optional and is used when we want to limit number of objects that we consider for projection
        - Choose random pair of objects from selected objects
        - Create projection based on selected distance and selected objects

        Parameters
        ----------
            X : array-like
                The training input samples.
            y : None, added to follow Scikit-Learn convention
        Returns
        -------
            RandomIsolationSimilarityTree
                The fitted tree.
        """
        self.prepare_to_fit(X)

        # TODO: this variable should be called unique_features
        non_unique_features = splitting.get_features_with_unique_values(
            self.X, self.distances, self.features_span
        )

        if (
            (self.max_depth == self.depth)
            or (self.X.shape[0] == 1)
            or (len(non_unique_features) == 0)
        ):
            self._set_leaf()
        else:
            self.feature_index = self.random_state.choice(non_unique_features, size=1)[0]  # fmt: skip

            self.distance_index = self.random_state.randint(
                0, len(self.distances[self.feature_index])
            )

            selected_distance = self.distances[self.feature_index][self.distance_index]

            selected_objects_indices = self._get_selected_objects(selected_distance)
            if selected_objects_indices is None:
                self._set_leaf()
                return self

            self.feature_start, self.feature_end = self.features_span[
                self.feature_index
            ]

            self.Oi, self.Oj = self.choose_reference_points(
                selected_objects_indices, selected_distance
            )
            self.projection = selected_distance.project(
                self.X[:, self.feature_start : self.feature_end],
                self.Oi,
                self.Oj,
                tree=self,
                random_instance=self.random_state,
            )

            self.split_point = self.select_split_point()

            self.left_samples, self.right_samples = self._partition()

            if (self.left_samples.shape[0] == 0) or (self.right_samples.shape[0] == 0):
                self._set_leaf()
            else:
                self.left_node: RandomIsolationSimilarityTree = self._create_node(self.left_samples)  # fmt: skip
                self.right_node: RandomIsolationSimilarityTree = self._create_node(self.right_samples)  # fmt: skip

        return self

    def prepare_to_fit(self, X):
        """Run all necessary checks and preparation steps for fit"""
        self.X = X
        self.test_distances = self.distances

    def set_test_distances(self, distances: List[List[TestDistanceMixin]]):
        """If we use distance based on precalculated matrix we must set it here before predicting on new data

        Parameters
        ----------
        distances : List[List[TestDistanceMixin]]
            list of distances between test objects and training objects
        """
        self.test_distances = distances

        if self.is_leaf:
            return

        self.left_node.set_test_distances(distances)
        self.right_node.set_test_distances(distances)

    def select_split_point(self) -> np.ndarray:
        """
        Choses random split point between min and max value of the projections

        Returns
        -------
        np.ndarray
            random split point selected uniformly from the interval [min_projection_value, max_projection_value]
        """
        self.min_projection_value = self.projection.min()
        self.max_projection_value = self.projection.max()
        return self.random_state.uniform(
            low=self.min_projection_value, high=self.max_projection_value, size=1
        )

    def _partition(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns indices of samples that should go to the left node and right
        node respectively

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            indices of samples that should go to the left node and right
        """

        left_samples = np.nonzero(self.projection - self.split_point <= 0)[0]
        right_samples = np.setdiff1d(range(self.X.shape[0]), left_samples)
        return left_samples, right_samples

    def _set_leaf(self):
        self.is_leaf = True

    def _create_node(self, samples: np.ndarray) -> RandomIsolationSimilarityTree:
        """Create child node"""
        return RandomIsolationSimilarityTree(
            distances=self.distances,
            features_span=self.features_span,
            max_depth=self.max_depth,
            random_state=self.random_state,  # Random state must be a random instance for now
            depth=self.depth + 1,
        ).fit(self.X[samples])

    def choose_reference_points(
        self, selected_objects_indices, selected_distance=None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Randomly selects 2 data points that will create a pair and based on
        which we will create our projection direction
        """
        if (
            selected_distance is None
            or self.pair_strategy == "random"
            or isinstance(selected_distance, SelectiveDistance)
        ):
            i, j = self.random_state.choice(
                selected_objects_indices, size=2, replace=False
            )
            Oi = self.X[i, self.feature_start : self.feature_end]
            Oj = self.X[j, self.feature_start : self.feature_end]
        else:
            selected_objects = (
                self.X[selected_objects_indices, self.feature_start : self.feature_end]
                .flatten()
                .astype(np.uint32)
            )
            if self.pair_strategy == "local":
                distance_matrix = selected_distance.distance_matrix[selected_objects]
                distance_matrix = distance_matrix[:, selected_objects]
                best_pair = np.unravel_index(
                    np.argmax(distance_matrix), distance_matrix.shape
                )
                Oi, Oj = selected_objects[best_pair[0]], selected_objects[best_pair[1]]

            elif self.pair_strategy == "global":
                id_ = self.random_state.choice(
                    len(selected_distance.top_k_pairs), size=1
                )
                Oi, Oj = selected_distance.top_k_pairs[id_[0]]
            elif self.pair_strategy == "two_step":
                Oi = self.random_state.choice(selected_objects, size=1)[0]
                j = selected_distance.distance_matrix[Oi, selected_objects].argmax()
                Oj = selected_objects[j]
            else:
                raise ValueError("Unsupported pair strategy")

        return Oi, Oj

    def path_lengths_(self, X: np.ndarray) -> np.ndarray:
        """Estimates depth at which data point would be located in this tree"""
        path_lengths = np.array(
            [self.get_leaf_x(x.reshape(1, -1)).depth_estimate() for x in X]
        )
        return path_lengths

    def depth_estimate(self) -> int:
        """Returns leaf in which X from this node  would lie.
        If current node is a leaf (is pure), then return its depth,
        if not add average_path_length from that leaf to estimate when it would be pure.
        """
        n = self.X.shape[0]
        c = 0
        if n > 1:
            c = measures._average_path_length(n)
        return self.depth + c

    def _get_selected_objects(
        self, selected_distance: Union[TrainDistanceMixin, SelectiveDistance]
    ) -> Union[int, np.ndarray, None]:
        """Returns selected objects for current node. Useful when we restrict number of possible objects for efficiency,
        It is used only if we use `TrainDistanceMixin

        Parameters
        ----------
        selected_distance : Union[TrainDistanceMixin, SelectiveDistance]
            distance used

        Returns
        -------
        Union[int, np.ndarray]
            selected objects. Int means number of objects to use. Array means which objects to use
        """

        # Second part in if is used not to make redundant calculations when we want to use all objects
        if (
            hasattr(selected_distance, "selected_objects")
            and not selected_distance.selected_objects.shape[0]
            == selected_distance.distance_matrix.shape[0]
        ):
            # Intersect1d function returns indices
            selected_objects = selected_distance.selected_objects
            selected_objects = np.intersect1d(
                self.X[:, self.feature_index],  # these are just indices
                selected_objects,  # these are just indices
                assume_unique=True,
                return_indices=True,
            )[1]
            # If we limit number of possible objects we may end up with less than 2 objects quite often
            if len(selected_objects) < 2:
                return None

            return selected_objects

        return np.arange(self.X.shape[0])

    def get_leaf_x(self, x: np.ndarray) -> RandomIsolationSimilarityTree:
        """Returns leaf in which our X would lie. By performing projections and then comparing them to split points

        Parameters
        ----------
        x : a data-point with shape (1, n_features)

        Returns
        -------
        data-point's path length, according to single a tree.
        """
        if self.is_leaf:
            return self

        assert self.Oi is not None
        assert self.Oj is not None

        test_distance = self.test_distances[self.feature_index][self.distance_index]

        t = (
            self.left_node
            if test_distance.project(
                x[:, self.feature_start : self.feature_end],
                self.Oi,
                self.Oj,
                tree=self,
                random_instance=self.random_state,
            ).item()
            <= self.split_point
            else self.right_node
        )

        return t.get_leaf_x(x)

    def get_used_points(self) -> set:
        """Return set of indices of all points used in this tree

        Returns
        -------
        set
            indices of all points used to perform projections in this tree
        """
        # TODO: Think about dividing this also to points used for particular feature so that we perform even less calculations for mixed datatypes
        if self.is_leaf:
            return set()

        left_used_points = self.left_node.get_used_points()
        right_used_points = self.right_node.get_used_points()
        my_used_points = set([self.Oi, self.Oj])

        return left_used_points.union(right_used_points).union(my_used_points)
