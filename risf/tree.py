import numpy as np

import risf.splitting as splitting
import risf.utils.measures as measures
from risf.utils.validation import check_distance, check_random_state, prepare_X


class RandomIsolationSimilarityTree:
    """Unsupervised Similarity Tree measuring outlyingness score.
    Random Isolation Similarity Trees are base models used as building blocks
    for Random Isolation Similarity Forest ensemble.
        Parameters
        ----------
        random_state : int optional (default=1)
            If int, random_state is the seed used
            by the random number generator;
        distance : str, risf.distance.Distance object or
                   list of them (default='euclidean')
            If str or risf.distance.Distance then same function is
                                            used for all features
            if list then each feature is assosiated with corresponding distance
        max_depth : integer or None, optional (default=None)
            The maximum depth of the tree. If None, then nodes
            are expanded until all leaves are pure.
        depth : int depth of the tree count
        Attributes
        ----------
        _left_node : SimilarityTreeClassifier current node's left child node
        _right_node : SimilarityTreeClassifier current node's right child node
        Oi : first data-point used for drawing
            split direction in the current node
        Oj : second data-point used for drawing
            split direction in the current node
        _split_point = float similarity value decision boundary
        _is_leaf :
            bool indicating if current node is a leaf
            has been reached (depth == max_depth)
    """

    def __init__(self, distance, random_state=None, max_depth=8, depth=0):
        self.distance = distance
        self.max_depth = max_depth
        self.depth = depth
        self.left_node = None
        self.right_node = None
        self.is_leaf = False
        # This is very important part. We must assert that each tree will have independent random choices!
        # If we pass same state to every sub tree and create new random instance then this will fail
        self.random_state = check_random_state(random_state)

    def fit(self, X, y=None):
        """
        Build a Isolation Similarity Tree from the training set X.
               Parameters
               ----------
                   X : array-like of any type, as long as suitable similarity
                        function is provided
                       The training input samples.
                   y : None, added to follow Scikit-Learn convention
                   check_input : bool (default=False), allows to skip input
                        validation multiple times.
               Returns
               -------
                   self : object
        """
        self.prepare_to_fit(X)

        # From IF perspective this is also beneficial, if there is no variance
        # in feature, how could we detect outliers using it?
        non_unique_features = splitting.get_features_with_nonunique_values(
            self.X, self.distances_
        )

        if (
            (self.max_depth == self.depth)
            or (self.X.shape[0] == 1)
            or (len(non_unique_features) == 0)
        ):
            self._set_leaf()
        else:
            self.feature_index = self.random_state.choice(
                non_unique_features, size=1
            )[0]

            self.Oi, self.Oj, i, j = self.choose_reference_points()
            self.projection = splitting.project(
                self.X[:, self.feature_index],
                self.Oi,
                self.Oj,
                self.distances_[self.feature_index],
            )

            self.split_point = self.select_split_point()

            self.left_samples, self.right_samples = self._partition()

            if ((self.left_samples.shape[0] == 0)
                    or (self.right_samples.shape[0] == 0)):
                self._set_leaf()
            else:
                self.left_node = self._create_node(self.left_samples)
                self.right_node = self._create_node(self.right_samples)

        return self

    def prepare_to_fit(self, X):
        """Run all necessary checks and preparation steps for fit"""
        self.X = prepare_X(X)
        self.distances_ = check_distance(self.distance, self.X.shape[1])

    def set_distances(self, distances):
        self.distances_ = distances

    def select_split_point(self):
        """
        Choses random split point between min and max value of the projections
        """
        self.min_projection_value = self.projection.min()
        self.max_projection_value = self.projection.max()
        return self.random_state.uniform(
            low=self.min_projection_value, high=self.max_projection_value,
            size=1
        )

    def _partition(self):
        """
        Returns indices of samples that should go to the left node and right
        node respectively
        """
        left_samples = np.nonzero(self.projection - self.split_point <= 0)[
            0
        ]  # This return nonzero indices as a tuple of arrays
        right_samples = np.setdiff1d(
            range(self.X.shape[0]), left_samples
        )  # This return elements that are in first set but not in second set
        return left_samples, right_samples

    def _set_leaf(self):
        self.is_leaf = True

    def _create_node(self, samples):
        """Create child node"""
        return RandomIsolationSimilarityTree(
            distance=self.distance,
            max_depth=self.max_depth,
            random_state=self.random_state,  # Random state mustn't be an INTEGER now!
            depth=self.depth + 1,
        ).fit(self.X[samples])

    def choose_reference_points(self):
        """
        Randomly selects 2 data points that will create a pair and based on
        which we will create our projection direction
        """
        i, j = self.random_state.choice(
            self.X.shape[0], size=2, replace=False)

        Oi = self.X[i, self.feature_index]
        Oj = self.X[j, self.feature_index]
        return Oi, Oj, i, j

    def path_lengths_(self, X):
        """Estimates depth at which ach data point would be located in this
        tree"""
        path_lengths = np.array([self.get_leaf_x(x.reshape(1, -1)).depth_estimate()
                                 for x in X])
        return path_lengths

    def depth_estimate(self):
        """Returns leaf in which our X would lie.
        If current node is a leaf (is pure), then return its depth,
        if not add average_path_length from that leaf to estimate when it
        would be pure.
        """
        n = self.X.shape[0]  # how many instances we have at this node
        c = 0
        if n > 1:  # node is not pure
            c = measures._average_path_length(
                n
            )  # how far current node would expand on average
        return self.depth + c

    def get_leaf_x(self, x):
        """Returns leaf in which our X would lie.
        Parameters
        ----------
        x : a data-point
        Returns
        -------
        data-point's path length, according to single a tree.
        """
        if self.is_leaf:
            return self

        assert self.Oi is not None
        assert self.Oj is not None

        t = (
            self.left_node
            if splitting.project(
                x[:, self.feature_index],
                self.Oi,
                self.Oj,
                self.distances_[self.feature_index],
            ).item()
            <= self.split_point
            else self.right_node
        )

        return t.get_leaf_x(x)

    def get_used_points(self):
        if self.is_leaf:
            return set()

        left_used_points = self.left_node.get_used_points()
        right_used_points = self.right_node.get_used_points()
        my_used_points = set([self.Oi, self.Oj])

        return left_used_points.union(right_used_points).union(my_used_points)
