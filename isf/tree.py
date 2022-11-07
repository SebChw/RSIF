from isf.utils.validation import prepare_X, check_random_state, check_distance, get_random_instance
import isf.splitting as splitting 
import isf.utils.measures as measures
import numpy as np

class RandomIsolationSimilarityTree():
    def __init__(self,
                 distance,
                 random_state=None,
                 max_features="auto",
                 max_depth=8,
                 most_different=False,
                 depth=0):
        self.distance = distance
        self.random_state = random_state
        self.max_depth = max_depth
        self.most_different = most_different
        self.depth = depth
        self.is_leaf=False
        self.random_instance = get_random_instance(self.random_state)

        

    def fit(self, X, y=None):
        self.X = prepare_X(X)

        self.distances_ = check_distance(self.distance, self.X.shape[1])
        #From IF perspective this is also beneficial, if there is no variance in feature, how could we detect outliers using it?
        features_with_nonunique_values = splitting.get_features_with_nonunique_values(self.X, self.distances_)

        if (self.max_depth == self.depth) or (self.X.shape[0] == 1)  or (len(features_with_nonunique_values) == 0):
             self._set_leaf()
        else:
            self.feature_index = self.random_instance.choice(features_with_nonunique_values, size=1)[0]
            self.Oi, self.Oj, i, j = self.choose_reference_points()
            indices, self.projection_sorted = splitting.project(self.X[:, self.feature_index], self.Oi, self.Oj, self.distances_[self.feature_index])

            self.split_point = self.random_instance.uniform(low=self.projection_sorted[0], high=self.projection_sorted[-1], size=1)

            self.left_samples, self.right_samples = self._partition()

            if (self.left_samples.shape[0] == 0) or (self.right_samples.shape[0] == 0):
                self._set_leaf()
            else:
                self.left_node = self._create_node(self.left_samples)
                self.right_node = self._create_node(self.right_samples)

        return self

    def _partition(self):
        left_samples = np.nonzero(self.projection_sorted - self.split_point <= 0)[0]
        right_samples = np.setdiff1d(range(self.X.shape[0]), left_samples)
        return left_samples, right_samples

    def _set_leaf(self):
        self.is_leaf = True

    def _create_node(self, samples):
        return RandomIsolationSimilarityTree(
            distance=self.distances_,
            max_depth=self.max_depth,
            random_state=self.random_state,
            depth = self.depth +1
        ).fit(self.X[samples])

    def choose_reference_points(self):
        i, j = self.random_instance.choice(self.X.shape[0], size=2 , replace=False)

        Oi = self.X[i, self.feature_index]
        Oj = self.X[j, self.feature_index]
        return Oi, Oj, i, j

    def path_lengths_(self, X, check_input=True):
        return np.array([self.get_leaf_x(x.reshape(1, -1)).depth_estimate() for x in X])

    def depth_estimate(self):
        """ Returns leaf in which our X would lie.
            If current node is a leaf (is pure), then return its depth, 
            if not add average_path_length from that leaf to estimate when it would be pure.
        """
        n = self.X.shape[0] # how many instances we have at this particular node
        c = 0
        if n > 1: # node is not pure
            c = measures._average_path_length(n) # how far current node would expand on average
        return self.depth + c

    def get_leaf_x(self, x):
        """ Returns leaf in which our X would lie.
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

        t = self.left_node if splitting.project(x[:, self.feature_index], self.Oi, self.Oj, self.distances_[self.feature_index], just_projection=True).item() <= self.split_point else self.right_node
        if t is None:
            return self

        return t.get_leaf_x(x)