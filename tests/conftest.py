from unittest.mock import MagicMock

import numpy as np
import pytest
from rsif.tree import RandomSimilarityIsolationTree


@pytest.fixture
def sample_tree():
    # Lets imagine we have such data [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]] We will partition it into some sample tree
    # !we assume smaller or equal for left part
    # !we assume grater for right part
    DISTANCE = MagicMock()
    DISTANCE.project = lambda x, Oi, Oj, tree, random_instance: x
    root = RandomSimilarityIsolationTree(
        "euclidean", features_span=[(0, 1), (1, 2), (1, 3)]
    )
    root.feature_start = 0
    root.feature_end = 1
    root.X = np.arange(0, 10).reshape((-1, 1))
    root.split_point = 3
    root.distance_index = 0
    root.feature_index = 0
    root.Oi = 3
    root.Oj = 4
    root.distances = [[DISTANCE]]
    root.test_distances = [[DISTANCE]]

    root.left_node = RandomSimilarityIsolationTree(
        "euclidean", depth=1, features_span=[(0, 1), (1, 2), (1, 3)]
    )
    root.left_node.X = np.array([[0], [1], [2], [3]])
    root.left_node.feature_start = 0
    root.left_node.feature_end = 1
    root.left_node.split_point = 0
    root.left_node.feature_index = 0
    root.left_node.distance_index = 0
    root.left_node.Oi = 1
    root.left_node.Oj = 3
    root.left_node.distances = [[DISTANCE]]
    root.left_node.test_distances = [[DISTANCE]]

    root.left_node.left_node = RandomSimilarityIsolationTree(
        "euclidean", depth=2, features_span=[(0, 1), (1, 2), (1, 3)]
    )
    root.left_node.left_node.X = np.array([[0]])
    root.left_node.left_node.is_leaf = True
    root.left_node.right_node = RandomSimilarityIsolationTree(
        "euclidean", depth=2, features_span=[(0, 1), (1, 2), (1, 3)]
    )
    root.left_node.right_node.X = np.array([[1], [2], [3]])
    root.left_node.right_node.is_leaf = True

    root.right_node = RandomSimilarityIsolationTree(
        "euclidean", depth=1, features_span=[(0, 1), (1, 2), (1, 3)]
    )
    root.right_node.X = np.array([[4], [5], [6], [7], [8], [9]])
    root.right_node.is_leaf = True

    return root


@pytest.fixture()
def broader_tree(sample_tree):
    sample_tree.left_node.left_node.is_leaf = False
    sample_tree.left_node.left_node.Oi = 9
    sample_tree.left_node.left_node.Oj = 7

    sample_tree.left_node.left_node.left_node = RandomSimilarityIsolationTree(
        "euclidean", depth=3, features_span=[(0, 1), (1, 2), (1, 3)]
    )
    sample_tree.left_node.left_node.left_node.is_leaf = True

    sample_tree.left_node.left_node.right_node = RandomSimilarityIsolationTree(
        "euclidean", depth=3, features_span=[(0, 1), (1, 2), (1, 3)]
    )
    sample_tree.left_node.left_node.right_node.is_leaf = True

    return sample_tree
