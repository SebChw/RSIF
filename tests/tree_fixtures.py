import pytest
from risf.tree import RandomIsolationSimilarityTree
import numpy as np
@pytest.fixture
def sample_tree():
    # Lets imagine we have such data [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]] We will partition it into some sample tree
    #! we assume smaller or equal for left part
    #! we assume grater for right part
    root = RandomIsolationSimilarityTree("euclidean")
    root.X = np.arange(0, 10).reshape((-1, 1))
    root.split_point = 3
    root.feature_index = 0
    root.Oi = 3
    root.Oj = 4
    root.distances_ = ["euclidean"]

    root.left_node = RandomIsolationSimilarityTree("euclidean", depth=1)
    root.left_node.X = np.array([[0], [1], [2], [3]])
    root.left_node.split_point = 0
    root.left_node.feature_index = 0
    root.left_node.Oi = 1
    root.left_node.Oj = 3
    root.left_node.distances_ = ["euclidean"]

    root.left_node.left_node = RandomIsolationSimilarityTree(
        "euclidean", depth=2)
    root.left_node.left_node.X = np.array([[0]])
    root.left_node.left_node.is_leaf = True
    root.left_node.right_node = RandomIsolationSimilarityTree(
        "euclidean", depth=2)
    root.left_node.right_node.X = np.array([[1], [2], [3]])
    root.left_node.right_node.is_leaf = True

    root.right_node = RandomIsolationSimilarityTree("euclidean", depth=1)
    root.right_node.X = np.array([[4], [5], [6], [7], [8], [9]])
    root.right_node.is_leaf = True

    return root

@pytest.fixture()
def broader_tree(sample_tree):
    sample_tree.left_node.left_node.is_leaf = False
    sample_tree.left_node.left_node.Oi = 9
    sample_tree.left_node.left_node.Oj = 7

    sample_tree.left_node.left_node.left_node = RandomIsolationSimilarityTree("euclidean", depth = 3)
    sample_tree.left_node.left_node.left_node.is_leaf = True

    sample_tree.left_node.left_node.right_node = RandomIsolationSimilarityTree("euclidean", depth = 3)
    sample_tree.left_node.left_node.right_node.is_leaf = True
    
    return sample_tree