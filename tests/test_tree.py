import sys
sys.path.append("C:\\University\\ProblemClasses\\Isolation-Similarity-Forest")

import pytest
import numpy as np
from unittest.mock import patch, MagicMock, call
from isf.tree import RandomIsolationSimilarityTree

@pytest.fixture
def sample_tree():
    #Lets imagine we have such data [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]] We will partition it into some sample tree
    #! we assume smaller or equal for left part
    #! we assume grater for right part
    root = RandomIsolationSimilarityTree("euclidean")
    root.X = np.arange(0,10).reshape((-1,1))
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

    root.left_node.left_node = RandomIsolationSimilarityTree("euclidean", depth=2)
    root.left_node.left_node.X = np.array([[0]])
    root.left_node.left_node.is_leaf = True
    root.left_node.right_node = RandomIsolationSimilarityTree("euclidean", depth=2)
    root.left_node.right_node.X = np.array([[1],[2],[3]])
    root.left_node.right_node.is_leaf = True

    root.right_node = RandomIsolationSimilarityTree("euclidean", depth = 1)
    root.right_node.X = np.array([[4],[5],[6],[7],[8],[9]])
    root.right_node.is_leaf = True

    return root

@patch("isf.splitting.project", side_effect=lambda x,Oi,Oj,dist,just_projection: x) # identity function as a projection
def test_get_leaf_x(projection_mock, sample_tree):
    leaf = sample_tree.get_leaf_x(np.array([[0]]))
    assert leaf.depth == 2
    assert np.array_equal(leaf.X, np.array([[0]]))

    leaf = sample_tree.get_leaf_x(np.array([[6]]))
    assert leaf.depth == 1
    assert np.array_equal(leaf.X, np.array([[4],[5],[6],[7],[8],[9]]))

@patch("isf.utils.measures._average_path_length", side_effect = lambda n: n) # Lets assume that average path length is equal to number of instances inside a node
def test_depth_estimate(avg_path_mock, sample_tree):
    #order is roor, root.left, root.left.left, root.left.right, root.right
    correct_depth = [10, 5, 2, 5, 7] # depth + average_path_length if node is not pure else depth

    nodes = [sample_tree, sample_tree.left_node, sample_tree.left_node.left_node, sample_tree.left_node.right_node, sample_tree.right_node]
    calculated_depths = [n.depth_estimate() for n in nodes]

    assert np.array_equal(correct_depth, calculated_depths)

def test_path_lengths():
    root = RandomIsolationSimilarityTree("euclidean")
    child = RandomIsolationSimilarityTree("euclidean")

    child.depth_estimate = MagicMock(side_effect = [4, 10 ,7]) # Simulating such depths were returned
    root.get_leaf_x = MagicMock(return_value = child)

    X = np.array([[1,2,3], [4,7,8], [6, 3, 2]])
    path_lengths = root.path_lengths_(X)
    
    assert np.array_equal(path_lengths, np.array([4,10,7]))

    proper_args = [ np.array([[1,2,3]]), np.array([[4,7,8]]), np.array([[6,3,2]])] # We check if we pass input always as a 2 dimensional array, otherwise project error may fail.
    #Has calls method works bad with numpy arrays as arguments
    for call, proper_arg in zip(root.get_leaf_x.call_args_list, proper_args):
        assert np.array_equal(call.args[0], proper_arg) #args of call object are tuples always


def test_choose_reference_point():
    root = RandomIsolationSimilarityTree("euclidean")
    root.X = np.array([[5, 8, 10], [4, 2, 3], [10, 11, 12]])
    root.random_instance = MagicMock()
    root.random_instance.choice = MagicMock(return_value = (0, 2))
    root.feature_index = 1

    Oi, Oj, i, j = root.choose_reference_points()

    assert (Oi, Oj, i ,j) == (8, 11, 0, 2)

def test_create_node():
    pass

