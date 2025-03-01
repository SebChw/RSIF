from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from rsif.tree import RandomSimilarityIsolationTree


def test_get_leaf_x(sample_tree):
    leaf = sample_tree.get_leaf_x(np.array([[0]]))
    assert leaf.depth == 2
    assert np.array_equal(leaf.X, np.array([[0]]))

    leaf = sample_tree.get_leaf_x(np.array([[6]]))
    assert leaf.depth == 1
    assert np.array_equal(leaf.X, np.array([[4], [5], [6], [7], [8], [9]]))


@patch(
    "rsif.utils.measures._average_path_length", side_effect=lambda n: n
)  # Lets assume that average path length is equal to number of instances inside a node
def test_depth_estimate(avg_path_mock, sample_tree):
    # order is roor, root.left, root.left.left, root.left.right, root.right
    correct_depth = [
        10,
        5,
        2,
        5,
        7,
    ]  # depth + average_path_length if node is not pure else depth

    nodes = [
        sample_tree,
        sample_tree.left_node,
        sample_tree.left_node.left_node,
        sample_tree.left_node.right_node,
        sample_tree.right_node,
    ]
    calculated_depths = [n.depth_estimate() for n in nodes]

    assert np.array_equal(correct_depth, calculated_depths)


def test_path_lengths():
    root = RandomSimilarityIsolationTree("euclidean", features_span=[])
    child = RandomSimilarityIsolationTree("euclidean", features_span=[])

    child.depth_estimate = MagicMock(
        side_effect=[4, 10, 7]
    )  # Simulating such depths were returned
    root.get_leaf_x = MagicMock(return_value=child)

    X = np.array([[1, 2, 3], [4, 7, 8], [6, 3, 2]])
    path_lengths = root.path_lengths_(X)

    assert np.array_equal(path_lengths, np.array([4, 10, 7]))

    proper_args = [
        np.array([[1, 2, 3]]),
        np.array([[4, 7, 8]]),
        np.array([[6, 3, 2]]),
    ]  # We check if we pass input always as a 2 dimensional array, otherwise project error may fail.
    # Has calls method works bad with numpy arrays as arguments
    for call, proper_arg in zip(root.get_leaf_x.call_args_list, proper_args):
        assert np.array_equal(
            call.args[0], proper_arg
        )  # args of call object are tuples always


def test_choose_reference_point():
    root = RandomSimilarityIsolationTree("euclidean", features_span=[])
    root.feature_start = 1
    root.feature_end = 2
    root.X = np.array([[5, 8, 10], [4, 2, 3], [10, 11, 12]])
    root.random_state = MagicMock()
    root.random_state.choice = MagicMock(return_value=(0, 2))
    root.feature_index = 1

    selected_objects = [0, 1, 2]

    Oi, Oj = root.choose_reference_points(selected_objects)

    assert root.random_state.choice.called_one_with(
        selected_objects, size=2, replace=False
    )

    assert (Oi, Oj) == (8, 11)


def test_create_node():
    """I couldn't think of how to mock this fit function actually it must be called"""
    # with patch.object(RandomSimilarityIsolationTree, "fit", side_effect = lambda x: x) as mock_fit:
    root = RandomSimilarityIsolationTree(
        [["euclidean"], ["euclidean"]], max_depth=10, random_state=5, features_span=[]
    )
    root.X = np.array([[30, 25], [5, 8], [2, 11], [4, 13], [2, 10], [35, 12]])

    child = root._create_node(np.array([1, 3, 4]))

    # Check if parameters are rewritten
    assert child.distances == [
        ["euclidean"],
        ["euclidean"],
    ]
    assert child.max_depth == 10
    assert child.random_state == root.random_state
    # to make sure the random generators are passed well and don't produce identical scores
    assert (
        child.random_state.randint(0, 100, 100)
        == root.random_state.randint(0, 100, 100)
    ).sum() < 10

    # Now I mock it to check if fit will be called with proper parameters
    with patch.object(
        RandomSimilarityIsolationTree, "fit", side_effect=lambda x: x
    ) as mock_fit:
        root = RandomSimilarityIsolationTree(
            "euclidean", max_depth=10, random_state=5, features_span=[]
        )
        root.X = np.array([[30, 25], [5, 8], [2, 11], [4, 13], [2, 10], [35, 12]])

        child = root._create_node(np.array([1, 3, 4]))
        assert np.array_equal(
            np.array([[5, 8], [4, 13], [2, 10]]), mock_fit.call_args_list[0][0][0]
        )  # the array is quite nested


@pytest.fixture
def projection_data():
    root = RandomSimilarityIsolationTree("euclidean", features_span=[])
    root.projection = np.array([10, -7, 0, -2, 20])
    root.X = np.array([1, -3, 0, 2, 10])

    return root


def test_partition_typical_partitioning(projection_data):
    projection_data.split_point = 0  # Equal values should go into left node
    left_samples, right_samples = projection_data._partition()

    assert np.array_equal(left_samples, np.array([1, 2, 3]))
    assert np.array_equal(right_samples, np.array([0, 4]))


def test_partition_all_left(projection_data):
    projection_data.split_point = -7.0001
    left_samples, right_samples = projection_data._partition()

    assert np.array_equal(left_samples, np.array([]))
    assert np.array_equal(right_samples, np.array([0, 1, 2, 3, 4]))


def test_partition_all_right(projection_data):
    projection_data.split_point = 20.0001  # Equal values should go into left node
    left_samples, right_samples = projection_data._partition()

    assert np.array_equal(left_samples, np.array([0, 1, 2, 3, 4]))
    assert np.array_equal(right_samples, np.array([]))


def test_select_split_point(projection_data):
    projection_data.random_state = MagicMock
    projection_data.random_state.uniform = MagicMock(
        side_effect=lambda low, high, size: low + high / 2
    )
    projection_data.select_split_point()

    assert projection_data.min_projection_value == -7
    assert projection_data.max_projection_value == 20
    projection_data.random_state.uniform.assert_called_with(low=-7, high=20, size=1)


@pytest.fixture
def fit_data():
    return np.array([[10, 20], [20, 10], [5, -8]])


@pytest.fixture
def mocked_tree(fit_data):
    root = RandomSimilarityIsolationTree("euclidean", features_span=[(0, 1), (1, 2)])
    root.prepare_to_fit = MagicMock()
    root._set_leaf = MagicMock()
    root.random_state = MagicMock()
    root.random_state.choice = MagicMock(return_value=[1])  # to get 1st feature
    root.random_state.randint = MagicMock(return_value=0)  # To get 0th distance
    root.choose_reference_points = MagicMock(return_value=(0, 1))
    root.select_split_point = MagicMock()
    root._partition = MagicMock(return_value=(np.array([0, 2]), np.array([3, 4])))
    root._create_node = MagicMock()

    root.X = fit_data
    DISTANCE = MagicMock()
    DISTANCE.project = lambda x, Oi, Oj, tree, random_instance: x
    root.distances = [[DISTANCE], [DISTANCE]]

    return root


@patch("rsif.splitting.get_features_with_unique_values", return_value=[1, 2])
def test_fit_positive_scenario(mock_get_features, fit_data, mocked_tree):
    # Check if everything is runned with correct parameters
    mocked_tree._get_selected_objects = MagicMock(return_value=np.array([0, 1, 2]))
    mocked_tree.fit(fit_data)
    mock_get_features.assert_called_with(
        fit_data, mocked_tree.distances, mocked_tree.features_span
    )
    mocked_tree.prepare_to_fit.assert_called_with(fit_data)
    mocked_tree.random_state.choice.assert_called_with([1, 2], size=1)
    mocked_tree.choose_reference_points.assert_called_once()

    mocked_tree.select_split_point.assert_called_once()
    mocked_tree._partition.assert_called_once()
    (
        create_node_left_correct,
        create_node_right_correct,
    ) = mocked_tree._create_node.call_args_list
    assert np.array_equal(create_node_left_correct[0][0], np.array([0, 2]))
    assert np.array_equal(create_node_right_correct[0][0], np.array([3, 4]))


@patch("rsif.splitting.get_features_with_unique_values", return_value=np.array([]))
def test_fit_no_features_with_unique_values(fit_data, mocked_tree):
    mocked_tree.fit(fit_data)
    mocked_tree._set_leaf.assert_called_once()


def test_fit_no_selected_objects(fit_data, mocked_tree):
    mocked_tree._get_selected_objects = MagicMock(return_value=None)
    mocked_tree.fit(fit_data)
    mocked_tree._set_leaf.assert_called_once()


@patch("rsif.splitting.get_features_with_unique_values", return_value=np.array([1, 2]))
def test_fit_one_instance(fit_data, mocked_tree):
    mocked_tree.X = np.array([[1, 2]])
    mocked_tree.fit(fit_data)
    mocked_tree._set_leaf.assert_called_once()


@patch("rsif.splitting.get_features_with_unique_values", return_value=np.array([1, 2]))
def test_fit_max_depth_reached(mock_get_features, fit_data, mocked_tree):
    mocked_tree.max_depth = 10
    mocked_tree.depth = 10
    mocked_tree.fit(fit_data)
    mocked_tree._set_leaf.assert_called_once()


@patch("rsif.splitting.get_features_with_unique_values", return_value=np.array([1, 2]))
def test_fit_empty_partition(mock_get_features, fit_data, mocked_tree):
    mocked_tree._partition = MagicMock(return_value=(np.array([]), np.array([1, 2])))
    mocked_tree.fit(fit_data)
    mocked_tree._set_leaf.assert_called_once()


@patch("rsif.splitting.get_features_with_unique_values", return_value=np.array([1, 2]))
def test_fit_empty_partition2(mock_get_features, fit_data, mocked_tree):
    mocked_tree._partition = MagicMock(return_value=(np.array([1, 2]), np.array([])))
    mocked_tree.fit(fit_data)
    mocked_tree._set_leaf.assert_called_once()


def test_get_used_points(broader_tree):
    assert broader_tree.get_used_points() == set([3, 4, 1, 9, 7])


def test_set_test_distances(broader_tree):
    TEST_DIST = ["test1", "test2"]
    broader_tree.set_test_distances(TEST_DIST)

    assert broader_tree.test_distances == TEST_DIST
    assert broader_tree.left_node.test_distances == TEST_DIST
    assert broader_tree.left_node.left_node.test_distances == TEST_DIST


def test_get_selected_objects_no_selected_objects(mocked_tree):
    N_OBJECTS = mocked_tree.X.shape[0]
    assert len(mocked_tree._get_selected_objects("euclidean")) == N_OBJECTS
    assert len(mocked_tree._get_selected_objects(lambda x: x**2)) == N_OBJECTS

    selected_distance = MagicMock()
    selected_distance.selected_objects = np.array([0, 1, 2])
    selected_distance.distance_matrix = np.zeros((3, 3))

    assert len(mocked_tree._get_selected_objects(selected_distance)) == N_OBJECTS


def test_get_selected_objects_distance(mocked_tree):
    selected_distance = MagicMock()
    selected_distance.selected_objects = np.array([20, 11, 49])
    selected_distance.distance_matrix = np.zeros((5, 5))

    mocked_tree.X = np.array([[11, 11, 11], [50, 50, 50], [20, 20, 20]])
    mocked_tree.feature_index = 1
    assert np.array_equal(
        mocked_tree._get_selected_objects(selected_distance), np.array([0, 2])
    )

    mocked_tree.X = np.array([[0, 0, 0], [50, 50, 50], [20, 20, 20]])
    mocked_tree.feature_index = 0
    assert mocked_tree._get_selected_objects(selected_distance) is None
