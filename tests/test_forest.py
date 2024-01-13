import copy
from unittest.mock import MagicMock, Mock, call, patch

import numpy as np
import pytest

from rsif.forest import RandomSimilarityIsolationForest, _build_tree
from rsif.rsif_data import RsifData
from rsif.tree import RandomSimilarityIsolationTree


@patch.object(
    RandomSimilarityIsolationForest,
    "calculate_mean_path_lengths",
    return_value=np.array([2, 5, 10]),
)
@patch("rsif.utils.measures._average_path_length", return_value=5)
def test_score_samples(avg_p_length, rsif_mean_length):
    rsif = RandomSimilarityIsolationForest()
    rsif.subsample_size = (
        10  # Needed for method to run, method using it is mocked anyway
    )
    results = rsif.score_samples(
        np.ones((3, 3))
    )  # It should calculate -2^(-2/5), -2^(-5/5), -2^(-10/5)

    assert (results == [-0.757858283255199, -0.5, -0.25]).all()


@patch("sklearn.utils.validation.check_is_fitted")
@patch.object(
    RandomSimilarityIsolationForest,
    "score_samples",
    return_value=np.array([0, -1, -0.5]),
)  # decision function is calculated based on return of this function
def test_decision_function(score_samples, is_fitted):
    rsif = RandomSimilarityIsolationForest()
    rsif.decision_threshold_ = -0.5
    result = rsif.decision_function(
        np.array([1, 5, 3])
    )  # this should check all needed things and then calculate 0 - (-0.5), -1 -(-0.5), -0.5 -(-0.5)

    assert is_fitted.called
    assert score_samples.called
    assert np.allclose(result, [0.5, -0.5, 0])

    # test with different offset
    rsif.decision_threshold_ = -0.7
    result = rsif.decision_function(np.array([1, 5, 3]))

    assert is_fitted.called
    assert score_samples.called
    assert np.allclose(
        result, [0.7, -0.3, 0.2]
    )  # bigger offset it is harder to become an outlier scores are higher


@patch.object(
    RandomSimilarityIsolationForest,
    "decision_function",
    return_value=np.array([-0.5, 0.01, 0, -0.001, 0.75]),
)  # 0 shouldn't be outlier
def test_predict(decision_function):
    rsif = RandomSimilarityIsolationForest()
    predictions = rsif.predict(
        np.ones((5, 3))
    )  # we pass 5 objects with 3 attributes each

    assert decision_function.called
    assert np.array_equal(predictions, [1, 0, 0, 1, 0])


@patch.object(
    RandomSimilarityIsolationForest,
    "decision_function",
    return_value=np.array([-0.5, 0.01, 0, -0.001, 0.75]),
)
@patch("rsif.forest.prepare_X", side_effect=lambda x: (x, []))
def test_predict_rsif_data(prepare_X_mock, decision_function):
    rsif = RandomSimilarityIsolationForest()
    rsif.trees_ = [MagicMock(), MagicMock()]

    X = MagicMock(spec=RsifData)
    X.distances = ["euclidean", "euclidean"]
    X.shape = (5, 3)

    rsif.X = MagicMock(spec=RsifData)
    rsif.X.distances = ["euclidean2", "euclidean2"]

    predictions = rsif.predict(X)

    # !calls must be in order at first we swap calls to test set now we unswap to training data
    for tree in rsif.trees_:
        tree.assert_has_calls([call.set_test_distances(X.distances)])

    assert decision_function.called
    assert np.array_equal(predictions, [1, 0, 0, 1, 0])


def test_calculate_mean_path_lengths():
    rsif = RandomSimilarityIsolationForest()
    rsif.trees_ = []
    for i in range(
        1, 11
    ):  # lets add 10 trees to the forest and mock their path_length function
        rist = RandomSimilarityIsolationTree(distances="euclidean", features_span=[])
        rist.path_lengths_ = MagicMock(
            return_value=i * np.array([1, 2, 3, 4, 5])
        )  # this is the situation when we gave 5 object to be tested
        # and path lengths were equal to i * [1,2,3,4,5]
        rsif.trees_.append(rist)

    mean_path_lenghts = rsif.calculate_mean_path_lengths([[1], [2], [3], [4], [5]])
    assert np.allclose(mean_path_lenghts, [5.5, 11, 16.5, 22, 27.5])


def test_build_tree():
    tree = MagicMock()
    tree.random_state = MagicMock()
    tree.random_state.choice = MagicMock(
        side_effect=lambda x, size, replace: np.arange(0, size * 2)[::2]
    )
    tree.fit = MagicMock()

    X = np.random.rand(1000, 4)

    _build_tree(tree, X, tree_idx=10, n_trees=20)

    tree.random_state.choice.assert_called_once_with(1000, size=256, replace=False)

    assert np.array_equal(tree.fit.call_args_list[0][0][0], X[np.arange(0, 512)[::2]])


def test_set_offset():
    forest = RandomSimilarityIsolationForest()
    forest.set_offset()
    assert forest.decision_threshold_ == -0.5  # Default setting

    # Mocking
    forest.score_samples = MagicMock(
        return_value=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    )
    forest.X = np.array([0])

    # Testing
    forest.contamination = 0.9  # This means I assume 90% of my data are outliers
    forest.set_offset()
    assert forest.decision_threshold_ == 9

    forest.contamination = 0.2  # This means I assume 20% of my data are outliers
    forest.set_offset()
    assert forest.decision_threshold_ == 2

    # With y given
    forest.set_offset(np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]))
    forest.set_offset()
    assert forest.contamination == 6 / 11
    assert forest.decision_threshold_ == 5.454545454545454


def test_get_used_points(broader_tree):
    forest = RandomSimilarityIsolationForest()
    forest.trees_ = []
    trees_used_points = [
        [1, 6, 4, 2, 1, 2],
        [5, 1, 5, 1, 2, 1],
        [10, 12, 15, 12, 10, 2],
        [1, 5, 9, 13, 17, 21],
    ]
    for tree_used_points in trees_used_points:
        # It must be deepcopy as otherwise we overwrite same Oi's and Oj's
        tree = copy.deepcopy(broader_tree)
        o1, o2, o3, o4, o5, o6 = tree_used_points

        tree.Oi, tree.Oj = o1, o2
        tree.left_node.Oi, tree.left_node.Oj = o3, o4
        tree.left_node.left_node.Oi, tree.left_node.left_node.Oj = o5, o6
        forest.trees_.append(tree)

    assert forest.get_used_points() == set([1, 6, 4, 2, 5, 10, 12, 15, 9, 13, 17, 21])


@pytest.fixture
def dummy_forest() -> RandomSimilarityIsolationForest:
    return RandomSimilarityIsolationForest(
        random_state=10, max_samples=250, distances=[["euclidean"], ["euclidean"]]
    )


@patch("rsif.forest.prepare_X", side_effect=lambda x: (x, [(0, 1), (1, 2)]))
@patch("rsif.forest.check_random_state")
@patch("rsif.forest.check_max_samples")
def test_prepare_to_fit_np_array(
    mock_max_samples, mock_random_state, mock_prepare_X, dummy_forest
):
    X = np.array([[10, 20], [30, 40]])
    dummy_forest.prepare_to_fit(X)

    mock_max_samples.assert_called_once_with(250, X)
    mock_random_state.assert_called_once_with(10)
    mock_prepare_X.assert_called_once_with(X)

    assert np.array_equal(dummy_forest.X, X)


@patch("rsif.forest.prepare_X", side_effect=lambda x: (x, [(0, 1), (1, 2)]))
@patch("rsif.forest.check_random_state")
@patch("rsif.forest.check_max_samples", side_effect=lambda x, y: x)
def test_prepare_to_fit_rsif_data(
    mock_max_samples, mock_random_state, mock_prepare_X, dummy_forest
):
    X = Mock(spec=RsifData)
    dummy_forest.prepare_to_fit(X)

    mock_max_samples.assert_called_once_with(250, X)
    mock_random_state.assert_called_once_with(10)
    mock_prepare_X.assert_called_once_with(X)

    assert dummy_forest.X is X
    assert dummy_forest.max_depth == 8  # ceil(log2(250))


@patch("rsif.forest.RandomSimilarityIsolationTree")
def test_create_trees(tree_mock):
    R_STATE = "test"
    N_ESTIMATORS = 100
    DISTANCE = ["euclidean"]
    rsif = RandomSimilarityIsolationForest(
        random_state=R_STATE,
        n_estimators=N_ESTIMATORS,
        distances=DISTANCE,
    )
    FEATURES_SPAN = []
    MAX_DEPTH = 8
    rsif.features_span = FEATURES_SPAN
    rsif.max_depth = MAX_DEPTH
    trees = rsif.create_trees()

    assert len(trees) == N_ESTIMATORS
    tree_mock.assert_has_calls(
        [
            call(
                distances=DISTANCE,
                max_depth=MAX_DEPTH,
                random_state=random_state,
                features_span=FEATURES_SPAN,
                pair_strategy="two_step",
            )
            for random_state in range(N_ESTIMATORS)
        ],
        any_order=True,
    )
