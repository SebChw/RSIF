import risf.splitting as splitting
import numpy as np
import pytest
from unittest.mock import patch, Mock
from risf.distance import TrainDistanceMixin, DistanceMixin


@pytest.mark.parametrize(
    "bad_X",
    [
        np.array([[1, 2, 3, 4, 5]]),
        np.array(
            [[[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]], [
                [1, 2, 3, 4, 5], [2, 3, 4, 5, 6]]]
        ),
    ],
)  # 2dim nor 3 dim data is not allowed
def test_project_badX(bad_X):
    with pytest.raises(ValueError, match="1 dimensional"):
        splitting.project(bad_X, 0, 1, "euclidean")


@pytest.mark.parametrize(
    "bad_Oi", [[10, 20], np.array([20, 30, 40]), object()]
)  # arrays, lists, objects not allowed
def test_project_badOi(bad_Oi):
    with pytest.raises(TypeError, match="Unsupported Pair object data"):
        splitting.project(np.array([0, 1, 2, 3]), bad_Oi, 1, "euclidean")


# No integers, no functions!
@pytest.mark.parametrize("bad_dist", [1, lambda x: x])
def test_project_bad_dist(bad_dist):
    with pytest.raises(TypeError, match="Unsupported projection type"):
        splitting.project(np.array([0, 1, 2, 3]), 0, 1, bad_dist)


@patch("risf.projection.make_projection")
def test_project_correct_call_string_distance(projection_mock):
    splitting.project(
        np.array([0, 1, 2, 3]), 0, 1, "euclidean"
    )  # Since input is numeric this X should become a 2dimensional array and Oi and Oj a one dimensional vectors
    X, Oi, Oj, dist = projection_mock.call_args_list[0][0]
    assert np.array_equal(np.array([[0], [1], [2], [3]]), X)
    assert np.array_equal(np.array([0]), Oi)
    assert np.array_equal(np.array([1]), Oj)
    assert dist == "euclidean"


@patch.object(TrainDistanceMixin, "project")
def test_project_correct_call_string_distance(project_mock):
    dist_mock = TrainDistanceMixin(None)
    splitting.project(
        np.array([0, 1, 2, 3]), 0, 1, dist_mock
    )  # Since input is numeric this X should become a 2dimensional array and Oi and Oj a one dimensional vectors
    X, *objects = project_mock.call_args_list[0][0]
    assert np.array_equal(np.array([0, 1, 2, 3]), X)
    assert objects == [0, 1]


def test_get_features_with_unique_values_distance_mixin():
    # In the node we have object 0,2 and 3
    X = np.array([
        [0, 0],
        [2, 2],
        [3, 3]
    ])
    dist1 = Mock(spec=DistanceMixin)
    #! In reality this matrix should be entirely filled with zeros for obj 0,2 and 3 but I want to make sure it looks on the column
    dist1.distance_matrix = np.array([
        [0, 5, 0, 0],
        [5, 0, 2, 3],
        [0, 2, 0, 1],
        [0, 3, 1, 0]
    ])
    #! This column has more than 1 unique values
    dist2 = Mock(spec=DistanceMixin)
    dist2.distance_matrix = np.array([
        [0, 1, 2, 3],
        [1, 0, 2, 3],
        [2, 2, 0, 3],
        [3, 3, 3, 0]
    ])

    unique_columns = splitting.get_features_with_unique_values(X, [
                                                               dist1, dist2])

    # at the end I expect just the first column to be used later on
    assert unique_columns == [1]


def test_get_features_with_unique_values_numerical():
    X = np.zeros((5, 3))
    X[:, 0] = np.random.randn(5)
    X[:, 2] = np.array([0, 1, 1, 1, 1])

    unique_columns = splitting.get_features_with_unique_values(
        X, [None, None, None])
    assert unique_columns == [0, 2]
