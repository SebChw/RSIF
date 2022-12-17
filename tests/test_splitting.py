import risf.splitting as splitting
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from risf.distance import DistanceMixin


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


@patch.object(DistanceMixin, "project")
def test_project_correct_call_string_distance(project_mock):
    dist_mock = DistanceMixin(None)
    splitting.project(
        np.array([0, 1, 2, 3]), 0, 1, dist_mock
    )  # Since input is numeric this X should become a 2dimensional array and Oi and Oj a one dimensional vectors
    X, *objects = project_mock.call_args_list[0][0]
    assert np.array_equal(np.array([0, 1, 2, 3]), X)
    assert objects == [0, 1]
