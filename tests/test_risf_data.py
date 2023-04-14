import pytest
import numpy as np
from risf.risf_data import RisfData, list_to_numpy
from risf.distance import TrainDistanceMixin, TestDistanceMixin
from unittest.mock import patch, Mock, call
import pandas as pd


def test_list_to_numpy():
    # The deal is that we give varying length vectors, but it can be any kind of object
    data_before = [np.array([10, 20, 50, 20, 30]),
                   np.array([2, 10]),
                   np.array([30, 20, 30, 220, 30, 20, 30, 20, 30, 20])]
    data = list_to_numpy(data_before)

    assert isinstance(data, np.ndarray)
    assert data.dtype == object

    # !Just to make sure objects are by no means changed during this conversion
    for i in range(len(data_before)):
        assert np.array_equal(data_before[i], data[i])


def test_validate_column_np_success():
    array = np.array([[1, 2], [3, 4], [4, 5]])
    validated = RisfData.validate_column(array)

    # In thi situation we basically do nothing and return just original variable
    assert id(array) == id(validated)


def test_validate_column_list_success():
    # !tbh I don't know how to mock this well.
    array = [object(), object(), object()]  # Any list of objects is good
    RisfData.validate_column(array)


def test_validate_column_pd_series_success():
    series = pd.Series([[10, 20], 39, 40])
    validated = RisfData.validate_column(series)

    assert isinstance(validated, np.ndarray)
    assert validated.dtype == series.dtype


def test_validate_column_bad_type():
    with pytest.raises(TypeError, match="If you don't provide data_transform function"):
        data = pd.DataFrame()
        RisfData.validate_column(data)


def test_distance_check_succesfull():
    X = [0, 1, 2, 3, 4]
    dist = Mock()
    RisfData.distance_check(X, dist)
    # distance between first two items should be calculated
    dist.assert_called_once_with(0, 1)


def test_distance_check_failure():
    X = [0, 1, 2, 3, 4]
    dist = Mock()
    dist.side_effect = Exception()

    with pytest.raises(ValueError, match="Cannot' calculate distance between two instances of a given column!"):
        RisfData.distance_check(X, dist)


def test_calculate_data_transform_no_transform():
    X = [0, 1, 2, 3, 4]
    transformed = RisfData.calculate_data_transform(X, None)
    assert id(X) == id(transformed)


def test_calculate_data_transform():
    X = [1, 2, 3, 4, 5]
    transform = Mock()
    transform.side_effect = lambda x: np.zeros(x)
    transformed = RisfData.calculate_data_transform(X, transform)

    expected_transform = [[0.],
                          [0., 0.],
                          [0., 0., 0.],
                          [0., 0., 0., 0.],
                          [0., 0., 0., 0., 0.]]

    for i in range(len(expected_transform)):
        assert (expected_transform[i] == transformed[i]).all()

    transform.assert_has_calls([call(1), call(2), call(3), call(4), call(5)])


def test_calculate_data_transform_assertion():
    X = [1, 2, 3, 4, 5]
    transform = Mock()
    transform.side_effect = Exception()

    with pytest.raises(ValueError, match="Cannot' calculate data transform!"):
        RisfData.calculate_data_transform(X, transform)


def test_update_metadata_distance_function():
    data = RisfData()
    dist_func = Mock()
    transform = Mock()
    name = None

    data.update_metadata(dist_func, transform, name)

    assert data.transforms[0] == transform
    # If we pass None as a name it should be automatically assigned to number of attrrs
    assert data.names[0] == "attr0"
    assert isinstance(data.distances[0], TrainDistanceMixin)
    assert data.distances[0].distance_func == dist_func

    name = "new_attr"

    data.update_metadata(dist_func, transform, name)

    assert data.transforms[1] == transform
    # If we pass None as a name it should be automatically assigned to number of attrrs
    assert data.names[1] == "new_attr"
    assert isinstance(data.distances[1], TrainDistanceMixin)
    assert data.distances[1].distance_func == dist_func


def test_update_metadata_distance_mixin():
    data = RisfData()
    dist = Mock(spec=TrainDistanceMixin)
    transform = Mock()
    name = None

    data.update_metadata(dist, transform, name)

    assert data.distances[0] == dist

    test_dist = Mock(spec=TestDistanceMixin)
    data.update_metadata(test_dist, transform, name)

    assert data.distances[1] == test_dist


@patch.object(RisfData, "calculate_data_transform", side_effect=lambda x, y: x)
@patch.object(RisfData, "validate_column", side_effect=lambda x: x)
@patch.object(RisfData, "distance_check")
@patch.object(RisfData, "update_metadata")
def test_add_data(mock_meta, mock_dist, mock_val, mock_trans):
    X = np.array([0, 0, 0])
    dist = Mock()
    transform = Mock()
    name = "name"
    data = RisfData()
    data.add_data(X, dist, transform, name)

    mock_trans.assert_called_once_with(X, transform)
    mock_val.assert_called_once_with(X)
    mock_dist.assert_called_once_with(X, dist)
    mock_meta.assert_called_once_with(dist, transform, name)

    # It is not possible to mock list.append()
    assert np.array_equal(data[0], X)


def test_precompute_distances():
    data = RisfData()
    data.append("data0")
    data.append("data1")
    data.distances = [Mock(), Mock()]
    DEFAULT_N_JOBS = 1
    data.precompute_distances()

    data.distances[0].precompute_distances.assert_called_once_with(
        data[0], n_jobs=DEFAULT_N_JOBS)
    data.distances[1].precompute_distances.assert_called_once_with(
        data[1], n_jobs=DEFAULT_N_JOBS)


def test_shape_check_success():
    data = RisfData()
    N_OBJECTS = 10
    X = np.random.randn(N_OBJECTS, 5)
    data.shape_check(X)

    assert data.shape == (N_OBJECTS, 1)

    data.shape_check(X)
    data.shape_check(X)
    assert data.shape == (N_OBJECTS, 3)  # Now we have added 3 columns


def test_shape_check_assert():
    data = RisfData()
    N_OBJECTS = 10
    X = np.random.randn(N_OBJECTS, 5)
    data.shape_check(X)

    with pytest.raises(ValueError, match="You newly added column"):
        # Now we try to add column with more objects than previously
        data.shape_check(np.random.randn(N_OBJECTS + 5, 5))
