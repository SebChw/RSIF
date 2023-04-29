from unittest.mock import Mock, call, patch

import numpy as np
import pandas as pd
import pytest

from risf.distance import TestDistanceMixin, TrainDistanceMixin
from risf.risf_data import RisfData, list_to_numpy


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


def test_update_metadata():
    data = RisfData()
    transform = Mock()
    name = None

    data.update_metadata(transform, name)

    assert data.transforms[0] == transform
    # If we pass None as a name it should be automatically assigned to number of attrrs
    assert data.names[0] == "attr0"

    name = "new_attr"

    data.update_metadata(transform, name)

    assert data.transforms[1] == transform
    # If we pass None as a name it should be automatically assigned to number of attrrs
    assert data.names[1] == "new_attr"

@pytest.mark.parametrize("distances", [[Mock(), Mock()], [Mock(spec=TrainDistanceMixin), Mock(spec=TrainDistanceMixin), Mock(spec=TrainDistanceMixin)]])
@patch.object(RisfData, "distance_check", side_effect=lambda x, y: None)
def test_add_distances(mock_dist_check, distances):
    data = RisfData()
    X = [0, 1, 2, 3, 4]

    data.add_distances(X, distances)

    mock_dist_check.assert_has_calls([call(X, distance) for distance in distances])
    
    assert len(data.distances) == 1
    for distance in data.distances[0]:
        assert isinstance(distance, TrainDistanceMixin)


@patch.object(RisfData, "distance_check", side_effect=lambda x, y: None)
def test_add_distances_pickle(distance_check_mock):
    data = RisfData()
    X = [0, 1, 2, 3, 4]
    distances = ["tests/data/REDDIT-BINARY_DegreeDivergenceDist_train.pickle"]

    data.add_distances(X, distances)
    assert len(data.distances) == 1
    assert isinstance(data.distances[0][0], TrainDistanceMixin)


@patch.object(RisfData, "calculate_data_transform", side_effect=lambda x, y: x)
@patch.object(RisfData, "validate_column", side_effect=lambda x: x)
@patch.object(RisfData, "update_metadata")
@patch.object(RisfData, "add_distances")
def test_add_data(mock_add_dist, mock_meta, mock_val, mock_trans):
    X = np.array([0, 0, 0])
    dist = [Mock()]
    transform = Mock()
    name = "name"
    data = RisfData()
    data.add_data(X, dist, transform, name)

    mock_trans.assert_called_once_with(X, transform)
    mock_val.assert_called_once_with(X)
    mock_add_dist.assert_called_once_with(X, dist)
    mock_meta.assert_called_once_with(transform, name)

    # It is not possible to mock list.append()
    assert np.array_equal(data[0], X)

@pytest.fixture()
def precompute_data():
    data = RisfData()
    data.impute_missing_values = Mock()
    data.append("data0")
    data.append("data1")
    data.append("data2")
    data.distances = [[Mock(), Mock(), Mock()], [Mock(), Mock()], [Mock()]]
    for dist in data.distances:
        for d in dist:
            d.selected_objects = None
    return data


@patch("numpy.random.choice", return_value=np.array([0, 2]))
@pytest.mark.parametrize("num_of_selected_objects,expected_selected", [(None, np.array([0, 1, 2])), (2, np.array([0, 2]))])
def test_precompute_distances_train(mock_choice, num_of_selected_objects, expected_selected, precompute_data):
    DEFAULT_N_JOBS = 5
    precompute_data.num_of_selected_objects = num_of_selected_objects
    precompute_data.precompute_distances(n_jobs=DEFAULT_N_JOBS)

    for i, distances in enumerate(precompute_data.distances):
        for distance in distances:
            distance.precompute_distances.assert_called_once_with(
                X=precompute_data[i], X_test=None, n_jobs=DEFAULT_N_JOBS)
            
            if num_of_selected_objects is not None:
                assert np.array_equal(distance.selected_objects, expected_selected)
                # 5 is length of the word data1
                mock_choice.assert_called_with(5, num_of_selected_objects, replace=False, random_state=23)
            else:
                assert distance.selected_objects is None

    calls = []
    for distances in precompute_data.distances:
        for distance in distances:
            calls.append(call(distance))

    precompute_data.impute_missing_values.assert_has_calls(calls)


def test_precompute_distances_test(precompute_data):
    train_X = [[0,1,2], [2,3, 4], [1,2,2]]
    DEFAULT_N_JOBS = 5
    precompute_data.precompute_distances(train_data=train_X, n_jobs=DEFAULT_N_JOBS)

    for i, distances in enumerate(precompute_data.distances):
        for distance in distances:
            distance.precompute_distances.assert_called_once_with(
                X=train_X[i], X_test=precompute_data[i], n_jobs=DEFAULT_N_JOBS)

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


def test_impute_missing_values():
    data = RisfData()
    distance_obj = Mock()
    distance_obj.distance_matrix = np.array([
        [np.nan, 10, 20],
        [5, np.nan, -20],
        [0, 0, 0.0000001],
    ])

    data.impute_missing_values(distance_obj)

    assert np.array_equal(distance_obj.distance_matrix,
                          np.array([
                                    [20, 10, 20],
                                    [5, 20, -20],
                                    [0, 0, 0.0000001],
                                    ]))
