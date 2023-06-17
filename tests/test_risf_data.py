from collections import namedtuple
from unittest.mock import MagicMock, Mock, call, patch

import numpy as np
import pandas as pd
import pytest

from risf.distance import TrainDistanceMixin
from risf.forest import RandomIsolationSimilarityForest
from risf.risf_data import RisfData, list_to_numpy


def test_list_to_numpy():
    # The deal is that we give varying length vectors, but it can be any kind of object
    data_before = [
        np.array([10, 20, 50, 20, 30]),
        np.array([2, 10]),
        np.array([30, 20, 30, 220, 30, 20, 30, 20, 30, 20]),
    ]
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
    with pytest.raises(TypeError, match="Given data must be an instance of"):
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

    with pytest.raises(
        ValueError,
        match="Cannot' calculate distance between two instances of a given column!",
    ):
        RisfData.distance_check(X, dist)


def test_update_metadata():
    data = RisfData()
    name = None

    data.update_metadata(name)

    # If we pass None as a name it should be automatically assigned to number of attrrs
    assert data.names[0] == "attr0"

    name = "new_attr"

    data.update_metadata(name)

    # If we pass None as a name it should be automatically assigned to number of attrrs
    assert data.names[1] == "new_attr"


@pytest.mark.parametrize(
    "distances",
    [
        [Mock(), Mock()],
        [
            Mock(spec=TrainDistanceMixin),
            Mock(spec=TrainDistanceMixin),
            Mock(spec=TrainDistanceMixin),
        ],
    ],
)
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


@patch.object(RisfData, "validate_column", side_effect=lambda x: x)
@patch.object(RisfData, "update_metadata")
@patch.object(RisfData, "add_distances")
def test_add_data(mock_add_dist, mock_meta, mock_val):
    X = np.array([0, 0, 0])
    dist = [Mock()]
    name = "name"
    data = RisfData()
    data.add_data(X, dist, name)

    mock_val.assert_called_once_with(X)
    mock_add_dist.assert_called_once_with(X, dist)
    mock_meta.assert_called_once_with(name)

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


@pytest.mark.parametrize(
    "num_of_selected_objects,expected_selected",
    [(None, np.array([0, 1, 2])), (2, np.array([0, 2]))],
)
def test_precompute_distances_train(
    num_of_selected_objects, expected_selected, precompute_data
):
    DEFAULT_N_JOBS = 5
    precompute_data.num_of_selected_objects = num_of_selected_objects

    random_gen_mock = Mock()
    random_gen_mock.choice = Mock(return_value=np.array([0, 2]))
    precompute_data.random_gen = random_gen_mock

    precompute_data.precompute_distances(n_jobs=DEFAULT_N_JOBS)

    for i, distances in enumerate(precompute_data.distances):
        for distance in distances:
            distance.precompute_distances.assert_called_once_with(
                X=precompute_data[i], X_test=None, n_jobs=DEFAULT_N_JOBS
            )

            if num_of_selected_objects is not None:
                assert np.array_equal(distance.selected_objects, expected_selected)
                # 5 is length of the word data1
                random_gen_mock.choice.assert_called_with(
                    5, num_of_selected_objects, replace=False
                )
            else:
                assert distance.selected_objects is None

    calls = []
    for distances in precompute_data.distances:
        for distance in distances:
            calls.append(call(distance))

    precompute_data.impute_missing_values.assert_has_calls(calls)


def test_precompute_distances_test(precompute_data):
    train_X = [[0, 1, 2], [2, 3, 4], [1, 2, 2]]
    DEFAULT_N_JOBS = 5
    precompute_data.precompute_distances(train_data=train_X, n_jobs=DEFAULT_N_JOBS)

    for i, distances in enumerate(precompute_data.distances):
        for distance in distances:
            distance.precompute_distances.assert_called_once_with(
                X=train_X[i], X_test=precompute_data[i], n_jobs=DEFAULT_N_JOBS
            )


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
    distance_obj.distance_matrix = np.array(
        [
            [np.nan, 10, 20],
            [5, np.nan, -20],
            [0, 0, 0.0000001],
        ]
    )

    data.impute_missing_values(distance_obj)

    assert np.array_equal(
        distance_obj.distance_matrix,
        np.array(
            [
                [20, 10, 20],
                [5, 20, -20],
                [0, 0, 0.0000001],
            ]
        ),
    )


@patch(
    "risf.risf_data.RisfData",
    spec=RisfData,
    **{"__getitem__.side_effect": lambda x: x + 10}
)
@patch("risf.risf_data.TestDistanceMixin")
@patch.object(
    RandomIsolationSimilarityForest, "get_used_points", return_value=[1, 2, 3]
)
def test_transform(get_used_points_mock, test_dist_mix_mock, risf_data_mock):
    list_of_X = [[object(), object()], [10, 20]]
    risf = RandomIsolationSimilarityForest()
    risf_data = RisfData()
    risf_data.append(list_of_X[0])
    risf_data.append(list_of_X[1])
    risf_data.names = ["name1", "name2"]
    Distance = namedtuple("Distance", ["distance_func"])
    risf_data.distances = [[Distance(1)], [Distance(2)]]
    risf.X = risf_data
    DEFAULT_N_JOBS = 1

    test_data = risf_data.transform(list_of_X, forest=risf, n_jobs=DEFAULT_N_JOBS)

    assert isinstance(test_data, RisfData)
    test_dist_mix_mock.assert_has_calls([call(1, [1, 2, 3]), call(2, [1, 2, 3])])

    risf_data_mock.assert_has_calls(
        [
            call().add_data(list_of_X[0], [test_dist_mix_mock()], "name1"),
            call().add_data(list_of_X[1], [test_dist_mix_mock()], "name2"),
            call().precompute_distances(train_data=risf.X, n_jobs=DEFAULT_N_JOBS),
        ],
        any_order=True,
    )
