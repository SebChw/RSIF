from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from risf.distance import DistanceMixin
# It's hard to mock it's functionalities here
from risf.risf_data import RisfData
from risf.utils.validation import (check_distance, check_max_samples,
                                   check_random_state, prepare_X)


def test_prepare_X_risf_data():
    data = RisfData()  # we use only it's inharent list functionalities not RisfData
    # each of 4 columns is a 2 element vector, we have 5 objects in dataset
    for i in range(4):
        data.append(np.random.randn(5, 2))

    data_for_forest = prepare_X(data)

    # As a result we expect object's indices
    correct_array = np.array([
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        [2, 2, 2, 2],
        [3, 3, 3, 3],
        [4, 4, 4, 4]
    ])
    assert np.array_equal(data_for_forest, correct_array)


def test_prepare_X_data_frame():
    X = prepare_X(pd.DataFrame([[0, 1, 2], [2, 3, 4]]))
    assert isinstance(X, np.ndarray)


@patch("risf.utils.validation.check_array")
def test_prepare_X_other_type(mock_check_array):
    X = np.array([[0, 1], [2, 3]])
    prepare_X(X)
    mock_check_array.assert_called_once_with(X)


def test_prepare_X_1d_vector():
    X = np.array([0, 1, 2, 3])
    X = prepare_X(X)
    assert X.shape == (4, 1)


def test_prepare_X_bad_type():
    X = "bad choice"
    with pytest.raises(TypeError, match="Unsupported data type: You can pass only one of"):
        prepare_X(X)


def test_check_max_samples_auto_more_than_256():
    assert check_max_samples(
        "auto", np.random.randn(1000, 5)) == 256


def test_check_max_samples_auto_less_than_256():
    assert check_max_samples(
        "auto", np.random.randn(100, 5)) == 100


def test_check_max_sample_float():
    assert check_max_samples(0.5, np.random.randn(
        509, 5)) == 254  # floor should be applied


def test_check_max_sample_float_bad():
    with pytest.raises(ValueError, match="is a float"):
        check_max_samples(1.5, np.random.randn(500, 5))


def test_check_max_sample_int():
    assert check_max_samples(399, np.random.randn(
        509, 5)) == 399  # floor should be applied


def test_check_max_sample_int_bad():
    with pytest.raises(ValueError, match="is an int"):
        check_max_samples(1000, np.random.randn(500, 5))


def test_check_max_samples_bad_max_samples():
    with pytest.raises(TypeError, match="max_samples should be"):
        check_max_samples("bad", np.random.randn(500, 5))


def test_check_random_state_none():
    assert isinstance(check_random_state(None), np.random.mtrand.RandomState)


@patch('numpy.random.RandomState')
def test_check_random_state_int(r_state_mock):
    check_random_state(10)
    r_state_mock.assert_called_once_with(10)


def test_check_random_state_random_state():
    r_state = np.random.RandomState()
    r_state2 = check_random_state(r_state)
    assert r_state is r_state2


def test_check_random_state_assertion():
    with pytest.raises(TypeError, match="Unsupported type. Only None, int and np.random.mtrand.RandomState are supported."):
        check_random_state("bad")


def test_check_distance_list():
    distances = [[1, 2], 2, 3, [4, 5, 6], [7]]
    distances2 = check_distance(distances, 5)
    assert distances2 == [[1, 2], [2], [3], [4, 5, 6], [7]]


def test_check_distance_str():
    dist = "euclidean"
    distances = check_distance(dist, 5)
    assert distances == [["euclidean"], ["euclidean"], ["euclidean"], ["euclidean"], ["euclidean"]]


def test_check_distance_assertion():
    with pytest.raises(TypeError, match="Unsupported distance type. Only list or str supported"):
        check_distance(123, 5)

def test_check_distance_assertion_mismatch():
    with pytest.raises(ValueError, match="If you provide a list of distances you must give one distance for each feature"):
        check_distance([1, 2], 5)