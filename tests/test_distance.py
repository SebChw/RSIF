import pytest
import numpy as np
from risf.distance import TrainDistanceMixin, TestDistanceMixin, DistanceMixin
from unittest.mock import patch, Mock


def test_project():
    # They both inherit project from DistanceMixin
    distance_mixin = TrainDistanceMixin(Mock())
    distance_mixin.distance_matrix = np.array([
        [0, 2, 1],
        [2, 0, 5],
        [1, 5, 0]
    ])
    # it should be dist(1,0) - dist(2,0) => 2 - 1
    assert distance_mixin.project(0, 1, 2) == 1


def test_get_all_objects_ids_with_selected_at_front():
    distance_mixin = TrainDistanceMixin(distance=None)

    # Case when someone passes selected object
    assert distance_mixin.get_all_objects_ids_with_selected_at_front(
        10, [2, 5, 9]) == [2, 5, 9, 0, 1, 3, 4, 6, 7, 8]
    # Default case when we want to calculate distance between everything
    assert distance_mixin.get_all_objects_ids_with_selected_at_front(
        10, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


class MockDist():
    # just to mock distance function. Thanks to this we also see in what order distances were calculated
    def __init__(self):
        self.i = 0

    def __call__(self, x, y):
        self.i += 1
        return self.i


@pytest.fixture
def x_train_data():
    N_TRAIN_OBJECTS = 5
    # To test the most general case everything is just an object
    X = np.empty(N_TRAIN_OBJECTS, dtype=object)
    X[:] = object()
    return X


@pytest.fixture
def train_distance_mixin():
    return TrainDistanceMixin(distance=MockDist())


def test_precompute_train_distance_everything_selected(x_train_data, train_distance_mixin):
    OBJ_INDICES = list(range(x_train_data.shape[0]))

    with patch.object(train_distance_mixin, 'get_all_objects_ids_with_selected_at_front',
                      return_value=OBJ_INDICES) as mock:
        train_distance_mixin.precompute_distances(x_train_data)

    # On the diagonal distance shouldn't be calculated. Matrix must be symmetric
    # Indices indicate when distance was calculated between which objects.
    mock.assert_called_once_with(x_train_data.shape[0], OBJ_INDICES)
    assert (train_distance_mixin.distance_matrix == [[0,  1,  2,  3,  4,],
                                                     [1,  0,  5,  6,  7,],
                                                     [2,  5,  0,  8,  9,],
                                                     [3,  6,  8,  0, 10,],
                                                     [4,  7,  9, 10,  0,]]).all()


def test_precompute_train_distance_custom_selection(x_train_data, train_distance_mixin):
    OBJ_INDICES = [2, 4, 0, 1, 3]  # selected object are 2 and 4
    SELECTED_OBJ = [2, 4]

    with patch.object(train_distance_mixin, 'get_all_objects_ids_with_selected_at_front',
                      return_value=OBJ_INDICES) as mock:
        train_distance_mixin.precompute_distances(
            x_train_data, selected_objects=SELECTED_OBJ)

    # On the diagonal distance shouldn't be calculated. Matrix must be symmetric
    # Now only 2 and 4th rows and columns should be filled as only 2 and 4 obj can create a pair.
    # Order of calculations now is also different to recover it iterate over [2,4,0,1,3]
    mock.assert_called_once_with(x_train_data.shape[0], SELECTED_OBJ)
    assert (train_distance_mixin.distance_matrix == [[0,  0,  2,  0,  5,],
                                                     [0,  0,  3,  0,  6,],
                                                     [2,  3,  0,  4,  1,],
                                                     [0,  0,  4,  0,  7,],
                                                     [5,  6,  1,  7,  0,]]).all()


def test_precompute_test_distance(x_train_data, train_distance_mixin):
    # Assume that during training only 2 objects were used
    used_points = [2, 4]

    # We have 3 object to be predicted
    N_TEST_OBJECTS = 3
    x_test_data = np.empty(N_TEST_OBJECTS, dtype=object)
    x_test_data[:] = object()

    test_distance_mixin = TestDistanceMixin(train_distance_mixin, used_points)
    test_distance_mixin.precompute_distances(x_train_data, x_test_data)

    # matrix is always N_TRAIN_OBJECTS X N_TEST_OBJECTS
    # distance should be calculated between all test_objects and points used during training.
    # Order of calculations should go as in used_points
    assert (test_distance_mixin.distance_matrix == [[0, 0, 0],
                                                    [0, 0, 0],
                                                    [1, 2, 3],
                                                    [0, 0, 0],
                                                    [4, 5, 6]]).all()
