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


def test_generate_indices_no_selected_obj(train_distance_mixin):
    N_ALL_OBJECTS = 5

    indices = train_distance_mixin._generate_indices(
        N_ALL_OBJECTS)

    assert (indices == [[0, 1], [0, 2], [0, 3], [0, 4], [
            1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]).all()


def test_generate_indices_selected_obj(train_distance_mixin):
    N_ALL_OBJECTS = 5
    train_distance_mixin.selected_objects = [1, 3]

    indices = train_distance_mixin._generate_indices(
        N_ALL_OBJECTS)

    assert (indices == [[1, 0], [1, 2], [1, 3], [
            1, 4], [3, 0], [3, 2], [3, 4]]).all()


def test_generate_indices_splits(train_distance_mixin):
    N_PAIRS = 99
    PAIRS_OF_INDICES = [[0, 0] for _ in range(N_PAIRS)]

    N_JOBS = 1
    assert train_distance_mixin._generate_indices_splits(
        PAIRS_OF_INDICES, N_JOBS) == [(0, 99)]

    N_JOBS = 2
    # End intervals are exluded during selection so it's good that they overlap
    # Moreover if we give too big index during slicing it won't raise error but give as many elements it can
    assert train_distance_mixin._generate_indices_splits(
        PAIRS_OF_INDICES, N_JOBS) == [(0, 50), (50, 100)]

    N_JOBS = 3
    assert train_distance_mixin._generate_indices_splits(
        PAIRS_OF_INDICES, N_JOBS) == [(0, 33), (33, 66), (66, 99)]


def test_precompute_train_distance_everything_selected(x_train_data, train_distance_mixin):
    train_distance_mixin.precompute_distances(x_train_data)
    # On the diagonal distance shouldn't be calculated. Matrix must be symmetric
    # Indices indicate when distance was calculated between which objects.
    assert (train_distance_mixin.distance_matrix == [[0,  1,  2,  3,  4,],
                                                     [1,  0,  5,  6,  7,],
                                                     [2,  5,  0,  8,  9,],
                                                     [3,  6,  8,  0, 10,],
                                                     [4,  7,  9, 10,  0,]]).all()


def test_precompute_train_distance_custom_selection(x_train_data, train_distance_mixin):
    train_distance_mixin.selected_objects = [2, 4]

    train_distance_mixin.precompute_distances(
        x_train_data)

    # Now only 2 and 4th rows and columns should be filled as only 2 and 4 obj can create a pair.
    assert (train_distance_mixin.distance_matrix == [[0,  0,  1,  0,  5,],
                                                     [0,  0,  2,  0,  6,],
                                                     [1,  2,  0,  3,  4,],
                                                     [0,  0,  3,  0,  7,],
                                                     [5,  6,  4,  7,  0,]]).all()


def test_precompute_test_distance(x_train_data, train_distance_mixin):
    # Assume that during training only 2 objects were used
    used_points = [2, 4]

    # We have 3 object to be predicted
    N_TEST_OBJECTS = 3
    x_test_data = np.empty(N_TEST_OBJECTS, dtype=object)
    x_test_data[:] = object()

    test_distance_mixin = TestDistanceMixin(
        train_distance_mixin.distance, used_points)
    test_distance_mixin.precompute_distances(x_train_data, x_test_data)

    # matrix is always N_TRAIN_OBJECTS X N_TEST_OBJECTS
    # distance should be calculated between all test_objects and points used during training.
    # Order of calculations should go as in used_points
    assert (test_distance_mixin.distance_matrix == [[0, 0, 0],
                                                    [0, 0, 0],
                                                    [1, 2, 3],
                                                    [0, 0, 0],
                                                    [4, 5, 6]]).all()
