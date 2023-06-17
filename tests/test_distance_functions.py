import numpy as np
import pytest

from risf.distance_functions import euclidean_projection


def euclidean_projection_full(X, p, q):
    X_p = X - p
    X_q = X - q
    return (X_q * X_q).sum(axis=1) - (X_p * X_p).sum(
        axis=1
    )  # Actually we can't use sqrt here!
    # If we used it then we will get same results for every point.


@pytest.fixture
def euclidean_p():
    return np.array([1]), np.array([-3, -3, -3]), np.array([2, 2])


@pytest.fixture
def euclidean_q():
    return np.array([3]), np.array([3, 3, 3]), np.array([0, -2])


def test_euclidean_single_vector_correct_results(euclidean_p, euclidean_q):
    X = (
        np.array([[-1]]),
        np.array([[0, 2, 2]]),
        np.array([[4, 4]]),
    )  # I assume that we will represent single vectors as matrices too
    correct = [
        [2],
        [-24],
        [24],
    ]  # I calculated this results on paper using equation P = x^t(p-q)
    results = []
    for x, p, q in zip(X, euclidean_p, euclidean_q):
        results.append(euclidean_projection(x, p, q))

    assert np.array_equal(correct, results)


def test_euclidean_matrix_correct_results(euclidean_p, euclidean_q):
    X = (
        np.array([[-1], [0], [2]]),
        np.array([[0, 2, 2], [1, 1, 1]]),
        np.array([[4, 4], [1, 1]]),
    )
    correct = [
        [2, 0, -4],
        [-24, -18],
        [24, 6],
    ]  # I calculated this results on paper using equation P = x^t(p-q)

    for i, (x, p, q) in enumerate(zip(X, euclidean_p, euclidean_q)):
        assert np.array_equal(correct[i], euclidean_projection(x, p, q))


def test_euclidean_preserve_1d_ordering():
    points = np.array([[-10], [-5], [0], [3], [3.5], [8]])
    order_original = [0, 1, 2, 3, 4, 5]
    # Lets check all pairs
    for i in range(points.shape[0]):
        for j in range(i + 1, points.shape[0]):
            p, q = points[i], points[j]
            proj = euclidean_projection(points, p, q)
            proj_full = euclidean_projection_full(points, p, q)
            order = np.argsort(proj)  # this sorts in ascending order
            order_full = np.argsort(proj_full)
            assert np.array_equal(order_original, order[::-1])
            assert np.array_equal(order_full, order)
