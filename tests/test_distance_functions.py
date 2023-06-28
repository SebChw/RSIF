import numpy as np
import pytest

from risf.distance_functions import (
    cosine_projection,
    cosine_sim,
    euclidean_projection,
    jaccard_projection,
    manhattan_projection,
)


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


def test_cosine_projection():
    X = np.array([[1, 1, 1], [0.5, 0.3, 0.2], [-1, -1, -1], [-0.5, -0.3, -0.2]])

    p = np.array([1, 1, 1])
    q = np.array([-1, -1, -1])

    sim_p = cosine_sim(X, p)
    sim_q = cosine_sim(X, q)

    assert np.allclose(
        sim_p, np.array([1.0, 0.93658581, -1.0, -0.93658581], dtype=np.float64)
    )
    assert np.allclose(
        sim_q, np.array([-1.0, -0.93658581, 1.0, 0.93658581], dtype=np.float64)
    )
    projection = cosine_projection(X, p, q)
    assert np.allclose(
        projection, np.array([-2.0, -1.87317162, 2.0, 1.87317162], dtype=np.float64)
    )


def test_manhattan_projection():
    X = np.array([[1, 1, 1], [0.5, 0.3, 0.2], [-1, -1, -1], [-0.5, -0.3, -0.2]])

    p = np.array([1, 1, 1])
    q = np.array([0, 0.5, -0.1])

    projection = manhattan_projection(X, p, q)

    assert np.allclose(projection, np.array([-2.6, 1, 2.6, 2.6], dtype=np.float64))


def test_jaccard_projection():
    X = np.array([[0, 1, 1], [0, 1, 0], [1, 1, 1], [0, 0, 0]], dtype=bool)

    p = np.array([0, 1, 0], dtype=bool)  # distances are: 0.5, 0, 0.66, 1
    q = np.array([1, 1, 1], dtype=bool)  # distances are: 0.33, 0.66, 0, 1

    projection = jaccard_projection(X, p, q)

    assert np.allclose(
        projection,
        np.array([0.16666667, -0.66666667, 0.66666667, 0.0], dtype=np.float64),
    )
