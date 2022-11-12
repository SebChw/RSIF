import pytest
from unittest.mock import patch, MagicMock
from risf.forest import RandomIsolationSimilarityForest
from risf.tree import RandomIsolationSimilarityTree
import numpy as np


@patch.object(
    RandomIsolationSimilarityForest,
    "calculate_mean_path_lengths",
    return_value=np.array([2, 5, 10]),
)
@patch("risf.utils.measures._average_path_length", return_value=5)
def test_score_samples(avg_p_length, risf_mean_length):
    risf = RandomIsolationSimilarityForest()
    risf.subsample_size = (
        10  # Needed for method to run, method using it is mocked anyway
    )
    results = risf.score_samples(
        np.ones((3, 3))
    )  # It should calculate -2^(-2/5), -2^(-5/5), -2^(-10/5)

    assert (results == [-0.757858283255199, -0.5, -0.25]).all()


@patch("sklearn.utils.validation.check_array", return_value=np.array([1, 5]))
@patch("sklearn.utils.validation.check_is_fitted")
@patch.object(
    RandomIsolationSimilarityForest,
    "_validate_X_predict",
    return_value=np.array([1, 5]),
)
@patch.object(
    RandomIsolationSimilarityForest,
    "score_samples",
    return_value=np.array([0, -1, -0.5]),
)  # decision function is calculated based on return of this function
def test_decision_function(score_samples, validate_x, is_fitted, check_array):
    risf = RandomIsolationSimilarityForest()
    risf.offset_ = -0.5
    result = risf.decision_function(
        np.array([1, 5, 3])
    )  # this should check all needed things and then calculate 0 - (-0.5), -1 -(-0.5), -0.5 -(-0.5)

    assert validate_x.called
    assert is_fitted.called
    assert check_array.called
    assert score_samples.called
    assert np.allclose(result, [0.5, -0.5, 0])

    # test with different offset
    risf.offset_ = -0.7
    result = risf.decision_function(np.array([1, 5, 3]))

    assert validate_x.called
    assert is_fitted.called
    assert check_array.called
    assert score_samples.called
    assert np.allclose(
        result, [0.7, -0.3, 0.2]
    )  # bigger offset it is harder to become an outlier scores are higher


@patch.object(
    RandomIsolationSimilarityForest,
    "decision_function",
    return_value=np.array([-0.5, 0.01, 0, -0.001, 0.75]),
)  # 0 shouldn't be outlier
def test_predict(decision_function):
    risf = RandomIsolationSimilarityForest()
    predictions = risf.predict(
        np.ones((5, 3))
    )  # we pass 5 objects with 3 attributes each

    assert decision_function.called
    assert np.array_equal(predictions, [1, 0, 0, 1, 0])


def test_calculate_mean_path_lengths():
    risf = RandomIsolationSimilarityForest()
    risf.trees_ = []
    for i in range(
        1, 11
    ):  # lets add 10 trees to the forest and mock their path_length function
        rist = RandomIsolationSimilarityTree(distance="euclidean")
        rist.path_lengths_ = MagicMock(
            return_value=i * np.array([1, 2, 3, 4, 5])
        )  # this is the situation when we gave 5 object to be tested
        # and path lengths were equal to i * [1,2,3,4,5]
        risf.trees_.append(rist)

    mean_path_lenghts = risf.calculate_mean_path_lengths([[1], [2], [3], [4], [5]])
    assert np.allclose(mean_path_lenghts, [5.5, 11, 16.5, 22, 27.5])
