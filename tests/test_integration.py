import numpy as np
from risf.forest import RandomIsolationSimilarityForest
from sklearn.datasets import load_wine
import pytest


@pytest.mark.integration
def test_result_on_dummy_data():
    X = [[-1.1], [0.3], [0.5], [-1], [-0.9], [0.2], [0.1], [100]]
    clf = RandomIsolationSimilarityForest(random_state=0).fit(X)
    assert np.array_equal(
        clf.predict(np.array([[0.1], [0], [90]])), np.array([0, 0, 1])
    )


@pytest.mark.integration
def test_pipeline_sucess_on_bigger_dataset():
    """In this test we just check if it suceeds to fit a tree and return any scores"""
    # 13 numerical attributes anb 178 instances
    wine_data = load_wine()["data"]
    clf = RandomIsolationSimilarityForest(
        random_state=0, contamination=0.8).fit(wine_data)
    predictions = clf.predict(np.ones((2, 13)))
    assert predictions.size == 2
