import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.datasets import load_wine

import pytest

from risf.forest import RandomIsolationSimilarityForest


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


@pytest.mark.integration
def test_metrics_on_small_dataset():
    """In this test we check if we didn't mess up anything, so that we get bad scores comparing to our first implementation
    We also check if agreement with ISF is quite high"""
    data = np.load('data/numerical/01_breastw.npz',
                   allow_pickle=True)  # very simple dataset
    X, y = data['X'], data['y']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=True, stratify=y, random_state=23)

    isf = IsolationForest(random_state=0).fit(X_train)
    isf_pred = isf.predict(X_test)
    isf_pred_shifted = isf_pred.copy()
    isf_pred_shifted[isf_pred_shifted == 1] = 0
    isf_pred_shifted[isf_pred_shifted == -1] = 1

    risf = RandomIsolationSimilarityForest(random_state=0).fit(X_train)
    risf_pred = risf.predict(X_test)

    # I got better result on the first correct implementation but let's save some small margin
    assert (((risf_pred == isf_pred_shifted).sum()) /
            isf_pred.shape[0]) > 0.9  # agreement
    assert precision_score(y_test, risf_pred) > 0.95
    assert accuracy_score(y_test, risf_pred) > 0.9
    assert recall_score(y_test, risf_pred) > 0.85
