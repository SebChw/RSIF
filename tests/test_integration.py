import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score
from sklearn.ensemble import IsolationForest
from sklearn.datasets import load_wine

import pytest

from risf.forest import RandomIsolationSimilarityForest
from risf.risf_data import RisfData


@pytest.mark.integration
def test_result_on_dummy_data():
    X = np.array([[-1.1], [0.3], [0.5], [-1], [-0.9], [0.2], [0.1], [100]])
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


@pytest.fixture()
def train_data():
    data = np.load('data/numerical/01_breastw.npz',
                   allow_pickle=True)  # very simple dataset
    X, y = data['X'].astype(np.float32), data['y']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=True, stratify=y, random_state=23)

    return X_train, X_test, y_train, y_test


@pytest.mark.parametrize("n_jobs", [1, 2, -1])
@pytest.mark.integration
def test_metrics_on_small_dataset(train_data, n_jobs):
    """In this test we check if we didn't mess up anything, so that we get bad scores comparing to our first implementation
    We also check if agreement with ISF is quite high"""

    X_train, X_test, y_train, y_test = train_data

    isf = IsolationForest(random_state=0).fit(X_train)
    isf_pred = isf.predict(X_test)
    isf_pred_shifted = isf_pred.copy()
    isf_pred_shifted[isf_pred_shifted == 1] = 0
    isf_pred_shifted[isf_pred_shifted == -1] = 1

    risf = RandomIsolationSimilarityForest(random_state=0, n_jobs=n_jobs).fit(X_train)
    risf_pred = risf.predict(X_test)

    assert (((risf_pred == isf_pred_shifted).sum()) /
            isf_pred.shape[0]) == 0.96585365853658536585365853658537  # agreement

    assert precision_score(y_test, risf_pred) == 0.9571428571428572
    assert accuracy_score(y_test, risf_pred) == 0.9609756097560975
    assert recall_score(y_test, risf_pred) == 0.9305555555555556
    assert roc_auc_score(y_test, -1*risf.predict(X_test,
                         return_raw_scores=True)) == 0.9915413533834586


@ pytest.mark.integration
def test_result_on_dummy_data_given_y():
    data = np.load('data/numerical/01_breastw.npz',
                   allow_pickle=True)
    X, y = data['X'], data['y']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=True, random_state=23)

    risf = RandomIsolationSimilarityForest(
        random_state=0).fit(X_train, y_train)
    risf_pred = risf.predict(X_train)
    computedP = sum(risf_pred)/len(risf_pred)
    correctP = sum(y_train)/len(y_train)
    assert computedP == correctP


@pytest.mark.parametrize("n_jobs", [1, 2, -1])
@ pytest.mark.integration
def test_results_similarity_forest_imitation(train_data, n_jobs):
    X_train, X_test, y_train, y_test = train_data

    X_risf = RisfData()
    X_risf.add_data(X_train, dist=[lambda x, y: np.dot(x, y)])
    X_risf.precompute_distances(n_jobs=n_jobs)

    risf = RandomIsolationSimilarityForest(random_state=0, distance=X_risf.distances, n_jobs=n_jobs).fit(X_risf)

    X_test_risf = risf.transform([X_test], n_jobs=n_jobs)

    risf.decision_threshold_ = -0.35

    predictions = risf.predict(X_test_risf)

    assert accuracy_score(y_test, predictions) == 0.9317073170731708
    assert precision_score(y_test, predictions) == 0.8452380952380952
    assert recall_score(y_test, predictions) == 0.9861111111111112

    assert roc_auc_score(y_test, -1*risf.predict(X_test_risf,
                         return_raw_scores=True)) == 0.9754594820384294
