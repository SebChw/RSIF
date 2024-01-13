import pickle
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import load_wine
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from rsif.distance import (
    SelectiveDistance,
    TestDistanceMixin,
    TrainDistanceMixin,
    split_distance_mixin,
)
from rsif.distance_functions import euclidean_projection
from rsif.forest import RandomSimilarityIsolationForest
from rsif.rsif_data import RsifData


@pytest.mark.integration
def test_result_on_dummy_data():
    X = np.array([[-1.1], [0.3], [0.5], [-1], [-0.9], [0.2], [0.1], [100]])
    clf = RandomSimilarityIsolationForest(
        random_state=0, distances=[SelectiveDistance(euclidean_projection, 1, 1)]
    ).fit(X)
    assert np.array_equal(
        clf.predict(np.array([[0.1], [0], [90]])), np.array([0, 0, 1])
    )


@pytest.mark.integration
def test_pipeline_sucess_on_bigger_dataset():
    """In this test we just check if it suceeds to fit a tree and return any scores"""
    # 13 numerical attributes anb 178 instances
    wine_data = load_wine()["data"]
    clf = RandomSimilarityIsolationForest(random_state=0, contamination=0.8).fit(
        wine_data
    )
    predictions = clf.predict(np.ones((2, 13)))
    assert predictions.size == 2


@pytest.fixture()
def train_data():
    data = np.load(
        "tests/data/01_breastw.npz", allow_pickle=True
    )  # very simple dataset
    X, y = data["X"].astype(np.float32), data["y"]
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=True, stratify=y, random_state=23
    )

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

    rsif = RandomSimilarityIsolationForest(random_state=0, n_jobs=n_jobs).fit(X_train)
    rsif_pred = rsif.predict(X_test)

    assert (
        ((rsif_pred == isf_pred_shifted).sum()) / isf_pred.shape[0]
    ) == 182 / 205  # agreement

    assert precision_score(y_test, rsif_pred) == 0.9629629629629629
    assert accuracy_score(y_test, rsif_pred) == 0.8926829268292683
    assert recall_score(y_test, rsif_pred) == 0.7222222222222222
    assert (
        roc_auc_score(y_test, -1 * rsif.predict(X_test, return_raw_scores=True))
        == 0.9820384294068504
    )


@pytest.mark.integration
def test_result_on_dummy_data_given_y(train_data):
    X_train, X_test, y_train, y_test = train_data

    rsif = RandomSimilarityIsolationForest(random_state=0).fit(X_train, y_train)
    rsif_pred = rsif.predict(X_train)
    computedP = sum(rsif_pred) / len(rsif_pred)
    correctP = sum(y_train) / len(y_train)
    assert np.isclose(computedP, correctP, atol=0.02)


@pytest.mark.parametrize("n_jobs", [1, 2, -1])
@pytest.mark.integration
def test_results_similarity_forest_imitation(train_data, n_jobs):
    X_train, X_test, y_train, y_test = train_data

    rsif = RandomSimilarityIsolationForest(
        random_state=0,
        distances=[
            SelectiveDistance(euclidean_projection, X_train.shape[1], X_train.shape[1])
        ],
        n_jobs=n_jobs,
    ).fit(X_train)

    answer = roc_auc_score(y_test, -1 * rsif.predict(X_test, return_raw_scores=True))
    assert np.floor(answer * 1000) / 1000 == 0.964


def distance(x, y):
    return np.dot(x, y)


@pytest.mark.parametrize("n_jobs", [1, 2, -1])
@pytest.mark.integration
def test_with_precalculated_distances(train_data, n_jobs):
    """Same as test above but uses precalculated distances"""
    TRAIN_DIST_PATH = "train.pickle"
    TEST_DIST_PATH = "test.pickle"

    X_train, X_test, y_train, y_test = train_data

    train_distance = TrainDistanceMixin(distance)
    train_distance.precompute_distances(X_train, n_jobs=n_jobs)
    pickle.dump(train_distance, open(TRAIN_DIST_PATH, "wb"))

    test_distance = TestDistanceMixin(
        distance, selected_objects=np.arange(X_train.shape[0])
    )
    test_distance.precompute_distances(X_train, X_test=X_test, n_jobs=n_jobs)
    pickle.dump(test_distance, open(TEST_DIST_PATH, "wb"))

    X_rsif = RsifData()
    X_rsif.add_data(
        np.arange(X_train.shape[0]).reshape(-1, 1), dist=[TRAIN_DIST_PATH]
    )  # I pass indices not full vectors

    rsif = RandomSimilarityIsolationForest(
        random_state=0, distances=X_rsif.distances, n_jobs=n_jobs
    ).fit(X_rsif)

    X_test_rsif = X_rsif.transform(
        [np.arange(X_test.shape[0]).reshape(-1, 1)],
        forest=rsif,
        precomputed_distances=[[TEST_DIST_PATH]],
        n_jobs=n_jobs,
    )

    assert (
        roc_auc_score(y_test, -1 * rsif.predict(X_test_rsif, return_raw_scores=True))
        == 0.9707602339181286
    )

    Path(TRAIN_DIST_PATH).unlink()
    Path(TEST_DIST_PATH).unlink()


def test_with_splitted_distances(
    train_data,
):
    X_train, X_test, y_train, y_test = train_data

    split_point = int(X_train.shape[0])

    X = np.concatenate([X_train, X_test])

    whole_distance = TrainDistanceMixin(distance)
    whole_distance.precompute_distances(X)

    indices = np.arange(X.shape[0])
    train_indices = indices[:split_point]
    np.random.shuffle(train_indices)  # Just to check if it works with shuffled indices

    train_distance, test_distance = split_distance_mixin(whole_distance, train_indices)

    X_rsif = RsifData()
    X_rsif.add_data(
        np.arange(X_train.shape[0]).reshape(-1, 1), dist=[train_distance]
    )  # I pass indices not full vectors

    rsif = RandomSimilarityIsolationForest(
        random_state=0, distances=X_rsif.distances, n_jobs=-1
    ).fit(X_rsif)

    X_test_rsif = X_rsif.transform(
        [np.arange(X_test.shape[0]).reshape(-1, 1)],
        forest=rsif,
        precomputed_distances=[[test_distance]],
        n_jobs=-1,
    )

    assert (
        roc_auc_score(y_test, -1 * rsif.predict(X_test_rsif, return_raw_scores=True))
        == 0.9707602339181286
    )
