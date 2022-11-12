import numpy as np
from risf.forest import RandomIsolationSimilarityForest


def test_result_on_dummy_data():
    X = [[-1.1], [0.3], [0.5], [-1], [-0.9], [0.2], [0.1], [100]]
    clf = RandomIsolationSimilarityForest(random_state=0).fit(X)
    assert np.array_equal(
        clf.predict(np.array([[0.1], [0], [90]])), np.array([0, 0, 1])
    )
