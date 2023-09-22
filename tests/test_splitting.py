from unittest.mock import Mock, patch

import numpy as np
import rsif.splitting as splitting
from rsif.distance import DistanceMixin


def test_get_features_with_unique_values_distance_mixin():
    # In the node we have object 0,2 and 3
    X = np.array([[0, 0], [2, 2], [3, 3]])
    dist1 = Mock(spec=DistanceMixin)
    # !In reality this matrix should be entirely filled with zeros for obj 0,2 and 3 but I want to make sure it looks on the column
    dist1.distance_matrix = np.array(
        [[0, 5, 0, 0], [5, 0, 2, 3], [0, 2, 0, 1], [0, 3, 1, 0]]
    )
    # !This column has more than 1 unique values
    dist2 = Mock(spec=DistanceMixin)
    dist2.distance_matrix = np.array(
        [[0, 1, 2, 3], [1, 0, 2, 3], [2, 2, 0, 3], [3, 3, 3, 0]]
    )

    unique_columns = splitting.get_features_with_unique_values(
        X, [dist1, dist2], features_span=[(0, 1), (1, 2)]
    )

    # at the end I expect just the first column to be used later on
    assert unique_columns == [1]


def test_get_features_with_unique_values_numerical():
    X = np.zeros((5, 3))
    X[:, 0] = np.random.randn(5)
    X[:, 2] = np.array([0, 1, 1, 1, 1])

    unique_columns = splitting.get_features_with_unique_values(
        X, [None, None, None], features_span=[(0, 1), (1, 2), (2, 3)]
    )
    assert unique_columns == [0, 2]
