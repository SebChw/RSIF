import numpy as np
import pandas as pd
import numbers

import risf.projection as projection
from risf.distance import DistanceMixin


def project(X, Oi, Oj, dist):
    if X.ndim != 1:
        raise ValueError(
            "Project accepts 1 dimensional array of complex object \
                indices or numeric values"
        )

    if isinstance(Oi, numbers.Number):  # Simple numeric features
        if isinstance(
            dist, str
        ):  # This basically support only euclidean distance computation
            projection_ = projection.make_projection(
                X.astype(np.float32).reshape(-1, 1),
                np.array([Oi], dtype=np.float32),
                np.array([Oj], dtype=np.float32),
                dist,
            )
        elif isinstance(dist, DistanceMixin):
            projection_ = dist.project(X, Oi, Oj)

        else:
            raise TypeError("Unsupported projection type")

    else:
        raise TypeError(
            "Unsupported Pair object data. If data is not numperic, \
            complex features must be stored as pointers in 2d array."
        )

    return projection_


def get_features_with_unique_values(X, distances):
    features_with_unique_values = []

    for column_id in range(X.shape[1]):
        if isinstance(distances[column_id], DistanceMixin):
            distance_matrix = distances[column_id].distance_matrix
            row = X[:, column_id]
            distances_to_first = distance_matrix[row[0], row]
            # If all objects doesn't differ in distance, splitting is without a sense
            # !this assumption that we check only first object vs rest is valid iff measure is a METRIC
            if np.count_nonzero(distances_to_first) > 0:
                features_with_unique_values.append(column_id)

        elif isinstance(X[0, column_id], numbers.Number):
            # numerical features, built-in distance
            if np.unique(X[:, column_id].astype(np.float32), axis=0).shape[0] > 1:
                features_with_unique_values.append(column_id)
        else:
            # complex features, built-in distance
            if pd.Series(X[:, column_id]).apply(tuple).nunique() > 1:
                features_with_unique_values.append(column_id)

    return features_with_unique_values
