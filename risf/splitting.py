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


def get_features_with_nonunique_values(X, distances):
    features_with_nonunique_values = []

    for column in range(X.shape[1]):
        if not isinstance(distances[column], str):
            # custom distance, e.g. lookup distance,
            # different values do not mean something is non-unique
            # Not supported for complex objects!
            features_with_nonunique_values.append(column)
            # distances_to_first = np.array(
            #     [distances[column](X[0, column], i) for i in X[:, column]]
            # )
            # if np.count_nonzero(distances_to_first) > 0:
            #     features_with_nonunique_values.append(column)
        elif isinstance(X[0, column], numbers.Number):
            # numerical features, built-in distance
            if np.unique(X[:, column].astype(np.float32), axis=0).shape[0] > 1:
                features_with_nonunique_values.append(column)
        else:
            # complex features, built-in distance
            if pd.Series(X[:, column]).apply(tuple).nunique() > 1:
                features_with_nonunique_values.append(column)

    return features_with_nonunique_values
