import numpy as np
import pandas as pd
import numbers

from isf.projection import make_projection


def project(X, Oi, Oj, dist):
    if isinstance(dist, str): # built-in distance measure
        if isinstance(Oi, numbers.Number): # Simple numeric features
            projection = np.array(make_projection(X.astype(np.float32).reshape(-1, 1),
                                                  np.array([Oi], dtype=np.float32), np.array([Oj], dtype=np.float32),
                                                  dist))
        else: # complex features
            if X.ndim == 1:
                try:
                    X_stacked = np.stack(X).astype(np.float32) # When dealing with nested arrays
                    projection = np.array(make_projection(X_stacked, np.array(Oi, dtype=np.float32),
                                                          np.array(Oj, dtype=np.float32), dist))
                except:
                    raise TypeError('Unsupported data-distance combination. Complex features (if they are not '
                                    'numerical lists of the same length) require passing a custom callable distance '
                                    'function')
            else:
                raise TypeError('Unsupported data-distance combination. Complex features (if they are not numerical '
                                'lists of the same length) require passing a custom callable distance function')
    else: # custom (callable) distance measure
        projection = np.array([project_example(X[k], Oi, Oj, dist) for k in range(X.shape[0])])

    # order samples according to their projection
    projected_sample_order = projection.argsort()

    return projected_sample_order, projection[projected_sample_order]


def project_example(x, Oi, Oj, dist):
    return dist(Oi, x) - dist(Oj, x)


def get_features_with_nonunique_values(X, distances):
    features_with_nonunique_values = []

    for column in range(X.shape[1]):
        if not isinstance(distances[column], str):
            # custom distance, e.g. lookup distance, different values do not mean something is non-unique
            distances_to_first = np.array([distances[column](X[0, column], i) for i in X[:, column]])
            if np.count_nonzero(distances_to_first) > 0:
                features_with_nonunique_values.append(column)
        elif isinstance(X[0, column], numbers.Number):
            # numerical features, built-in distance
            if np.unique(X[:, column].astype(np.float32), axis=0).shape[0] > 1:
                features_with_nonunique_values.append(column)
        else:
            # complex features, built-in distance
            if pd.Series(X[:, column]).apply(tuple).nunique() > 1:
                features_with_nonunique_values.append(column)

    return features_with_nonunique_values
    
