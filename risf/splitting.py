import numpy as np

from risf.distance import DistanceMixin


def get_features_with_unique_values(X, distances, features_span):
    features_with_unique_values = []

    for feature_id, (feature_start, feature_end) in enumerate(features_span):
        if isinstance(distances[feature_id], DistanceMixin):
            distance_matrix = distances[feature_id].distance_matrix
            row = X[:, feature_start:feature_end]
            distances_to_first = distance_matrix[row[0], row]
            # If all objects doesn't differ in distance, splitting is without a sense
            # !this assumption that we check only first object vs rest is valid iff measure is a METRIC
            if np.count_nonzero(distances_to_first) > 0:
                features_with_unique_values.append(feature_id)
        else:
            # numerical features, built-in distance
            if (
                np.unique(
                    X[:, feature_start:feature_end].astype(np.float32), axis=0
                ).shape[0]
                > 1
            ):
                features_with_unique_values.append(feature_id)

    return features_with_unique_values
