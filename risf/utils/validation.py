import numpy as np
import pandas as pd
from sklearn.utils.validation import check_array

from risf.distance import DistanceMixin
from risf.risf_data import RisfData


def prepare_X(X):
    # TODO handle case when X is RisfData and smart view on indices is returned
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    elif isinstance(X, RisfData):  # Here check about RisfData Should be done
        num_of_instances = X[0].shape[0]
        num_of_columns = len(X)
        one_row = np.arange(num_of_instances).reshape(-1,
                                                      1)  # transpose didn't work
        # THIS could be done more memory-efficient for sure
        return np.repeat(one_row, num_of_columns, axis=1)
    elif isinstance(X, np.ndarray):
        # Convert series/vector to a 2D array with one column
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X = check_array(X)
    else:
        raise TypeError(
            "Unsupported data type: You can pass only one of (pd.DataFrame, RisfData, np.ndarray)")

    return X


def check_max_samples(max_samples, X):
    n = X.shape[0]
    if max_samples == "auto":
        subsample_size = min(256, n)

    elif isinstance(max_samples, float):
        if max_samples > 1:
            raise ValueError(
                "If max_sample is a float, it should be in the range (0,1]")
        subsample_size = int(max_samples * n)

    elif isinstance(max_samples, int):
        if max_samples > n:
            raise ValueError(
                "If max_sample is an int, it should be in the range (0, num_of_samples]")

        subsample_size = max_samples

    else:
        raise TypeError("max_samples should be 'auto' or either float or int")

    return subsample_size


def check_random_state(random_state):
    """Checks the random_state parameter

    Args:
        random_state (int/None): \'None\' - creates a random state without
        seed, \'int\' - creates a random state with a given seed.

    Raises:
        TypeError: Unsupported type

    Returns:
        RandomState: A random state for reproducibility
    """
    if random_state is None:
        result = np.random.mtrand._rand
    elif isinstance(random_state, int):
        result = np.random.RandomState(random_state)
    elif isinstance(random_state, np.random.mtrand.RandomState):
        return random_state
    else:
        raise TypeError(
            "Unsupported type. Only None, int and np.random.mtrand.RandomState are supported.")

    return result


def check_distance(distance, n_features):
    """Validates the distance parameter of RandomSimilarityForest and
       RandomSimilarityTree.

    Args:
        distance (object): The distance that should be calculated for
        each feature.Accepts \'list\', \'callable\', and \'str\': \'list\'
            - a list of distances (functions) for each feature
        (should be the same length as the number of features), \'callable\'
            - a distance function for all features,
        \'str\' - name of the distance measure that should be used for all
        features. n_features (int): Total number of features in the input
        dataset

    Raises:
        TypeError: Unsupported distance type

    Returns:
        list: A list of distances for each feature
    """
    if isinstance(distance, str):
        result = [[distance] for i in range(n_features)]
        
    elif isinstance(distance, list):
        if len(distance) != n_features :
            raise ValueError(
                "If you provide a list of distances you must give one distance for each feature"
            )
        result = []
        for dist in distance:
            if not isinstance(dist, list):
                dist = [dist]
            result.append(dist)
    else:
        raise TypeError(
            "Unsupported distance type. Only list or str supported"
        )

    return result
