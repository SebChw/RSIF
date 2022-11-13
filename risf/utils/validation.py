import numpy as np
import pandas as pd
from sklearn.utils.validation import check_array
from risf.distance import Distance


def prepare_X(X):
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    else:
        X = check_array(X)

    # Convert series/vector to a 2D array with one column
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    return X


def check_max_samples(max_samples, X):
    n = X.shape[0]
    if max_samples == "auto":
        subsample_size = min(256, n)

    elif isinstance(max_samples, float):
        subsample_size = int(max_samples * n)
        if subsample_size > n:
            subsample_size = n

    elif isinstance(max_samples, int):
        subsample_size = max_samples
        if subsample_size > n:
            subsample_size = n
    else:
        raise ValueError("max_samples should be 'auto' or either float or int")

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
    result = None

    if random_state is None:
        result = np.random.mtrand._rand
    elif isinstance(random_state, int):
        result = np.random.RandomState(random_state)
    else:
        raise TypeError("Unsupported type. Only None and int supported.")

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
    result = None

    if isinstance(distance, list):
        result = distance
    elif isinstance(distance, Distance):
        result = [distance for i in range(n_features)]
    elif isinstance(distance, str):
        result = [distance for i in range(n_features)]
    else:
        raise TypeError(
            "Unsupported distance type. Only list, str, or callable supported."
        )

    return result
