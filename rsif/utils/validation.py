from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from rsif.distance import DistanceMixin
from rsif.rsif_data import RsifData
from sklearn.utils.validation import check_array


def prepare_X(
    X: Union[RsifData, pd.DataFrame, list, np.ndarray]
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Function that merger all features into one array. Numerical features are unchanged. Object like features are replaced with their indices.

    Parameters
    ----------
    X : Union[RsifData, pd.DataFrame, list, np.ndarray]
        dataset

    Returns
    -------
    Tuple[np.ndarray, List[Tuple[int, int]]]
        unified numpy array that can be passed to trees and list of features boundaries

    Raises
    ------
    TypeError
        If unsupported data type is passed
    """
    data = []
    features_span = []

    if not isinstance(X, RsifData):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        elif isinstance(X, list):
            X = np.array(X)
        elif isinstance(X, np.ndarray):
            X = X
        else:
            raise TypeError(
                "Unsupported data type: You can pass only RsifData, np.ndarray, pd.DataFrame or list"
            )

        if X.ndim == 1:
            X = X.reshape(-1, 1)
            X = check_array(X)

        data.append(X)
        features_span.append((0, X.shape[1]))

    else:
        # Now we need to treat everything separately
        curr_feature = 0
        num_of_instances = X[0].shape[0]

        for i, feature in enumerate(X):
            if isinstance(X.distances[i][0], DistanceMixin):
                feature = np.arange(num_of_instances).reshape(-1, 1)
                features_span.append((curr_feature, curr_feature + 1))
                curr_feature += 1
            else:
                features_span.append((curr_feature, curr_feature + feature.shape[1]))
                curr_feature += feature.shape[1]

            data.append(feature)

    return np.concatenate(data, axis=1), features_span


def check_max_samples(max_samples: Union[str, float, int], X: np.ndarray) -> int:
    """Function that checks the max_samples parameter and returns the number of samples that should be used for subsampling

    Parameters
    ----------
    max_samples : Union[str, float, int]
        if str - "auto" - subsample_size = min(256, n)
        if float - subsample_size = int(max_samples * n)
        if int - subsample_size = max_samples
    X : np.ndarray
        dataset. Needed to calculate minimum number of samples

    Returns
    -------
    int
        number of samples that should be used for subsampling

    Raises
    ------
    ValueError
        If max_samples is float and is not in the range (0,1]
    TypeError
        If max_samples is not str, float or int
    """
    n = X.shape[0]

    if max_samples == "auto":
        subsample_size = min(256, n)

    elif isinstance(max_samples, float):
        if max_samples > 1:
            raise ValueError(
                "If max_sample is a float, it should be in the range (0,1]"
            )
        subsample_size = int(max_samples * n)
    elif isinstance(max_samples, int) or isinstance(max_samples, np.int32):
        # if max_samples > n:
        # print("max sample is bigger than number of sample selecting n_samples")
        # raise ValueError(
        #     "If max_sample is an int, it should be in the range (0, num_of_samples]"
        # )

        # subsample_size = max_samples
        subsample_size = min(max_samples, n)

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
            "Unsupported type. Only None, int and np.random.mtrand.RandomState are supported."
        )

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
    if isinstance(distance, list):
        if len(distance) != n_features:
            raise ValueError(
                "If you provide a list of distances you must give one distance for each feature"
            )
        result = []
        for dist in distance:
            if not isinstance(dist, list):
                dist = [dist]
            result.append(dist)
    else:
        raise TypeError("Unsupported distance type. Only list of distances supported")

    return result
