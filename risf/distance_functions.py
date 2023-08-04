import numpy as np
from fastdtw import fastdtw
from scipy.interpolate import interp1d
from scipy.signal import correlate
from scipy.spatial.distance import cosine
from scipy.stats import entropy, wasserstein_distance

"""
All predefined distance functions
"""


def euclidean_projection(X, p, q):
    return X @ (p - q)


def manhattan_projection(X, p, q):
    dist_X_p = np.abs(X - p).sum(axis=1)
    dist_X_q = np.abs(X - q).sum(axis=1)
    return dist_X_p - dist_X_q


def chebyshev_projection(X, p, q):
    dist_X_p = np.abs(X - p).max(axis=1)
    dist_X_q = np.abs(X - q).max(axis=1)
    return dist_X_p - dist_X_q


def dummy_projection(X, p, q):
    return X


def cosine_sim(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    denominator = (X * X).sum(axis=1) ** 0.5 * (y @ y) ** 0.5
    return np.divide(
        X @ y, denominator, out=np.zeros_like(denominator), where=denominator != 0
    )


def cosine_projection(X, p, q):
    dist_X_p = 1 - cosine_sim(X, p)
    dist_X_q = 1 - cosine_sim(X, q)
    return dist_X_p - dist_X_q


def jaccard_sim(X, p):
    numerator = np.bitwise_and(X, p).sum(axis=1, dtype=np.float64)
    denominator = np.bitwise_or(X, p).sum(axis=1, dtype=np.float64)

    return np.divide(
        numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0
    )


def jaccard_projection(X, p, q):
    X = X.astype(bool)
    p = p.astype(bool)
    q = q.astype(bool)

    dist_X_p = 1 - jaccard_sim(X, p)
    dist_X_q = 1 - jaccard_sim(X, q)

    return dist_X_p - dist_X_q


class EditDistanceSequencesOfSets:
    def jaccard(self, set1, set2):
        intersection = 0
        total = set1.shape[0] + set2.shape[0]

        for i in range(set1.shape[0]):
            for j in range(set2.shape[0]):
                if set1[i] == set2[j]:
                    intersection += 1
                    break

        return 1.0 - (intersection / (total - intersection))

    def __call__(self, s1, s2):
        M = np.zeros((len(s1) + 1, len(s2) + 1))

        M[0, 0] = 0

        for i in range(1, len(s1) + 1):
            M[i, 0] = i

        for j in range(1, len(s2) + 1):
            M[0, j] = j

        for i in range(1, len(s1) + 1):
            for j in range(1, len(s2) + 1):
                M[i, j] = min(
                    M[i - 1, j] + 1,
                    M[i, j - 1] + 1,
                    M[i - 1, j - 1] + self.jaccard(s1[i - 1], s2[j - 1]),
                )

        return M[len(s1), len(s2)]


class DiceDist:
    def __call__(self, x, y):
        nominator = 0
        denominator = len(x)
        for x_val, y_val in zip(x, y):
            if x_val == y_val:
                nominator += 1

        return 1 - (nominator / denominator)


class EuclideanDist:
    def __call__(self, x, y):
        return np.linalg.norm(x - y)


class ManhattanDist:
    def __call__(self, x, y):
        return np.abs(x - y).sum()


class CosineDist:
    def __call__(self, x, y):
        return cosine(x, y)


class ChebyshevDist:
    def __call__(self, x, y):
        return np.abs(x - y).max()


class DTWDist:
    def __call__(self, *args, **kwargs):
        return self.dist(*args, **kwargs)

    def dist(self, x1, x2):
        return fastdtw(x1, x2)[0]


def get_frequency_table(data, normalize=True):
    max_attrs = len(np.unique(data))
    freq_table = np.zeros(shape=(max_attrs, data.shape[1]), dtype=int)
    for col_id in range(data.shape[1]):
        unique = np.unique(data[:, col_id], return_counts=True)
        freq_table[: len(unique[0]), col_id] = unique[1]

    return freq_table / data.shape[0] if normalize else freq_table


class LinDist:
    def __init__(self, data):
        self.freq_table = get_frequency_table(data)

    def __call__(self, *args, **kwargs):
        return self.dist(*args, **kwargs)

    def dist(self, x1, x2):
        agreement = np.zeros(len(x1))
        weights = np.zeros(len(x1))
        for k in range(len(x1)):
            c = x1[k]
            d = x2[k]
            if c == d:
                agreement[k] = 2 * np.log(self.freq_table[c, k])
            else:
                agreement[k] = 2 * np.log(self.freq_table[c, k] + self.freq_table[d, k])
            weights[k] = np.log(self.freq_table[c, k]) + np.log(self.freq_table[d, k])

        return 1 / (1 / sum(weights) * (sum(agreement))) - 1


class Goodall3Dist:
    def __init__(self, data):
        self.freq_table = get_frequency_table(data)

    def __call__(self, *args, **kwargs):
        return self.dist(*args, **kwargs)

    def dist(self, x1, x2):
        agreement = np.zeros(len(x1))
        for k in range(len(x1)):
            if x1[k] == x2[k]:
                agreement[k] = 1 - self.freq_table[x1[k]][k] ** 2

        return 1 - sum(agreement) / len(x1)


class OFDist:
    def __init__(self, data):
        self.freq_table = get_frequency_table(data, normalize=False)
        self.r = data.shape[0]

    def __call__(self, *args, **kwargs):
        return self.dist(*args, **kwargs)

    def dist(self, x1, x2):
        agreement = np.ones(len(x1))
        for k in range(len(x1)):
            c = x1[k]
            d = x2[k]
            if c != d:
                agreement[k] = agreement[k] = 1 / (
                    1
                    + np.log(self.r / self.freq_table[c, k])
                    * np.log(self.r / self.freq_table[d, k])
                )

        return (len(x1) / sum(agreement)) - 1


class GraphDist:
    def __init__(self, dist_class, params: dict = {}) -> None:
        self.distance = dist_class()
        self.params = params

    def __call__(self, G1, G2):
        return self.distance.dist(G1, G2, **self.params)


class CrossCorrelationDist:
    def __init__(self) -> None:
        self.results = {}

    def __call__(self, *args, **kwargs):
        return self.dist(*args, **kwargs)

    def dist(self, Arr1, Arr2):
        dist = np.max(correlate(Arr1, Arr2))
        self.results["dist"] = dist
        return dist


class WassersteinDist:
    def __call__(self, hist1, hist2):
        return wasserstein_distance(hist1, hist2)


class JensenShannonDivDist:
    def __init__(self) -> None:
        self.results = {}

    def __call__(self, *args, **kwargs):
        return self.dist(*args, **kwargs)

    def adjust(self, Arr1, Arr2):
        if len(Arr1) < len(Arr2):
            x = np.arange(len(Arr2))
            f = interp1d(x, Arr2, kind="linear")
            Arr2 = f(np.arange(len(Arr1)))
        else:
            x = np.arange(len(Arr1))
            f = interp1d(x, Arr1, kind="linear")
            Arr1 = f(np.arange(len(Arr2)))
        return Arr1, Arr2

    def dist(self, Arr1, Arr2):
        if len(Arr1) != len(Arr2):
            Arr1, Arr2 = self.adjust(Arr1, Arr2)
        Arr1 /= Arr1.sum()
        Arr2 /= Arr2.sum()
        m = (Arr1 + Arr2) / 2

        dist = (entropy(Arr1, m) + entropy(Arr2, m)) / 2

        self.results["dist"] = dist
        return dist


class TSEuclidean:
    def __init__(self) -> None:
        self.results = {}

    def __call__(self, *args, **kwargs):
        return self.dist(*args, **kwargs)

    def adjust(self, Arr1, Arr2):
        if len(Arr1) < len(Arr2):
            x = np.arange(len(Arr2))
            f = interp1d(x, Arr2, kind="linear")
            Arr2 = f(np.arange(len(Arr1)))
        else:
            x = np.arange(len(Arr1))
            f = interp1d(x, Arr1, kind="linear")
            Arr1 = f(np.arange(len(Arr2)))
        return Arr1, Arr2

    def dist(self, Arr1, Arr2):
        if len(Arr1) != len(Arr2):
            Arr1, Arr2 = self.adjust(Arr1, Arr2)
        dist = np.linalg.norm(Arr1 - Arr2) ** 2
        self.results["dist"] = dist
        return dist


class HistEuclidean:
    def __init__(self) -> None:
        self.results = {}

    def __call__(self, *args, **kwargs):
        return self.dist(*args, **kwargs)

    def adjust(self, values1, bins1, values2, bins2):
        min1, max1 = min(bins1), max(bins1)
        min2, max2 = min(bins2), max(bins2)

        bins = np.arange(min(min1, min2), max(max1, max2) + 1)

        values1_new = [0] * len(bins)
        for bin in bins1:
            i = np.where(bins == bin)[0][0]
            j = np.where(bins1 == bin)[0][0]
            values1_new[i] = values1[j]
        values2_new = [0] * len(bins)
        for bin in bins2:
            i = np.where(bins == bin)[0][0]
            j = np.where(bins2 == bin)[0][0]
            values2_new[i] = values2[j]
        return values1_new, values2_new

    def dist(self, hist1, hist2):
        bins1, values1 = hist1
        bins2, values2 = hist2
        if bins1 != bins2:
            bins1, bins2 = np.array(bins1), np.array(bins2)
            values1, values2 = self.adjust(values1, bins1, values2, bins2)
        values1, values2 = np.array(values1), np.array(values2)

        dist = np.linalg.norm(values1 - values2) ** 2
        self.results["dist"] = dist
        return dist
