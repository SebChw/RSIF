import netrd
import numpy as np
from fastdtw import fastdtw
from scipy.interpolate import interp1d
from scipy.signal import correlate
from scipy.spatial.distance import cosine, dice, jaccard
from scipy.stats import entropy, wasserstein_distance

"""
All predefined distance functions
"""

# TODO: add more distance functions and write tests for them. For now, just euclidean projection has been tested


def euclidean_projection(X, p, q):
    """
    Performs euclidean project in a form d(x,p) - d(x,q) where
    distance function is a dot product
    Parameters
    ----------
        X : np.array of shape = [n_samples, n_features]
        p : np.array of shape = [n_features]
        q : np.array of shape = [n_features]
    Returns
    -------
        np.array
    """
    return X @ (p - q)


def manhattan_projection(X, p, q):
    dist_X_p = np.abs(X - p).sum(axis=1)
    dist_X_q = np.abs(X - q).sum(axis=1)
    return dist_X_p - dist_X_q


def chebyshev_projection(X, p, q):
    dist_X_p = np.abs(X - p).max(axis=1)
    dist_X_q = np.abs(X - q).max(axis=1)
    return dist_X_p - dist_X_q


def cosine_projection(X, p, q):
    dist_X_p = 1 - cosine(X, p)
    dist_X_q = 1 - cosine(X, q)
    return dist_X_p - dist_X_q


def jaccard_projection(X, p, q):
    dist_X_p = 1 - np.double(np.bitwise_and(X, p).sum(axis=1)) / (
        np.double(np.bitwise_or(X, p).sum(axis=1) + 1e-10)
    )
    dist_X_q = 1 - np.double(np.bitwise_and(X, q).sum(axis=1)) / (
        np.double(np.bitwise_or(X, q).sum(axis=1) + 1e-10)
    )
    return dist_X_p - dist_X_q


def dice_projection(X, p, q):
    pass


class JaccardDist:
    def __call__(self, *args, **kwargs):
        return self.dist(*args, **kwargs)

    def dist(self, x1, x2):
        return jaccard(x1, x2)


class DiceDist:
    def __call__(self, *args, **kwargs):
        return self.dist(*args, **kwargs)

    def dist(self, x1, x2):
        return dice(x1, x2)


class DTWDist:
    def __call__(self, *args, **kwargs):
        return self.dist(*args, **kwargs)

    def dist(self, x1, x2):
        return fastdtw(x1, x2)[0]


class JaccardGraphDist:
    def __init__(self) -> None:
        self.results = {}

    def __call__(self, *args, **kwargs):
        return self.dist(*args, **kwargs)

    def dist(self, G1, G2):
        dist = netrd.distance.JaccardDistance().dist(G1, G2)
        self.results["dist"] = dist
        return dist


class IpsenMikailovDist:
    def __init__(self) -> None:
        self.results = {}

    def __call__(self, *args, **kwargs):
        return self.dist(*args, **kwargs)

    def dist(self, G1, G2, hwhm=0.08):
        dist = netrd.distance.IpsenMikhailov().dist(G1, G2, hwhm)
        self.results["dist"] = dist
        return dist


class NetSmileDist:
    def __init__(self) -> None:
        self.results = {}

    def __call__(self, *args, **kwargs):
        return self.dist(*args, **kwargs)

    def dist(self, G1, G2):
        dist = netrd.distance.NetSimile().dist(G1, G2)
        self.results["dist"] = dist
        return dist


class PortraitDivergenceDist:
    def __init__(self) -> None:
        self.distance_func = netrd.distance.PortraitDivergence()

    def __call__(self, *args, **kwargs):
        return self.dist(*args, **kwargs)

    def dist(self, G1, G2):
        return self.distance_func.dist(G1, G2)


class DegreeDivergenceDist:
    def __init__(self) -> None:
        self.results = {}

    def __call__(self, *args, **kwargs):
        return self.dist(*args, **kwargs)

    def dist(self, G1, G2):
        dist = netrd.distance.DegreeDivergence().dist(G1, G2)
        self.results["dist"] = dist
        return dist


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
    def __init__(self) -> None:
        self.results = {}

    def __call__(self, *args, **kwargs):
        return self.dist(*args, **kwargs)

    def dist(self, hist1, hist2):
        bins1, values1 = hist1
        bins2, values2 = hist2

        dist = wasserstein_distance(values1, values2, bins1, bins2)

        self.results["dist"] = dist
        return dist


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
