from this import d
import netrd
import networkx as nx
import numpy as np
from fastdtw import fastdtw
from scipy.spatial import distance
from scipy.signal import correlate

class JaccardDist():
    def __init__(self) -> None:
        self.results = {}

    def __call__(self, *args, **kwargs):
        return self.dist(*args, **kwargs)

    def dist(self, G1, G2):
        dist = netrd.distance.JaccardDistance().dist(G1, G2)
        self.results["dist"] =  dist
        return dist
    
class IpsenMikailovDist():
    def __init__(self) -> None:
        self.results = {}

    def __call__(self, *args, **kwargs):
        return self.dist(*args, **kwargs)

    def dist(self, G1, G2, hwhm=0.08):
        dist = netrd.distance.IpsenMikhailov().dist(G1, G2, hwhm)
        self.results["dist"] =  dist
        return dist

class ResistancePerturbationDist():
    def __init__(self) -> None:
        self.results = {}

    def __call__(self, *args, **kwargs):
        return self.dist(*args, **kwargs)

    def dist(self, G1, G2, p=2):
        dist = netrd.distance.ResistancePerturbation().dist(G1, G2, p)
        self.results["dist"] =  dist
        return dist

class NetSmileDist():
    def __init__(self) -> None:
        self.results = {}

    def __call__(self, *args, **kwargs):
        return self.dist(*args, **kwargs)

    def dist(self, G1, G2):
        dist = netrd.distance.NetSimile().dist(G1, G2)
        self.results["dist"] =  dist
        return dist
    
class DegreeDivergenceDist():
    def __init__(self) -> None:
        self.results = {}

    def __call__(self, *args, **kwargs):
        return self.dist(*args, **kwargs)

    def dist(self, G1, G2):
        dist = netrd.distance.DegreeDivergence().dist(G1, G2)
        self.results["dist"] =  dist
        return dist

class DynamicTimeWarpingDist():
    def __init__(self) -> None:
        self.results = {}

    def __call__(self, *args, **kwargs):
        return self.dist(*args, **kwargs)

    def dist(self, Arr1, Arr2, dist_arg=distance.euclidean):
        dist, path = fastdtw(Arr1, Arr2, dist=dist_arg)
        self.results["dist"] =  dist
        return dist
    
class CrossCorrelationDist():
    def __init__(self) -> None:
        self.results = {}

    def __call__(self, *args, **kwargs):
        return self.dist(*args, **kwargs)

    def dist(self, Arr1, Arr2):
        dist = np.max(correlate(Arr1, Arr2))
        self.results["dist"] =  dist
        return dist
    
class WassersteinDist():
    def __init__(self) -> None:
        self.results = {}

    def __call__(self, *args, **kwargs):
        return self.dist(*args, **kwargs)
    
    def adjust(self, x1, y1, x2, y2):
        
        y1_fitted = []
        y2_fitted = []
        
        for x in x1:
            if x in x2:
                x2_index = x2.index(x)
                y2_fitted.append(y2[x2_index])
            else:
                y2_fitted.append(0)
        for x in x2:
            if x in x1:
                x1_index = x1.index(x)
                y1_fitted.append(y1[x1_index])
            else:
                y1_fitted.append(0)
        
        # new_domain = sorted((x1 + list(set(x2) - set(x1))))
        
        return y1_fitted, y2_fitted
    
    def dist(self, hist1, hist2):
        x1, y1 = hist1
        x2, y2 = hist2
        
        if x1 != x2:
            y1, y2 = self.adjust(x1, y1, x2, y2)
        
        y1_cdf = np.cumsum(y1)
        y2_cdf = np.cumsum(y2)
        
        dist = 0
        for i in range(len(x1)):
            distance += abs(y1_cdf[i] - y2_cdf[i])
        
        self.results["dist"] =  dist
        return dist