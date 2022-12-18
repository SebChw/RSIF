import netrd
import networkx as nx

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
