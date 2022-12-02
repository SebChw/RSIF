from risf.forest import RandomIsolationSimilarityForest
from risf.distance import GraphDistanceMixin
import networkx as nx
import netrd
import numpy as np


G1 = nx.Graph()
G1.add_edges_from([(1, 2), (1, 3), (2, 3), (3, 4)])
G2 = nx.Graph()
G2.add_edges_from([(1, 2), (3, 1), (2, 3), (3, 4), (4, 5)])
G3 = nx.Graph()
G3.add_edges_from([(2, 1), (1, 3), (2, 3), (3, 4), (5, 4)])
G4 = nx.Graph()
G4.add_edges_from([(10, 3), (12, 4), (5, 8), (11, 6)])

dist = netrd.distance.JaccardDistance()
graph_dist = GraphDistanceMixin([G1, G1, G1, G2, G2, G2, G3, G3, G3, G4], dist)
graph_dist.precompute_distances()
print(graph_dist.distance_matrix)
X = np.array([[0], [1], [2], [3]])  # Just indices to complex objects
distance = [graph_dist]

clf = RandomIsolationSimilarityForest(
    random_state=0, distance=distance).fit(X)


# Indices to complex objects
print(clf.predict(np.array([[0], [1], [2], [3]])))
