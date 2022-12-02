from abc import ABC, abstractmethod

import itertools
import networkx as nx
import numpy as np
import netrd


class DistanceMixin(ABC):
    """This is a placeholder class"""

    def __init__(self) -> None:
        pass

    @abstractmethod
    def precompute_distances(self, selected_object=None):
        pass

    @abstractmethod
    def calculate_distance(self, id_p, id_q):
        pass

    @abstractmethod
    def project(self, id_p, id_q):
        pass


class GraphDistanceMixin(DistanceMixin):
    """
    Class documentation
    """

    def __init__(self, graphs, distance):
        """
            graphs - can be a list of NetworkX graphs or an ajdacency list or matrix representations which will be anyways converted to list of NetworkX graphs
            distance: Distance object from netrd module
        """
        self.graphs = graphs
        if isinstance(self.graphs, list):
            if isinstance(self.graphs[0], nx.Graph):
                self.graphs_nx = graphs
        self.distance = distance

    def precompute_distances(self, selected_objects=None):
        """
        selected_objects. If you want to calculate distances between only N nodes and all the rest pass their indices as this argument
            e.g if you pass [3,10,12] having 20 graphs in dataset distance matrix will be filled only for 3 rows 
        """
        m = len(self.graphs_nx)
        if selected_objects is None:
            selected_objects = list(range(0, m))

        n = len(selected_objects)

        # this matrix is symmetrix. We will only as many rows as n
        self.distance_matrix = np.zeros((m, m))

        not_selected_objects = list(
            set(list(range(m))) - set(selected_objects))
        # This list has at the beginning object selected for pairs and then all missing ones.
        all_objects = selected_objects + not_selected_objects
        # print(all_objects)
        for i in range(n):  # Iterations over objects that can constitute a pair
            # Iterations over all object so that there are no overlapping calculations
            for j in range(i+1, m):
                # i and first_obj corresponds to one another only if selected_objects is None
                first_obj_idx = selected_objects[i]
                second_obj_idx = all_objects[j]
                #print(first_obj_idx, second_obj_idx)
                g1, g2 = self.graphs_nx[first_obj_idx], self.graphs_nx[second_obj_idx]
                distance = self.distance.dist(g1, g2)

                self.distance_matrix[first_obj_idx, second_obj_idx] = distance
                self.distance_matrix[second_obj_idx, first_obj_idx] = distance

    def calculate_distance(self, id_p, id_q):
        pass

    def project(self, id_x, id_p, id_q):
        return self.distance_matrix[id_x, id_p] - self.distance_matrix[id_x, id_q]


if __name__ == "__main__":
    # G1 = nx.fast_gnp_random_graph(1000, 0.1)
    # G2 = nx.fast_gnp_random_graph(1000, 0.1)
    # G3 = nx.fast_gnp_random_graph(1000, 0.1)
    # G4 = nx.fast_gnp_random_graph(1000, 0.1)

    G1 = nx.Graph()
    G1.add_edges_from([(1, 2), (1, 3), (2, 3), (3, 4)])
    G2 = nx.Graph()
    G2.add_edges_from([(1, 2), (1, 3), (2, 3)])
    G3 = nx.Graph()
    G3.add_edges_from([(1, 2), (1, 3), (2, 3), (4, 5)])
    G4 = nx.Graph()
    G4.add_edges_from([(2, 1), (3, 1), (3, 2), (5, 4)])

    dist = netrd.distance.JaccardDistance()
    graph_dist = GraphDistanceMixin([G1, G2, G3, G4], dist)
    graph_dist.precompute_distances()
    print(graph_dist.distance_matrix)
