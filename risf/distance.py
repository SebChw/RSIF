from abc import ABC, abstractmethod

import itertools
import networkx as nx
import numpy as np
#from risf.datatypes import RisfDataType

# import netrd

class DistanceMixin():
    def __init__(self, distance: callable) -> None:
        self.distance = distance
        self.used_points = set()

    def precompute_distances(self, X, selected_objects=None):
        """
        selected_objects. If you want to calculate distances between only N 
        nodes and all the rest pass their indices as this argument
        e.g if you pass [3,10,12] having 20 graphs in dataset distance 
        matrix will be filled only for 3 rows 
        """
        m = len(X)
        if selected_objects is None:
            selected_objects = list(range(0, m))

        n = len(selected_objects)

        # this matrix is symmetrix. We will only as many rows as n
        self.distance_matrix = np.zeros((m, m)) #! Not sure if this should be n by m to have only selected objects at rows positions 

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
                objp, objq = X[first_obj_idx], X[second_obj_idx]
                distance = self.distance(objp, objq)

                self.distance_matrix[first_obj_idx, second_obj_idx] = distance
                self.distance_matrix[second_obj_idx, first_obj_idx] = distance
               
    def calculate_distance(self, obj_p, obj_q):
        return self.distance(obj_p, obj_q)

    def project(self, id_x, id_p, id_q):
        #! For now let's do it this but later each tree can do this after training!
        self.used_points.add(id_p)
        self.used_points.add(id_q)
        
        return self.distance_matrix[id_p, id_x] - self.distance_matrix[id_q, id_x]

class TestDistanceMixin:
    def __init__(self, train_distance_mixin: DistanceMixin):
        #! Actually if I pass different distance object I do not need offset indices will match anyway!
        #self.offset = X_train[0].shape[0] #We can't give same indices as train data had
        self.distance = train_distance_mixin.distance
        self.train_points_to_use = train_distance_mixin.used_points

    def precompute_distances(self, X_train, X_test):
        self.distance_matrix = np.zeros((X_train.shape[0], X_test.shape[0]))
        for used_point in self.train_points_to_use:
            for i, test_obj in enumerate(X_test):
                self.distance_matrix[used_point][i] = self.distance(X_train[used_point], test_obj)

    def project(self, id_x, id_p, id_q):
        return self.distance_matrix[id_p, id_x] - self.distance_matrix[id_q, id_x]
