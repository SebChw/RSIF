from risf.distance import DistanceMixin, TestDistanceMixin
import numpy as np


class RisfData(list):
    def __init__(self,):
        self.distances = []
        self.names = []
        self.transforms = []


    def add_data(self, X, dist: callable, data_transform: callable = None, name=None):
        if data_transform is None:
            if not isinstance(X, np.ndarray):
                raise Exception("If you don't provide data_transform function, given data must be an instance of nd.array")
            else:
                super().append(X)
        else:
            transformed = [data_transform(x) for x in X]
            if isinstance(transformed[0], np.ndarray):
                super().append(np.array(transformed))
            else:
                data = np.empty(len(transformed), dtype=object)
                data[:] = transformed
                super().append(data)

        self.transforms.append(data_transform)
        self.names.append(name if name is not None else f"attr{len(self)}")
        
        if isinstance(dist, DistanceMixin) or isinstance(dist, TestDistanceMixin):
            self.distances.append(dist)
        else:
            self.distances.append(DistanceMixin(dist))
    
    def create_test_data(self, list_of_X):
        test_data = RisfData()
        for i, X in enumerate(list_of_X):
            test_distance = TestDistanceMixin(self.distances[i]) # DistanceMixing goes here
            test_data.add_data(X, test_distance, self.transforms[i], self.names[i])
            test_distance.precompute_distances(self[i], test_data[i])

        return test_data

    def precompute_distances(self):
        for data, distance in zip(self, self.distances):
            distance.precompute_distances(data)

    def transform(self, X=None):
        return self