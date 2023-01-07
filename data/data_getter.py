import numpy as np
from sklearn.model_selection import train_test_split
import os
import pickle


def get_numerical_datasets():
    data_dir = {}
    for set_name in os.listdir("../data/numerical/"):
        data = np.load("../data/numerical/" + set_name, allow_pickle=True)
        X, y = data["X"], data["y"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, shuffle=True, stratify=y, random_state=23
        )
        data = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "name": set_name,
        }
        # data_dir[set_name] = data
        yield data


def get_graphs():
    graph_datasets = ["AIDS_pickles", "COX2_pickles"]
    data_dir = {}
    for dataset_name in graph_datasets:
        X = []
        for set_name in os.listdir("../data/complex/" + dataset_name):
            with open("../data/complex/" + dataset_name + "/" + set_name, "rb") as f:
                X.append(pickle.load(f))
        y = np.array(X.pop())
        y[y == -1] = 0

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, shuffle=True, stratify=y, random_state=23
        )
        data_dir[dataset_name] = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        }
    yield data_dir 


def get_histograms():
    graph_datasets = ["AIDS_pickles_histograms", "COX2_pickles_histograms"]
    data_dir = {}
    for dataset_name in graph_datasets:
        X = []
        for set_name in os.listdir("../data/complex/" + dataset_name):
            with open("../data/complex/" + dataset_name + "/" + set_name, "rb") as f:
                X.append(pickle.load(f))
        y = np.array(X.pop())
        y[y == -1] = 0
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, shuffle=True, stratify=y, random_state=23
        )
        data_dir[dataset_name] = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        }
    yield data_dir 

        
def get_time_series():
    data_dir = {}
    for set_name in os.listdir('../data/complex/UCR_Anomaly_FullData'):
        data = np.loadtxt('../data/complex/UCR_Anomaly_FullData/'+set_name, converters=float)
        info = set_name.split(sep='_')
        set_name = info[0]
        train_end = int(info[4])
        anomaly_start = int(info[5]) - train_end
        anomaly_end = int(info[6].split(sep='.')[0]) - train_end
        
        X_train = data[:train_end-1]
        y_train = np.zeros(X_train.shape)
        
        X_test = data[train_end:]
        y_test = np.zeros(X_test.shape)
        y_test[anomaly_start-1:anomaly_end-1] = 1
        
        data_dir[set_name] = {'X_train':X_train, 'y_train':y_train, 'X_test':X_test, 'y_test':y_test}
    yield data_dir
