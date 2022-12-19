import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import os
import pickle


OUTLIERS_RATIO = 0.01

def unify_y(y):
    y[y == -1] = 0
    if len(y[y == 0]) < len(y[y == 1]):
        y[y == 0], y[y == 1] = -1, 0
        y[y == -1] = 1
    return y

def downsample(X, y, p):
    ins_indices = np.where(y == 0)[0]
    outs_indices = np.where(y == 1)[0]
    n_outs_samples = round(len(ins_indices) * p)
     
    outs_indices_subset = resample(outs_indices, n_samples=n_outs_samples, random_state=23)
    undersampled_indices = np.concatenate([ins_indices, outs_indices_subset])
    
    X_undersampled = X[undersampled_indices]
    y_undersampled = y[undersampled_indices]

    return X_undersampled, y_undersampled
    
    
def get_numerical_datasets():
    for set_name in os.listdir("../data/numerical/"):
        data = np.load("../data/numerical/" + set_name, allow_pickle=True)
        X, y = data["X"], data["y"]
        X, y = downsample(X, y, OUTLIERS_RATIO)
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
        yield data


def get_graphs():
    graph_datasets = ["AIDS_pickles", "COX2_pickles"]
    for dataset_name in graph_datasets:
        X = []
        for set_name in os.listdir("../data/complex/" + dataset_name):
            with open("../data/complex/" + dataset_name + "/" + set_name, "rb") as f:
                X.append(pickle.load(f))
        y = np.array(X.pop())   
        y = unify_y(y)

        X = np.array(X)
        X, y = downsample(X, y, OUTLIERS_RATIO)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, shuffle=True, stratify=y, random_state=23
        )
        data = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        }
        yield data 


def get_histograms():
    graph_datasets = ["AIDS_pickles_histograms", "COX2_pickles_histograms"]
    for dataset_name in graph_datasets:
        X = []
        for set_name in os.listdir("../data/complex/" + dataset_name):
            with open("../data/complex/" + dataset_name + "/" + set_name, "rb") as f:
                X.append(pickle.load(f))
        y = np.array(X.pop())
        y = unify_y(y)
        
        X = np.array(X)
        X, y = downsample(X, y, OUTLIERS_RATIO)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, shuffle=True, stratify=y, random_state=23
        )
        data = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        }
        yield data 

        
def get_time_series():
    for set_name in os.listdir('../data/complex/UCR_Anomaly_FullData'):
        data = np.loadtxt('../data/complex/UCR_Anomaly_FullData/'+set_name, converters=float)
        info = set_name.split(sep='_')
        set_name = info[0]
        train_end = int(info[4])
        anomaly_start = int(info[5]) - train_end
        anomaly_end = int(info[6].split(sep='.')[0]) - train_end
        
        X_train = np.array(data[:train_end-1])
        y_train = np.zeros(X_train.shape)
        
        X_test = np.array(data[train_end:])
        y_test = np.zeros(X_test.shape)
        y_test[anomaly_start-1:anomaly_end-1] = 1

        data = {
            'X_train':X_train, 
            'y_train':y_train, 
            'X_test':X_test, 
            'y_test':y_test
        }
        yield data
