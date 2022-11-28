import numpy as np
from sklearn.model_selection import train_test_split
import os
from pygod.utils import load_data


def get_numerical_datasets():
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
        yield data


def get_graphs_organic():
    organic_names = ["weibo", "reddit", "disney", "books", "enron"]
    data_dir = {}
    for set_name in organic_names:
        data = load_data(set_name)
        X, y = data.x, data.y.bool()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, shuffle=True, stratify=y, random_state=23
        )
        data_dir[set_name] = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        }

    return data_dir


def get_graphs_injected():
    organic_names = ["inj_cora", "inj_amazon", "inj_flickr"]
    data_dir = {}
    for set_name in organic_names:
        data = load_data(set_name)
        X, y = data.x, data.y.bool()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, shuffle=True, stratify=y, random_state=23
        )
        data_dir[set_name] = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        }

    return data_dir


def get_graphs_synthetic():
    organic_names = [
        "gen_time",
        "gen_100",
        "gen_500",
        "gen_1000",
        "gen_5000",
        "gen_10000",
    ]
    data_dir = {}
    for set_name in organic_names:
        data = load_data(set_name)
        X, y = data.x, data.y.bool()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, shuffle=True, stratify=y, random_state=23
        )
        data_dir[set_name] = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        }

    return data_dir

def get_graphs():
    organic = get_graphs_organic()
    injected = get_graphs_injected()
    synthetic = get_graphs_synthetic()

    data_dir = dict(organic, **injected)
    data_dir.update(synthetic)
    
    return data_dir  
        
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

    return data_dir


datasets = [
    get_numerical_datasets(),
    get_graphs(),
    get_time_series()
]

def get_datasets():
    for dataset in datasets:
        yield dataset
        
