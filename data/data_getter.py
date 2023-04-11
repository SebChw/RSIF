import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import os
import pickle
import networkx as nx
from pathlib import Path
import pandas as pd

import load_graphs

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


def graph_centrality_measures(graph):
    centralityMeasures = []
    centralityMeasures.append(np.average(list(nx.degree_centrality(graph).values())))
    centralityMeasures.append(np.average(list(nx.katz_centrality(graph).values())))
    centralityMeasures.append(np.average(list(nx.closeness_centrality(graph).values())))
    centralityMeasures.append(np.average(list(nx.harmonic_centrality(graph).values())))

    return np.array(centralityMeasures)


def make_X_numeric(X_graphs):
    X_num = []
    for graph in X_graphs:
        X_num.append(graph_centrality_measures(graph))

    return np.array(X_num)


def remove_element(X, y, idx):
    mask = np.ones(len(X), dtype=bool)
    mask[idx] = 0
    return X[mask], y[mask]


def get_graphs():
    graph_datasets = ["AIDS_pickles", "COX2_pickles"]
    for dataset_name in graph_datasets:
        X = []
        for set_name in os.listdir("../data/complex/" + dataset_name):
            with open("../data/complex/" + dataset_name + "/" + set_name, "rb") as f:
                X.append(pickle.load(f))
        y = np.array(X.pop())
        y = unify_y(y)

        X = np.array(X, dtype=object)
        X, y = downsample(X, y, OUTLIERS_RATIO)

        if dataset_name == "AIDS_pickles":
            # these 2 graphs led to NaN distance in DivergenceDist
            X, y = remove_element(X, y, [1033, 1265])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, shuffle=True, stratify=y, random_state=23
        )

        data = {
            "X_train": X_train,
            "X_train_num": make_X_numeric(X_train),
            "y_train": y_train,
            "X_test": X_test,
            "X_test_num": make_X_numeric(X_test),
            "y_test": y_test,
            "name": dataset_name,
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


def get_ucr_time_series(wanted_datasets=["Computers", "HouseTwenty", "ToeSegmentation1"]):
    OUTLIERS_RATIO = 0.05  # With 1% some time series get 0 outliers
    ROOT_PATH = Path("../data/complex/UCR_Data")
    for dataset_name in wanted_datasets:

        df_train = pd.read_csv(ROOT_PATH / dataset_name / f"{dataset_name}_TRAIN.tsv", sep='\t', header=None)
        df_test = pd.read_csv(ROOT_PATH / dataset_name / f"{dataset_name}_TEST.tsv", sep='\t', header=None)

        X_train = df_train.iloc[:, 1:].to_numpy()
        y_train = df_train.iloc[:, 0].to_numpy()

        y_train[y_train == 1] = 0
        y_train[y_train == 2] = 1
        X_train, y_train = downsample(X_train, y_train, OUTLIERS_RATIO)

        X_test = df_test.iloc[:, 1:].to_numpy()
        y_test = df_test.iloc[:, 0].to_numpy()

        y_test[y_test == 1] = 0
        y_test[y_test == 2] = 1

        X_test, y_test = downsample(X_test, y_test, OUTLIERS_RATIO)

        yield {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "name": dataset_name
        }


def get_glocalkd_dataset(data_dir, dataset_name):
    if dataset_name in ["PROTEINS_full", "ENZYMES", "AIDS", "DHFR", "BZR", "COX2"]:
        raise ValueError(f"{dataset_name} contains Attributed graphs, RISF is not a good choice for such")

    if dataset_name in ["HSE", "p53", "MMP", "PPAR-gamma"]:
        X_train = np.array(load_graphs.read_graphfile(data_dir, f'Tox21_{dataset_name}_training'), dtype=object)
        y_train = np.array([graph.graph['label'] for graph in X_train])

        X_test = np.array(load_graphs.read_graphfile(data_dir, f'Tox21_{dataset_name}_testing'), dtype=object)
        y_test = np.array([graph.graph['label'] for graph in X_test])

    elif dataset_name in ['DD', 'NCI1', 'IMDB-BINARY', 'REDDIT-BINARY', 'COLLAB']:
        X = np.array(load_graphs.read_graphfile(data_dir, dataset_name), dtype=object)
        y = np.array([graph.graph['label'] for graph in X])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, shuffle=True, stratify=y, random_state=23
        )

        X_train, y_train = downsample(X_train, y_train, p=0.1)

    elif dataset_name == "hERG":
        raise ValueError("hERG dataset is not supported yet")
    else:
        raise ValueError("Unknown dataset")

    print(f"train_counts {np.unique(y_train, return_counts=True)}")
    print(f"test_counts {np.unique(y_test, return_counts=True)}")

    return {
        "X_train": X_train,
        # "X_train_num": make_X_numeric(X_train),
        "y_train": y_train,
        "X_test": X_test,
        # "X_test_num": make_X_numeric(X_test),
        "y_test": y_test,
        "name": dataset_name
    }
