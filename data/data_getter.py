import os
import pickle
from pathlib import Path

import load_graphs
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

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

    outs_indices_subset = resample(
        outs_indices, n_samples=n_outs_samples, random_state=23
    )
    undersampled_indices = np.concatenate([ins_indices, outs_indices_subset])

    X_undersampled = X[undersampled_indices]
    y_undersampled = y[undersampled_indices]

    return X_undersampled, y_undersampled


def get_npz_dataset(path):
    data = np.load(path, allow_pickle=True)
    X, y = data["X"], data["y"]
    #! For sure we want to have repeated repeated holdout. 70%-30%
    #! In ECOD they don't perform downsampling at all.
    #! We need 10 datasets for every type of data. We don't want to perform any subsampling, only if necessary.
    return {"X": X, "y": y, "name": path.stem}


def get_categorical_dataset(path):
    df = pd.read_csv(path)

    label_map = {
        "ad_nominal.csv": {"ad.": 1, "nonad.": 0},
        "AID362red_train_allpossiblenominal.csv": {"Active": 1, "Inactive": 0},
        "Reuters-corn-100.csv": {"yes": 1, "no": 0},
    }

    X = df.iloc[:, 1:-1].values
    y = df.iloc[:, -1]

    file_name = path.name
    if file_name in label_map:
        y = y.map(lambda x: label_map[file_name][x])

    return {"X": X, "y": y, "name": path.stem}


#! We may try to do it on ADBENCH CV and NLP datasets.


def graph_centrality_measures(graph, dataset_name):
    if dataset_name == "REDDIT-BINARY":
        functions = [
            nx.degree_centrality,
            nx.closeness_centrality,
            nx.harmonic_centrality,
        ]
    else:
        functions = [
            nx.degree_centrality,
            nx.closeness_centrality,
            nx.harmonic_centrality,
            nx.katz_centrality,
        ]

    centralityMeasures = []
    for f in functions:
        try:
            centralityMeasures.append(np.average(list(f(graph).values())))
        except Exception:
            print(f"failed to calculate {f.__name__} inserting 0")
            centralityMeasures.append(0)

    return np.array(centralityMeasures)


def make_X_numeric(X_graphs, dataset_name):
    X_num = []
    for graph in X_graphs:
        X_num.append(graph_centrality_measures(graph, dataset_name))

    return np.array(X_num)


def remove_element(X, y, idx):
    mask = np.ones(len(X), dtype=bool)
    mask[idx] = 0
    return X[mask], y[mask]


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


def get_time_series(data_dir, dataset_name):
    out_path = dataset_name + "_1_0.05_1.csv"  # First split of 5% outliers
    data_dir = os.path.join(data_dir, dataset_name)
    X_outliers = pd.read_csv(os.path.join(data_dir, out_path), header=None).values
    y_out = np.ones(X_outliers.shape[0])

    in_path = dataset_name + "_1_normal.csv"
    X_inliers = pd.read_csv(os.path.join(data_dir, in_path), header=None).values
    y_in = np.zeros(X_inliers.shape[0])

    X = np.concatenate((X_inliers, X_outliers))
    y = np.concatenate((y_in, y_out))

    return {"X": X, "y": y, "name": dataset_name}


def get_glocalkd_dataset(data_dir, dataset_name, numerical_features=True):
    if dataset_name in ["PROTEINS_full", "ENZYMES", "AIDS", "DHFR", "BZR", "COX2"]:
        raise ValueError(
            f"{dataset_name} contains Attributed graphs, RISF is not a good choice for such"
        )

    if dataset_name in ["HSE", "p53", "MMP", "PPAR-gamma"]:
        #! We preffer this datasets much more.
        #! join train and test and perform repeated holdout
        X_train = np.array(
            load_graphs.read_graphfile(data_dir, f"Tox21_{dataset_name}_training"),
            dtype=object,
        )
        y_train = np.array([graph.graph["label"] for graph in X_train])

        X_test = np.array(
            load_graphs.read_graphfile(data_dir, f"Tox21_{dataset_name}_testing"),
            dtype=object,
        )
        y_test = np.array([graph.graph["label"] for graph in X_test])

        X = np.concatenate([X_train, X_test])
        y = np.concatenate([y_train, y_test])

    elif dataset_name in ["DD", "NCI1", "IMDB-BINARY", "REDDIT-BINARY", "COLLAB"]:
        #! WE don't want to downsample
        X = np.array(load_graphs.read_graphfile(data_dir, dataset_name), dtype=object)
        y = np.array([graph.graph["label"] for graph in X])

        X, y = downsample(X, y, p=0.1)

    elif dataset_name == "hERG":
        raise ValueError("hERG dataset is not supported yet")
    else:
        raise ValueError("Unknown dataset")

    y = unify_y(y)
    print(f"counts {np.unique(y, return_counts=True)}")

    result = {
        "X": X,
        "y": y,
        "name": dataset_name,
    }
    if numerical_features:
        result["X_num"] = make_X_numeric(X, dataset_name)

    return result


# TIME SERIES:
# Initial classes - 2
# liczba przykladow -> im wiecej outlierow tym lepiej (5%) -> pierwszy csv z outlierami -> laczymy to z inlierami -> repeated holdout. Im krotsze tym lepsze.
