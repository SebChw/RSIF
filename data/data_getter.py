import os
from pathlib import Path
from typing import Tuple

import load_graphs
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils import resample


def unify_y(y: np.ndarray) -> np.ndarray:
    """Unify labels to 1 - outlier, 0 - inlier"""
    y[y == -1] = 0
    if len(y[y == 0]) < len(y[y == 1]):
        y[y == 0], y[y == 1] = -1, 0
        y[y == -1] = 1
    return y


def downsample(X: np.ndarray, y: np.ndarray, p: float) -> Tuple[np.ndarray, np.ndarray]:
    """Downsample outliers to the p ratio"""
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
    """We additionally perform standarization as RISF/ISF/LOF beign based on distances are sensitive to it"""
    data = np.load(path, allow_pickle=True)
    X, y = data["X"], data["y"]
    X = StandardScaler().fit_transform(X)
    return {"X": X, "y": y, "name": path.stem}


def get_categorical_dataset(path: Path):
    """Parse categorical dataset"""
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

    features = []
    for col_id in range(X.shape[1]):
        features.append(LabelEncoder().fit_transform(X[:, col_id]))

    X_cat = np.vstack(features).T

    drop_binary_enc = OneHotEncoder(drop="if_binary").fit(X)
    X = drop_binary_enc.transform(X).toarray()

    return {"X": X, "X_cat": X_cat, "y": y.values, "name": path.stem}


def graph_centrality_measures(graph, dataset_name: str):
    """Calculate centrality measures for a graph. Can be used with algorithm that works just for numerical data"""
    if dataset_name in ["DD"]:
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
    """Convert graph dataset into a numeric one"""
    X_num = []
    for graph in X_graphs:
        X_num.append(graph_centrality_measures(graph, dataset_name))

    X_num = np.array(X_num)

    return StandardScaler().fit_transform(X_num)


def get_timeseries(data_dir, dataset_name):
    """Load timeseries dataset and perform standarization"""
    out_path = dataset_name + "_1_0.05_1.csv"  # First split of 5% outliers
    data_dir = os.path.join(data_dir, dataset_name)
    X_outliers = pd.read_csv(os.path.join(data_dir, out_path), header=None).values
    y_out = np.ones(X_outliers.shape[0])

    in_path = dataset_name + "_1_normal.csv"
    X_inliers = pd.read_csv(os.path.join(data_dir, in_path), header=None).values
    y_in = np.zeros(X_inliers.shape[0])

    X = np.concatenate((X_inliers, X_outliers))
    y = np.concatenate((y_in, y_out))

    # Here we perform different kind of standarization. We standardize every time series separately
    X = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)

    return {"X": X, "y": y, "name": dataset_name}


def graph_bagofwordize(graph_db):
    """Builds a bag of words representation of a graph dataset"""
    columns = max(
        [max([graph.nodes[n]["label"] for n in graph.nodes]) for graph in graph_db]
    )

    result = np.zeros((len(graph_db), columns + 1))

    for i, graph in enumerate(graph_db):
        for k, v in nx.get_node_attributes(graph, "label").items():
            result[i, v] += 1

    return StandardScaler().fit_transform(result)


def get_glocalkd_dataset(data_dir, dataset_name):
    """Load graph datasets"""
    if dataset_name in ["DD", "NCI1", "DHFR", "BZR", "COX2", "AIDS", "PROTEINS_full", "ENZYMES"]:  # fmt: skip
        X = np.array(load_graphs.read_graphfile(data_dir, dataset_name), dtype=object)
        y = np.array([graph.graph["label"] for graph in X])

        y = unify_y(y)

        # In case of graphs dataset we used classification datasets. We downsample them on our own
        X, y = downsample(X, y, p=0.05)
    else:
        raise ValueError("Unknown dataset")

    result = {
        "y": y,
        "name": dataset_name,
        "X_graph": X,
        "X_num": make_X_numeric(X, dataset_name),
        "X": graph_bagofwordize(X),
    }

    return result


def get_multiomics_data(data_path, data_name, for_risf=True):
    """Load multiomics datasets. These are mixed type datasets so we need to prepere array of features."""
    y = pd.read_csv(os.path.join(data_path, data_name, "y.csv"), index_col=0).values

    features = []
    features_types = []
    if data_name in ["ovarian", "breast"]:
        X = pd.read_csv(os.path.join(data_path, data_name, "X.csv"), index_col=0)

        histogram_columns = ['cnv_del_LEN', 'cnv_dip_LEN', 'cnv_dup_LEN', 'sv_inv_LEN', 'sv_del_LEN', 'sv_dup_LEN']  # fmt: skip
        index_of_numerical = 6

        features.append(X.iloc[:, index_of_numerical:].values)
        features_types.append("multiomics")

        if for_risf:
            for histogram_column in histogram_columns:
                stuff = list(
                    X[histogram_column].apply(lambda x: np.fromstring(x[1:-1], sep=" "))
                )

                features.append(np.stack(stuff))
                features_types.append("histogram")
    elif data_name == "rosmap":
        X1 = pd.read_csv(os.path.join(data_path,data_name, "X_1.csv"), index_col=0).values  # fmt: skip
        X2 = pd.read_csv(os.path.join(data_path,data_name, "X_2.csv"), index_col=0).values  # fmt: skip
        X3 = pd.read_csv(os.path.join(data_path,data_name, "X_3.csv"), index_col=0).values  # fmt: skip
        y = pd.read_csv(os.path.join(data_path,data_name, "y.csv"), index_col=0).values  # fmt: skip

        features = [X1, X2, X3]
        features_types = ["multiomics", "multiomics", "multiomics"]

    if not for_risf:
        features = np.concatenate(features, axis=1)

    return {
        "X": features,
        "y": y,
        "name": data_name,
        "features_types": features_types,
    }


def sequence_of_sets_bagofwordize(sequences):
    """Builds a bag of words representation of a sequence of sets dataset"""
    columns = max([max([max(s) for s in seq]) for seq in sequences])

    result = np.zeros((len(sequences), columns))

    for i, seq in enumerate(sequences):
        for s in seq:
            result[i, s - 1] = 1

    return result


def seq_of_sets_lengths(sequences):
    """Builds a length representation of a sequence of sets dataset"""
    distances = [[len(s) for s in seq] for seq in sequences]
    max_n_sets = max([len(s) for s in distances])
    for i, seq in enumerate(distances):
        distances[i] = np.pad(seq, (0, max_n_sets - len(seq)), "constant")

    return np.array(distances)


def get_sets_data(data_path, data_name, for_risf=False):
    """Load sequence of sets datasets."""
    X = pd.read_csv(os.path.join(data_path, data_name, "X.csv"), index_col=0).values
    y = pd.read_csv(os.path.join(data_path, data_name, "y.csv"), index_col=0).values

    result = list()

    for line in X:
        line = str(line[0])
        seq_of_sets = [np.array(list(s)) for s in eval(line)]
        result.append(seq_of_sets)

    result_bow = sequence_of_sets_bagofwordize(result)
    results_seq_lengths = seq_of_sets_lengths(result)

    if for_risf:
        features = [result_bow, result, results_seq_lengths]
        features_types = ["bag_of_words", "seq_of_sets", "seq_of_lengths"]
        return {
            "X": features,
            "y": y,
            "name": data_name,
            "features_types": features_types,
        }

    else:
        return {"X": result_bow, "y": y, "name": data_name}
