import sys
sys.path.insert(0, "..")

from tqdm.contrib.concurrent import process_map

import numpy as np

from pyod.utils.utility import precision_n_scores
from sklearn.metrics import roc_auc_score
from data.data_getter import get_numerical_datasets
from notebooks.utils import new_clf, Timer

import pandas as pd

clfs_names = ["ECOD", "LOF", "IForest", "HBOS"]


def run(data, clf_name):
    timer = Timer(timer_type="long_running")
    clf = new_clf(clf_name, 23)
    timer.start()
    clf.fit(data["X_train"])
    timer.stop()

    # get the prediction labels and outlier scores of the training data
    y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
    y_train_scores = clf.decision_scores_  # raw outlier scores

    y_test_scores = clf.decision_function(data["X_test"])  # outlier scores

    if np.isnan(y_train_scores).any() or np.isnan(y_train_pred).any():
        return (
            clf_name,
            data["name"],
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )  # AUC/ROC, Rank@N for train,test ; fit Time

    roc_train = np.round(roc_auc_score(data["y_train"], y_train_scores),
                         decimals=4)
    prn_train = np.round(
        precision_n_scores(data["y_train"], y_train_scores), decimals=4
    )
    roc_test = np.round(roc_auc_score(data["y_test"], y_test_scores), decimals=4) # noqa
    prn_test = np.round(precision_n_scores(data["y_test"], y_test_scores),
                        decimals=4)

    return (
            clf_name,
            data["name"],
            roc_train,
            prn_train,
            roc_test,
            prn_test,
            timer.time_sec,
            )


def main():
    output = process_map(run, get_numerical_datasets(), clfs_names * 47)
    results = {x: {} for x in clfs_names}
    for clf_name, name, roc_train, prn_train, roc_test, prn_test, timer in output:  # noqa
        results[clf_name][name] = (roc_train, prn_train,
                                   roc_test, prn_test, timer)
    return results


if __name__ == "__main__":
    results = main()
    df = pd.DataFrame(results)
    df.to_pickle('../results/numerical.pkl')
