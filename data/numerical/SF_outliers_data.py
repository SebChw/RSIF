from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_kddcup99, fetch_covtype, fetch_openml
from sklearn.preprocessing import StandardScaler, LabelBinarizer
import numpy as np
import pandas as pd
# from sklearn.utils import shuffle


def fix_dtypes(df):
    float_cols = [c for c in df if df[c].dtype == 'float64']
    int_cols = [c for c in df if df[c].dtype in ['int64', 'int32']]
    #obj_cols = [c for c in df if df[c].dtype == 'object']
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int16)
    '''for c in obj_cols:
        df[c] = LabelEncoder().fit_transform(df[c])'''

    return df


def get_kddcup99_http():
    # fetch data
    X, y = fetch_kddcup99(subset='http', random_state=42, return_X_y=True)
    X, y = X.astype(np.float32), y.astype('str')

    # fix classes
    y_df = pd.DataFrame(y, columns=['class'])
    y_df.loc[y_df['class'] != 'normal.', 'class'] = 1
    y_df.loc[y_df['class'] == 'normal.', 'class'] = 0
    y = y_df.values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    y_train = pd.Series(y_train.flatten())
    y_test = pd.Series(y_test.flatten())

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, 'kddcup99_http'


def get_kddcup99_sf():
    X, y = fetch_kddcup99(subset='SF', random_state=42, return_X_y=True)
    lb = LabelBinarizer()
    x1 = lb.fit_transform(X[:, 1].astype(str))
    X = np.c_[X[:, :1], x1, X[:, 2:]]
    y = y.astype('str')
    y_df = pd.DataFrame(y, columns=['class'])
    y_df.loc[y_df['class'] != 'normal.', 'class'] = 1
    y_df.loc[y_df['class'] == 'normal.', 'class'] = 0
    y = y_df.values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    y_train = pd.Series(y_train.flatten())
    y_test = pd.Series(y_test.flatten())

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, 'kddcup99_sf'


def get_kddcup99_sa():
    X, y = fetch_kddcup99(subset='SA', random_state=42, return_X_y=True)
    lb = LabelBinarizer()
    x1 = lb.fit_transform(X[:, 1].astype(str))
    x2 = lb.fit_transform(X[:, 2].astype(str))
    x3 = lb.fit_transform(X[:, 3].astype(str))
    X = np.c_[X[:, :1], x1, x2, x3, X[:, 4:]]
    y = y.astype('str')
    y_df = pd.DataFrame(y, columns=['class'])
    y_df.loc[y_df['class'] != 'normal.', 'class'] = 1
    y_df.loc[y_df['class'] == 'normal.', 'class'] = 0
    y = y_df.values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    y_train = pd.Series(y_train.flatten())
    y_test = pd.Series(y_test.flatten())

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, 'kddcup99_sa'


# def get_shuttle():
#     dataset = fetch_openml('shuttle')
#     X = dataset.data
#     y = dataset.target
#     X, y = shuffle(X, y, random_state=1)
#     y = y.astype(int)
#     # we remove data with label 4
#     # normal data are then those of class 1
#     s = (y != 4)
#     X = X[s, :]
#     y = y[s]
#     y[(y == 1)] = 1
#     y[(y != 1)] = -1

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.3, random_state=42)

#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)

#     return X_train, X_test, y_train, y_test, 'shuttle'


def get_forestcover():
    dataset = fetch_covtype(shuffle=True, random_state=1)
    X = dataset.data
    y = dataset.target
    # normal data are those with attribute 2
    # abnormal those with attribute 4
    s = (y == 2) + (y == 4)
    X = X[s, :]
    y = y[s]
    y[(y == 2)] = 0
    y[(y == 4)] = 1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, 'forestcover'


# def get_thyronoid_disease():
#     df = pd.read_csv('examples/data/Annthyroid_real.csv', header=None)
#
#     y = df.pop(6)
#     X = df
#     y[(y == 1)] = -1
#     y[(y == 2)] = -1
#     y[(y == 3)] = 1
#
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.3, random_state=42)
#
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)
#
#     return X_train, X_test, y_train, y_test, 'thyronoid_disease'


def get_breastw():
    dataset = fetch_openml('breast-w')
    X = dataset.data
    y = dataset.target
    df = pd.DataFrame(np.column_stack((X, y)))
    df.dropna(inplace=True)
    y = df.pop(9)
    X = df
    y[(y == 'malignant')] = 1
    y[(y == 'benign')] = 0
    y = y.astype(np.int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, 'breastw'


datasets = [
            get_kddcup99_http(),
            get_kddcup99_sf(),
            get_kddcup99_sa(),
            # get_shuttle(),
            get_forestcover(),
            # get_thyronoid_disease(),
            get_breastw()
]


def get_datasets():
    for d in datasets:
        yield d
