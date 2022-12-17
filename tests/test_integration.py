import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.datasets import load_wine

import pytest

from risf.forest import RandomIsolationSimilarityForest


@pytest.mark.integration
def test_result_on_dummy_data():
    X = [[-1.1], [0.3], [0.5], [-1], [-0.9], [0.2], [0.1], [100]]
    clf = RandomIsolationSimilarityForest(random_state=0).fit(X)
    assert np.array_equal(
        clf.predict(np.array([[0.1], [0], [90]])), np.array([0, 0, 1])
    )


@pytest.mark.integration
def test_pipeline_sucess_on_bigger_dataset():
    """In this test we just check if it suceeds to fit a tree and return any scores"""
    # 13 numerical attributes anb 178 instances
    wine_data = load_wine()["data"]
    clf = RandomIsolationSimilarityForest(
        random_state=0, contamination=0.8).fit(wine_data)
    predictions = clf.predict(np.ones((2, 13)))
    assert predictions.size == 2


@pytest.mark.integration
def test_metrics_on_small_dataset():
    """In this test we check if we didn't mess up anything, so that we get bad scores comparing to our first implementation
    We also check if agreement with ISF is quite high"""
    data = np.load('data/numerical/01_breastw.npz',
                   allow_pickle=True)  # very simple dataset
    X, y = data['X'], data['y']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=True, stratify=y, random_state=23)

    isf = IsolationForest(random_state=0).fit(X_train)
    isf_pred = isf.predict(X_test)
    isf_pred_shifted = isf_pred.copy()
    isf_pred_shifted[isf_pred_shifted == 1] = 0
    isf_pred_shifted[isf_pred_shifted == -1] = 1

    risf = RandomIsolationSimilarityForest(random_state=0).fit(X_train)
    risf_pred = risf.predict(X_test)

    # I got better result on the first correct implementation but let's save some small margin
    assert (((risf_pred == isf_pred_shifted).sum()) /
            isf_pred.shape[0]) > 0.9  # agreement
    assert precision_score(y_test, risf_pred) > 0.95
    assert accuracy_score(y_test, risf_pred) > 0.9
    assert recall_score(y_test, risf_pred) > 0.85


@pytest.mark.integration
def test_interface_1():
    pass
    #!Just an interface sketch
    # import pandas as pd
    # #Assuming that user have some: numerical, graph and timeseries data.
    # #I literally have no idea how to connect timeseries with other types into mixed data.
    # #For any type user adds number of objects must be the same! 1000 rows in dataFrame -> 1000 graph objects etc

    # #Typical use, we change nothing 
    # numerical_data = pd.read_csv("file.csv")
    # new_data = pd.read_csv("to_predict.csv")
    # risf = RandomIsolationSimilarityForest().fit(numerical_data)
    # pred = risf.predict(new_data)

    # #!For any different case user must use our proxy which will be our fancy RisfData object
    # #which will behave like an array with all operators overloaded but internally
    # #this will just allow our trees to juggle with indices of objects 
    # #https://stackoverflow.com/questions/1957780/how-to-override-the-operator-in-python
    # #It will also map distance classes to appropriate columns.
    # #! the biggest positive would be that algorithm doesn't have to be changed at all.

    # #Mimic SimilarityForest
    # X = RisfData()
    # X.add_data(numerical_data, type_="numerical", dist = NumericalDistance(treatment="all"))
    # #Internally appropriate data representation is created based on distance function (that's why we have to add distance here. I know it's not too good.)
    # #This X behave like typical array but given treatment="all" One column is basically a vector of all columns.
    # risf = RandomIsolationSimilarityForest().fit(X, dist = X.get_distances())
    # X_test = X.create_test_data(new_data) #???????????????????????????? Not sure if this is a good idea
    # pred = risf.predict(X_test)

    # #Custom numerical vectors
    # X = RisfData()
    # X.add_data(numerical_data, type_="numerical", dist = [NumericalDistance(columns=(1,3)), NumericalDistance(columns=(4,8)), NumericalDistance(columns=(9,12))])
    # #This will create X having 3 columns. First is a vector of element 1,2 and 3 and so on.
    # #Internally appropriate data representation is created based on distance function
    # risf = RandomIsolationSimilarityForest().fit(X, dist = X.get_distances())

    # #Time series
    # time_series_train = pickle.load(open('/content/drive/My Drive/RISF/data-369.pickle', 'rb'))
    # time_series_test = pickle.load(open('/content/drive/My Drive/RISF/data-369.pickle', 'rb'))
    # X = RisfData()
    # X.add_data(time_series_train[:5], type_="timeseries", dist = TimeSeriesDistance(treatment="histogram", window=50))
    # X.add_data(time_series_train[5:10], type_="timeseries", dist = TimeSeriesDistance(treatment="timeseries", window=50)) #We must pass the same window
    # X.add_data(time_series_train[10:12], type_="timeseries", dist = TimeSeriesDistance(treatment="average", window=50)) #We must pass the same window
    # #Internally first 5 time series will be treated as histograms and next 5 as timeseties and next 2 as scalar averages
    # risf = RandomIsolationSimilarityForest().fit(X, dist = X.get_distances())
    # X_test = X.create_test_data([time_series_test[:5], time_series_test[5:10], time_series_test[10:12]]) # we must pass as many elements as many different data types we added

    # #Graphs + numerical
    # graph_data = load_adj_matrix("graph.txt")
    # graph_data2 = load_adj_matrix("graph2.txt")
    # X = RisfData()
    # X.add_data(graph_data, type_="graph", dist = GraphDistance(treatment="jaccard"))
    # X.add_data(graph_data2, type_="graph", dist = GraphDistance(treatment="fancy_custom_super_option_here"))
    # X.add_data(numerical_data, type_="numerical", dist = NumericalDistance(treatment="all")) #OFC we must have as many graphs as rows in numerical data
    # risf = RandomIsolationSimilarityForest.fit(X, dist = X.get_distances())
    # class NumericalDistance:
    #     pass
    # class RisfData:
    #     pass
    # class TimeSeriesDistance:
    #     pass

    # class DistributionDistance:
    #     pass

    # class GraphDistance:
    #     pass

    # numerical_data = 0
    # time_series_data = 0
    # histogram_data = 0
    # graph_data = 0

    # numerical_test = 0
    # time_series_test = 0
    # histogram_test = 0
    # graph_test = 0

    # #Preparing the data
    # X = RisfData() # This will work as array, we will overload [] operator
    # X.fit(numerical_data, type_="numerical", dist = NumericalDistance(metric="euclidean"))
    # X.fit(time_series_data, type_="timeseries", dist = TimeSeriesDistance(metric="dtw"))
    # X.fit(histogram_data, type_="histogram", dist = DistributionDistance(metric="kld"))
    # X.fit(graph_data, type_="graph", dist = GraphDistance(metric="jaccard"))

    # #Training
    # risf = RandomIsolationSimilarityForest.fit(X, dist = X.get_distances())
    
    # #Predicting
    # X_test = X.predict([numerical_test, time_series_test, histogram_test, graph_test])
    # risf.predict(X_test)

@pytest.mark.integration
def test_result_on_dummy_data():
    data = np.load('data/numerical/01_breastw.npz',
                   allow_pickle=True)  
    X, y = data['X'], data['y']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=True, random_state=23)

    risf = RandomIsolationSimilarityForest(random_state=0).fit(X_train, y_train)
    risf_pred = risf.predict(X_train)
    computedP = sum(risf_pred)/len(risf_pred)
    correctP = sum(y_train)/len(y_train)
    assert computedP == correctP
