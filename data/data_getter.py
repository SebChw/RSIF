import numpy as np
from sklearn.model_selection import train_test_split
import os

def get_numerical_datasets():
    data_dir = {}
    for d in os.listdir('data/numerical/'):
        data = np.load('data/numerical/'+d, allow_pickle=True)
        X, y = data['X'], data['y']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, stratify=y, random_state=23)
        data_dir[d] = {'X_train':X_train, 'y_train':y_train, 'X_test':X_test, 'y_test':y_test, 'name':d}

    return data_dir
