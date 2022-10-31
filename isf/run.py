from sklearn.ensemble import IsolationForest
import numpy as np
import sys
sys.path.append(".")
from isf.forest import RandomIsolationSimilarityForest


X = [[-1.1], [0.3], [0.5], [-1], [-0.9], [0.2], [0.1], [100]]
clf = IsolationForest(random_state=0).fit(X)
print(clf.predict([[0.1], [0], [90]]))
clf = RandomIsolationSimilarityForest(random_state=0).fit(X)
print(clf.predict(np.array([[0.1], [0], [90]])))