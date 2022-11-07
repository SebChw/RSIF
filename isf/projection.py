
import numpy as np

def euclidean_projection(X, p, q):
    return X @ (p - q)
    #return np.dot(p-q, xi)

def euclidean_projection_full(X, p, q):
    X_p = X - p
    X_q = X - q
    return (X_q * X_q).sum(axis=1) - (X_p * X_p).sum(axis=1) #Actually we can't use sqrt here!
    #If we used it then we will get same results for every point.

def make_projection(X,  p,  q, projection_type):
    if projection_type == "euclidean":
        return euclidean_projection(X, p, q)
    else:
        raise NameError('Unsupported projection type')