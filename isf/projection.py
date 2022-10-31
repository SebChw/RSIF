
import numpy as np

def euclidean_projection( xi, p, q):
    return np.dot(p-q, xi)

def make_projection(X,  p,  q, projection_type):
    n = X.shape[0]
    result = np.empty(n, dtype=np.float32)

    if projection_type == "euclidean":
        for i in range(n):
            result[i] = euclidean_projection(X[i, :], p, q)
    else:
        raise NameError('Unsupported projection type')

    return result

def single_projection(xi,  p,  q, projection_type):

    if projection_type == "euclidean":
        result = euclidean_projection(xi, p, q)

    return result