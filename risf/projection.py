def euclidean_projection(X, p, q):
    """
    Performs euclidean project in a form d(x,p) - d(x,q) where 
    distance function is a dot product
    Parameters
    ----------
        X : np.array of shape = [n_samples, n_features]
        p : np.array of shape = [n_features]
        q : np.array of shape = [n_features]
    Returns
    -------
        np.array
    """
    return X @ (p - q)


def make_projection(X, p, q, projection_type):
    if projection_type == "euclidean":
        return euclidean_projection(X, p, q)
    else:
        raise NameError("Unsupported projection type")
