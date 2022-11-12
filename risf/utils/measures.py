import numpy as np

def _average_path_length(n_samples):
    """A function estimating average external path length of Similarity Tree.
            Since Similarity Tree, the same as Isolation tree, has an equivalent structure to Binary Search Tree,
            the estimation of average h for external node terminations is the same as the unsuccessful search in BST.
            Parameters
            ----------
            n : int, number of objects used for tree construction
            Returns
            ----------
            average external path length : int
        """
    assert n_samples - 1 > 0
    return 2 * np.log(n_samples - 1) + np.euler_gamma - 2 * (n_samples - 1) / n_samples