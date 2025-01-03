import numpy as np


def ada_svr(X, y, rcond=1e-3):
    """
    Adaptative Permutation Threshold for SVR

    Statistical inference procedure presented in Gaonkar et al. [1]_.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Data.

    y : ndarray, shape (n_samples,)
        Target.

    rcond : float, optional (default=1e-3)
        Cutoff for small singular values. Singular values smaller
        than `rcond` * largest_singular_value are set to zero.

    Returns
    -------
    beta_hat : array, shape (n_features,)
        Estimated parameter vector.

    scale : ndarray, shape (n_features,)
        Value of the standard deviation of the parameters.

    References
    ----------
    .. [1] Gaonkar, B., & Davatzikos, C. (2012, October). Deriving statistical
           significance maps for SVM based image classification and group
           comparisons. In International Conference on Medical Image Computing
           and Computer-Assisted Intervention (pp. 723-730). Springer, Berlin,
           Heidelberg.
    """

    X = np.asarray(X)

    K = np.linalg.pinv(np.dot(X, X.T), rcond=rcond)
    sum_K = np.sum(K)

    L = -np.outer(np.sum(K, axis=0), np.sum(K, axis=1)) / sum_K
    C = np.dot(X.T, K + L)

    beta_hat = np.dot(C, y)

    scale = np.sqrt(np.sum(C**2, axis=1))

    return beta_hat, scale
