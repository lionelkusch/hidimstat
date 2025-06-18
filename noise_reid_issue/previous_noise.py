import numpy as np


def multivariate_temporal_simulation(
    n_samples=100,
    n_features=500,
    n_times=30,
    support_size=10,
    sigma_noise=1.0,
    rho_serial=0.0,
    rho=0.0,
    shuffle=True,
    seed=0,
):
    """
    Generate 1D temporal data with constant design matrix.

    Parameters
    ----------
    n_samples : int, default=100
        Number of samples.
    n_features : int, default=500
        Number of features.
    n_times : int, default=30
        Number of time points.
    support_size : int, default=10
        Size of the row support (number of non-zero coefficient rows).
    sigma : float, default=1.0
        Standard deviation of the additive white Gaussian noise.
    rho_noise : float, default=0.0
        Level of temporal autocorrelation in the noise. Must be between 0 and 1.
    rho_data : float, default=0.0
        Level of correlation between neighboring features. Must be between 0 and 1.
    shuffle : bool, default=True
        If True, randomly shuffle the features to break 1D data structure.
    seed : int, default=0
        Random seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Design matrix with Toeplitz correlation structure.
    Y : ndarray of shape (n_samples, n_times)
        Target matrix Y = X @ beta + noise.
    beta : ndarray of shape (n_features, n_times)
        Parameter matrix with first support_size rows equal to 1.
    noise : ndarray of shape (n_samples, n_times)
        Temporally correlated Gaussian noise matrix.
    """

    rng = np.random.default_rng(seed)

    X = np.zeros((n_samples, n_features))
    X[:, 0] = rng.standard_normal(n_samples)

    for i in np.arange(1, n_features):
        rand_vector = ((1 - rho**2) ** 0.5) * rng.standard_normal(n_samples)
        X[:, i] = rho * X[:, i - 1] + rand_vector

    if shuffle:
        rng.shuffle(X.T)

    beta = np.zeros((n_features, n_times))
    beta[0:support_size, :] = 1.0

    noise = np.zeros((n_samples, n_times))
    noise[:, 0] = rng.standard_normal(n_samples)

    for i in range(1, n_times):
        rand_vector = ((1 - rho_serial**2) ** 0.5) * rng.standard_normal(n_samples)
        noise[:, i] = rho_serial * noise[:, i - 1] + rand_vector

    noise = sigma_noise * noise

    Y = np.dot(X, beta) + noise

    return X, Y, beta, None, 1.0, noise
