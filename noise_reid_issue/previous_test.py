import numpy as np
from scipy.linalg import toeplitz

from hidimstat.noise_std import reid
from hidimstat._utils.scenario import multivariate_simulation
from previous_noise import multivariate_temporal_simulation


def test_group_reid(
    data_generator, n_samples=30, n_features=50, n_times=10, sigma=1.0, rho=0.9
):
    """Estimating (temporal) noise covariance matrix in two scenarios.
    First scenario: no data structure and a support of size 2.
    Second scenario: no data structure and an empty support."""

    corr = toeplitz(np.geomspace(1, rho ** (n_times - 1), n_times))
    cov = np.outer(sigma, sigma) * corr

    # First expe
    # ##########
    support_size = 2
    for seed in range(20):
        X, Y, beta, non_zero, noise_mag, eps = data_generator(
            n_samples=n_samples,
            n_features=n_features,
            n_times=n_times,
            support_size=support_size,
            sigma_noise=sigma,
            rho_serial=rho,
            seed=seed,
        )

        # max_iter=1 to get a better coverage
        cov_hat, _ = reid(X, Y, multioutput=True, tolerance=1e-3, max_iterance=1)
        error_ratio = cov_hat / cov
        np.set_printoptions(floatmode="fixed")
        print(
            f"1: seed:{seed}, ratio_error: {np.max(error_ratio)} {np.log(np.min(error_ratio))} expected sigma {np.diag(cov)[0]}, estimated sigma {np.diag(cov_hat)[0]}"
        )

        # assert_almost_equal(np.max(error_ratio), 1.0, decimal=0)
        # assert_almost_equal(np.log(np.min(error_ratio)), 0.0, decimal=1)

        cov_hat, _ = reid(X, Y, multioutput=True, method="AR")
        error_ratio = cov_hat / cov
        np.set_printoptions(floatmode="fixed")
        print(
            f"2: seed:{seed}, ratio_error: {np.max(error_ratio)} {np.log(np.min(error_ratio))} expected sigma {np.diag(cov)[0]}, estimated sigma {np.diag(cov_hat)[0]}"
        )

        # assert_almost_equal(np.max(error_ratio), 1.0, decimal=0)
        # assert_almost_equal(np.log(np.min(error_ratio)), 0.0, decimal=0)

        cov_hat, _ = reid(X, Y, multioutput=True, stationary=False)
        error_ratio = cov_hat / cov
        np.set_printoptions(floatmode="fixed")
        print(
            f"3: seed:{seed}, ratio_error: {np.max(error_ratio)} {np.log(np.min(error_ratio))} expected sigma {np.diag(cov)[0]}, estimated sigma {np.diag(cov_hat)[0]}"
        )

        # assert_almost_equal(np.max(error_ratio), 1.0, decimal=0)
        # assert_almost_equal(np.log(np.min(error_ratio)), 0.0, decimal=0)


if __name__ == "__main__":
    test_group_reid(data_generator=multivariate_temporal_simulation)
    test_group_reid(data_generator=multivariate_simulation)
