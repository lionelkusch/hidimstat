import numpy as np
from scipy.linalg import toeplitz

from hidimstat.noise_std import reid
from hidimstat._utils.scenario import multivariate_simulation
from previous_noise import multivariate_temporal_simulation


def test_group_reid_first_senario(
    data_generator=multivariate_simulation,
    n_samples=30,
    n_features=50,
    n_times=100,
    sigma=3.0,
    rho=0.0,
    rho_noise=0.1,
    n_seed=20,
    support_size=10,
):

    # First expe
    # ##########
    support_size = 2
    for seed in range(n_seed):
        X, Y, beta, non_zero, noise_mag, eps = data_generator(
            n_samples=n_samples,
            n_features=n_features,
            n_times=n_times,
            support_size=support_size,
            sigma_noise=sigma,
            rho_serial=rho_noise,
            rho=rho,
            seed=seed,
        )
        corr = toeplitz(np.geomspace(1, rho_noise ** (n_times - 1), n_times))
        cov = np.outer(sigma * noise_mag, sigma * noise_mag) * corr

        # max_iter=1 to get a better coverage
        cov_hat, _ = reid(X, Y, multioutput=True, tolerance=1e-3, max_iterance=300)
        np.set_printoptions(floatmode="fixed")
        print(
            f"seed:{seed}, expected value {np.diag(cov)[0]}, estimated value {np.diag(cov_hat)[0]}"
            # np.sum(np.abs(np.diff(np.diag((cov))))),
            # np.sum(np.abs(np.diff(np.diag((cov_hat))))),
        )


if __name__ == "__main__":
    test_group_reid_first_senario(data_generator=multivariate_temporal_simulation)
    test_group_reid_first_senario(data_generator=multivariate_simulation)
