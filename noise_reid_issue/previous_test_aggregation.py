import numpy as np
from scipy.linalg import toeplitz

from hidimstat.noise_std import reid
from hidimstat._utils.scenario import multivariate_simulation
from previous_noise import multivariate_temporal_simulation


def test_group_reid(
    data_generator,
    n_samples=30,
    n_features=50,
    n_times=10,
    sigma=1.0,
    rho=0.9,
    nb_seed=20,
):
    """Estimating (temporal) noise covariance matrix in two scenarios.
    First scenario: no data structure and a support of size 2.
    Second scenario: no data structure and an empty support."""

    corr = toeplitz(np.geomspace(1, rho ** (n_times - 1), n_times))
    cov = np.outer(sigma, sigma) * corr

    # First expe
    # ##########
    support_size = 2
    result = [[], [], []]
    for seed in range(nb_seed):
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
        result[0].append(
            (cov, cov_hat, error_ratio, np.diag(cov)[0], np.diag(cov_hat)[0])
        )
        np.set_printoptions(floatmode="fixed")
        # print(
        #     f"1: seed:{seed}, ratio_error: {np.max(error_ratio)} {np.log(np.min(error_ratio))} expected sigma {np.diag(cov)[0]}, estimated sigma {np.diag(cov_hat)[0]}"
        # )

        # assert_almost_equal(np.max(error_ratio), 1.0, decimal=0)
        # assert_almost_equal(np.log(np.min(error_ratio)), 0.0, decimal=1)

        cov_hat, _ = reid(X, Y, multioutput=True, method="AR")
        error_ratio = cov_hat / cov
        result[1].append(
            (cov, cov_hat, error_ratio, np.diag(cov)[0], np.diag(cov_hat)[0])
        )
        np.set_printoptions(floatmode="fixed")
        # print(
        #     f"2: seed:{seed}, ratio_error: {np.max(error_ratio)} {np.log(np.min(error_ratio))} expected sigma {np.diag(cov)[0]}, estimated sigma {np.diag(cov_hat)[0]}"
        # )

        # assert_almost_equal(np.max(error_ratio), 1.0, decimal=0)
        # assert_almost_equal(np.log(np.min(error_ratio)), 0.0, decimal=0)

        cov_hat, _ = reid(X, Y, multioutput=True, stationary=False)
        error_ratio = cov_hat / cov
        result[2].append(
            (cov, cov_hat, error_ratio, np.diag(cov)[0], np.diag(cov_hat)[0])
        )
        np.set_printoptions(floatmode="fixed")
        # print(
        #     f"3: seed:{seed}, ratio_error: {np.max(error_ratio)} {np.log(np.min(error_ratio))} expected sigma {np.diag(cov)[0]}, estimated sigma {np.diag(cov_hat)[0]}"
        # )

        # assert_almost_equal(np.max(error_ratio), 1.0, decimal=0)
        # assert_almost_equal(np.log(p.min(error_ratio)), 0.0, decimal=0)
    for i in range(3):
        error_array = np.array([a[2] for a in result[i]])
        sigma_cov = np.array([a[3] for a in result[i]])
        sigma_cov_hat = np.array([a[4] for a in result[i]])
        print(
            f"exp:{i} max ratio: {np.max(error_array)}, sdt: {np.std(error_array)}, diag ratio max: {np.max(sigma_cov/sigma_cov_hat)}, std: {np.std(sigma_cov/sigma_cov_hat)}, std hat: {np.std(sigma_cov_hat)}"
        )
    return result


if __name__ == "__main__":
    result = test_group_reid(
        data_generator=multivariate_temporal_simulation,
        n_samples=30,
        n_features=50,
        n_times=10,
    )
    # exp:0 max ratio: 3.439288434057792, sdt: 0.48217730090438066, diag ratio max: 1.1103285863263137, std: 0.1959813496946681, std hat: 0.3467094730285075
    # exp:1 max ratio: 3.9165866211893112, sdt: 0.6412922217672975, diag ratio max: 31.33583998091608, std: 6.556760147538239, std hat: 0.49979061307834366
    # exp:2 max ratio: 3.9769562707735555, sdt: 0.5410626268288273, diag ratio max: 1.4846400555431416, std: 0.2600512313886619, std hat: 0.45177274100920295
    result = test_group_reid(
        data_generator=multivariate_temporal_simulation,
        n_samples=30,
        n_features=50,
        n_times=1000,
    )
    # exp:0 max ratio: 6789735.32677102, sdt: 99016.70445693615, diag ratio max: 0.8673344911245104, std: 0.21196035511571865, std hat: 4.689410936379959
    # exp:1 max ratio: 1185690.4960165657, sdt: 18915.448701284862, diag ratio max: 1.2713617065395322, std: 0.06620695593749118, std hat: 0.05214848691764928
    # exp:2 max ratio: 7666165.042759163, sdt: 102369.59130088573, diag ratio max: 1.0937381533660795, std: 0.2299411466436768, std hat: 7.1993186446118616
    result = test_group_reid(
        data_generator=multivariate_temporal_simulation,
        n_samples=100,
        n_features=50,
        n_times=1000,
    )
    # exp:0 max ratio: 331.4389851363045, sdt: 13.612392514260577, diag ratio max: 0.9870821038839529, std: 0.049744587102200005, std hat: 0.06440020562989011
    # exp:1 max ratio: 290.8659608787585, sdt: 11.956885461824355, diag ratio max: 1.0536219976776595, std: 0.017161886210423447, std hat: 0.01662622669320492
    # exp:2 max ratio: 345.44843545629095, sdt: 13.89256200622719, diag ratio max: 1.193929610044668, std: 0.12581595442559174, std hat: 0.1453495605395544

    # test_group_reid(data_generator=multivariate_simulation)
