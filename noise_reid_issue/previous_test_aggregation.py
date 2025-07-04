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
    rho=0.9,
    snr=3.0,
    nb_seed=20,
    support_size=2,
):
    """Estimating (temporal) noise covariance matrix in two scenarios.
    First scenario: no data structure and a support of size 2.
    Second scenario: no data structure and an empty support."""

    corr = toeplitz(np.geomspace(1, rho ** (n_times - 1), n_times))
    cov = support_size / snr * corr
    diag_cov = np.diag(cov)

    # First expe
    # ##########
    result = [[], [], []]
    for seed in range(nb_seed):
        X, Y, beta, noise = data_generator(
            n_samples=n_samples,
            n_features=n_features,
            n_targets=n_times,
            support_size=support_size,
            rho_serial=rho,
            signal_noise_ratio=snr,
            seed=seed,
        )

        # max_iter=1 to get a better coverage
        cov_hat, _ = reid(X, Y, multioutput=True, tolerance=1e-3, max_iterance=1)
        error_ratio = np.abs(cov_hat - cov) / cov
        error_ratio_diag = np.abs(np.diag(cov_hat) - diag_cov) / diag_cov
        result[0].append(
            (cov, cov_hat, error_ratio, np.diag(cov)[0], np.diag(cov_hat)[0])
        )
        np.set_printoptions(floatmode="fixed")
        # print(
        #     f"1: seed:{seed}, ratio_error: {np.max(error_ratio)} {np.log(np.min(error_ratio))} expected sigma {np.diag(cov)[0]}, estimated sigma {np.diag(cov_hat)[0]}"
        # )
        print(np.mean(error_ratio_diag), np.max(error_ratio_diag))
        assert snr > 1e3 or np.mean(error_ratio_diag) < 0.3

        cov_hat, _ = reid(X, Y, multioutput=True, method="AR")
        error_ratio = np.abs(cov_hat - cov) / cov
        error_ratio_diag = np.abs(np.diag(cov_hat) - diag_cov) / diag_cov
        result[1].append(
            (cov, cov_hat, error_ratio, np.diag(cov)[0], np.diag(cov_hat)[0])
        )
        np.set_printoptions(floatmode="fixed")
        # print(
        #     f"2: seed:{seed}, ratio_error: {np.max(error_ratio)} {np.log(np.min(error_ratio))} expected sigma {np.diag(cov)[0]}, estimated sigma {np.diag(cov_hat)[0]}"
        # )
        print(np.mean(error_ratio_diag), np.max(error_ratio_diag))
        assert snr > 1e3 or np.mean(error_ratio_diag) < 0.3

        cov_hat, _ = reid(X, Y, multioutput=True, stationary=False)
        error_ratio = np.abs(cov_hat - cov) / cov
        error_ratio_diag = np.abs(np.diag(cov_hat) - diag_cov) / diag_cov
        result[2].append(
            (cov, cov_hat, error_ratio, np.diag(cov)[0], np.diag(cov_hat)[0])
        )
        np.set_printoptions(floatmode="fixed")
        # print(
        #     f"3: seed:{seed}, ratio_error: {np.max(error_ratio)} {np.log(np.min(error_ratio))} expected sigma {np.diag(cov)[0]}, estimated sigma {np.diag(cov_hat)[0]}"
        # )
        print(
            np.mean(error_ratio_diag),
            np.max(error_ratio_diag),
            np.min(error_ratio_diag),
        )
        # assert np.max(error_ratio_diag) > 0.4

        # assert np.all(error_ratio > 0.5)
    result_mean_std = []
    for i in range(3):
        error_array = np.array([a[2] for a in result[i]])
        sigma_cov = np.array([a[3] for a in result[i]])
        sigma_cov_hat = np.array([a[4] for a in result[i]])
        result_mean_std.append([np.mean(sigma_cov_hat), np.std(sigma_cov_hat)])
        # print(f"mean estimation: {np.mean(sigma_cov_hat)} {np.std(sigma_cov_hat)}")
        print(
            f"exp:{i} max ratio: {np.max(error_array)}, sdt: {np.std(error_array)}, diag ratio max: {np.max(sigma_cov/sigma_cov_hat)}, diag ratio min: {np.min(sigma_cov/sigma_cov_hat)}, std: {np.std(sigma_cov/sigma_cov_hat)}, std hat: {np.std(sigma_cov_hat)}"
        )
    return result_mean_std


if __name__ == "__main__":
    import os

    path = os.path.dirname(os.path.abspath(__file__))
    from matplotlib import pyplot as plt

    # result = test_group_reid(
    #     data_generator=multivariate_temporal_simulation,
    #     n_samples=30,
    #     n_features=50,
    #     n_times=10,
    # )
    # result = test_group_reid(
    #     data_generator=multivariate_temporal_simulation,
    #     n_samples=30,
    #     n_features=50,
    #     n_times=1000,
    # )
    # result = test_group_reid(
    #     data_generator=multivariate_temporal_simulation,
    #     n_samples=100,
    #     n_features=50,
    #     n_times=1000,
    # )
    # result = test_group_reid(
    #     data_generator=multivariate_temporal_simulation,
    #     n_samples=100,
    #     n_features=50,
    #     n_times=10,
    # )
    # result = test_group_reid(
    #     data_generator=multivariate_temporal_simulation,
    #     n_samples=100,
    #     n_features=50,
    #     n_times=100,
    # )
    range_value_snr = np.logspace(-10, 9, 30)
    range_support_size = range(1, 15)
    for support_size in range_support_size:
        result = []
        for i in range_value_snr:
            print("run tests", i, support_size)
            result.append(
                test_group_reid(
                    data_generator=multivariate_simulation,
                    nb_seed=20,
                    n_samples=100,
                    n_features=20,
                    n_times=50,
                    snr=i,
                    support_size=support_size,
                )
            )
        result = np.array(result)
        np.save("result_" + str(support_size) + ".npy", result)
