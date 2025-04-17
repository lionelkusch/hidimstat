"""
Test the desparsified_lasso module
"""

import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
from scipy.linalg import toeplitz

from hidimstat.desparsified_lasso import DesparsifiedLasso
from hidimstat._utils.scenario import (
    multivariate_1D_simulation,
    multivariate_temporal_simulation,
)
from hidimstat.statistical_tools.p_values import pval_corr_from_pval


def test_desparsified_lasso():
    """Testing the procedure on a simulation with no structure and
    a support of size 1. Computing 99% confidence bounds and checking
    that they contains the true parameter vector."""

    n_samples, n_features = 52, 50
    support_size = 1
    sigma = 0.1
    rho = 0.0

    X, y, beta, noise = multivariate_1D_simulation(
        n_samples=n_samples,
        n_features=n_features,
        support_size=support_size,
        sigma=sigma,
        rho=rho,
        shuffle=False,
        seed=2,
    )
    expected_pval_corr = np.concatenate(
        (np.zeros(support_size), 0.5 * np.ones(n_features - support_size))
    )
    desparsified_lasso = DesparsifiedLasso()
    desparsified_lasso.fit(X, y)
    pval = desparsified_lasso.importance(confidence=0.99)
    pval_corr = pval_corr_from_pval(pval)
    assert_almost_equal(desparsified_lasso.beta_hat, beta, decimal=1)
    assert_equal(desparsified_lasso.confidence_bound_min < beta, True)
    assert_equal(desparsified_lasso.confidence_bound_max > beta, True)
    assert_almost_equal(pval_corr, expected_pval_corr, decimal=1)

    desparsified_lasso_with_dot = DesparsifiedLasso(dof_ajdustement=True)
    desparsified_lasso_with_dot.fit(X, y)
    pval_with_dot = desparsified_lasso_with_dot.importance(confidence=0.99)
    pval_corr_with_dot = pval_corr_from_pval(pval_with_dot)

    assert_almost_equal(desparsified_lasso_with_dot.beta_hat, beta, decimal=1)
    assert_equal(desparsified_lasso_with_dot.confidence_bound_min < beta, True)
    assert_equal(desparsified_lasso_with_dot.confidence_bound_max > beta, True)
    assert_almost_equal(pval_corr_with_dot, expected_pval_corr, decimal=1)


def test_desparsified_group_lasso():
    """Testing the procedure on a simulation with no structure and
    a support of size 2. Computing one-sided p-values, we want
    low p-values for the features of the support and p-values
    close to 0.5 for the others."""

    n_samples = 50
    n_features = 100
    n_times = 10
    support_size = 2
    sigma = 0.1
    rho = 0.9
    corr = toeplitz(np.geomspace(1, rho ** (n_times - 1), n_times))
    cov = np.outer(sigma, sigma) * corr

    X, y, beta, noise = multivariate_temporal_simulation(
        n_samples=n_samples,
        n_features=n_features,
        n_times=n_times,
        support_size=support_size,
        sigma=sigma,
        rho_noise=rho,
    )
    desparsified_lasso = DesparsifiedLasso(group=True, covariance=cov)
    desparsified_lasso.fit(X, y)
    pval = desparsified_lasso.importance()
    pval_corr = pval_corr_from_pval(pval)

    expected_pval_corr = np.concatenate(
        (np.zeros(support_size), 0.5 * np.ones(n_features - support_size))
    )
    assert_almost_equal(desparsified_lasso.beta_hat, beta, decimal=1)
    assert_almost_equal(pval_corr, expected_pval_corr, decimal=1)

    desparsified_lasso_no_cov = DesparsifiedLasso(group=True)
    desparsified_lasso_no_cov.fit(X, y)
    pval_no_cov = desparsified_lasso_no_cov.importance(test="F")
    pval_corr_no_cov = pval_corr_from_pval(pval_no_cov)

    assert_almost_equal(desparsified_lasso_no_cov.beta_hat, beta, decimal=1)
    assert_almost_equal(pval_corr_no_cov, expected_pval_corr, decimal=1)

    # Testing error is raised when the covariance matrix has wrong shape
    bad_cov = np.delete(cov, 0, axis=1)
    desparsified_lasso_bad_cov = DesparsifiedLasso(group=True, covariance=bad_cov)
    np.testing.assert_raises(ValueError, desparsified_lasso_bad_cov.fit, X=X, y=y)

    with pytest.raises(ValueError, match=f"Unknown test 'r2'"):
        desparsified_lasso_error = DesparsifiedLasso(group=True)
        desparsified_lasso_error.fit(X, y)
        pval_no_cov = desparsified_lasso_error.importance(test="r2")
