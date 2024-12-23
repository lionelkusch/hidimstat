"""
ADA-SVR: Adaptive Permutation Threshold Support Vector Regression
==================================================================
Statistical inference procedure presented in :footcite:t:`gaonkar_deriving_2012`.
"""

#############################################################################
# Imports needed for this script
# ------------------------------
import matplotlib.pyplot as plt
import numpy as np
from hidimstat.ada_svr import ada_svr, ada_svr_pvalue
from hidimstat.permutation_test import permutation_test
from sklearn.svm import SVR
from hidimstat.scenario import multivariate_1D_simulation
from examples._utils.plot_dataset import (
    plot_dataset1D,
    plot_validate_variable_importance,
    plot_pvalue_H0,
    plot_pvalue_H1,
    plot_compare_proba_estimated,
)

#############################################################################
# Generate toy dataset
# --------------------
#

# Parameters for the generation of data
n_samples, n_features = 20, 100
support_size = 1

X, y, beta, _ = multivariate_1D_simulation(
    n_samples=n_samples,
    n_features=n_features,
    support_size=support_size,
    sigma=0.1,
    shuffle=False,
    seed=42,
)
plot_dataset1D(X=X, y=y, beta=beta)
#############################################################################
# Usage the methods
# -----------------
# see the API for more details about the optional parameter:
# :py:func:`hidimstat.ada_svr`

beta_hat, scale = ada_svr(X, y, rcond=1e-1)

#############################################################################
# | **beta_hat** is the estimated variable of importance
# | **scale** is the standard deviation of the distribution of the coefficients

#############################################################################
# Plot the results
# ----------------
#
plot_validate_variable_importance(beta, beta_hat)

#############################################################################
# The result shows that the only variable of importance has a higher score
# but the difference between beta and beta_hat is quite different.

#############################################################################
#
# The pvalue and the corrected pvalue can help to be confident in the previous
# results.


pvalue, pvalue_corrected, one_minus_pvalue, one_minus_pvalue_correlation = (
    ada_svr_pvalue(beta_hat, scale)
)
plot_pvalue_H0(
    beta_hat, pvalue, pvalue_corrected, one_minus_pvalue, one_minus_pvalue_correlation
)

#############################################################################
# The pvalue and the corrected pvalue show that the confidence in the variable
# of importance is high. The pvalue and the corrected pvalue are close to one.
# We can conclude that the variable of importance is estimated in this case.

plot_pvalue_H1(
    beta_hat, pvalue, pvalue_corrected, one_minus_pvalue, one_minus_pvalue_correlation
)
#############################################################################
# The results for the alternative hipothese shows that the confidence for not
# important variables is high. The 1-pvalues are not significant. However, the
# corrected 1-pvalue is close to one. We can conclude that the not importance
# variables are estimated in this case.

#############################################################################
#
# Principle of the methods
# ------------------------
# The ADA-SVR method is a statistical inference procedure that estimates the
# variable of importance of a Support Vector Regression (SVR).
# The method is a simplification of the permutation test for SVR
# (see :py:func:`hidimstat.permutation_test`).
# The principle is to shuffle the target variable and to estimate the variable
# of importance with the SVR in order to estimate the distribution of the
# coefficients of the SVR. For ADA-SVR, the distribution of the coefficients of
# the SVR is assumed to be a normal distribution centred around zeros. ADA-SVR
# uses the central limit theorem to estimate the
# standard deviation of this normal distribution for each coefficient
# (see the :ref:<following figure `fig:ada_svr_1`).
#
# .. figure:: ../../../examples/figures/ada_svr_1.jpg
#  :width: 400
#  :alt: Figure 1 of ::footcite:ct:`Gaonkar et al. 2012 <gaonkar_deriving_2012>`.
#  :align: center
#  :name: fig:ada_svr_1
#
#  Figure 1 of :footcite:ct:`Gaonkar et al. 2012 <gaonkar_deriving_2012>`.
#  **Left**: Concept of support vector machines in 2-D space
#  **Right**: Permutation testing for support vector machines
#

#############################################################################
# Comparison with the permutation test for SVR for validating the approach
# ------------------------------------------------------------------------
#
estimator = SVR(kernel="linear", epsilon=0.0, gamma="scale", C=1.0)
estimator.fit(X, y)
beta_hat_svr = estimator.coef_

# compare that the coefficiants are the same that the one of SVR
assert np.max(np.abs(beta_hat - beta_hat_svr.T[:, 0])) < 1e-4

#############################################################################
# This coefficient of SVR is an estimation of the importance of variables.
# For estimating the confidence interval of the importance of variables,
# ADA-SVR uses propose to estimate the distribution of the coefficients
# of SVR, instead of using the classical permutation test for this estimation.

#############################################################################
# Estimation of the distribution of the coefficients of SVR
# ---------------------------------------------------------
#

proba = permutation_test(
    X, y, estimator=estimator, n_permutations=10000, n_jobs=8, seed=42, proba=True
)
plot_compare_proba_estimated(proba, beta_hat, scale)

#############################################################################
# **Compare the distribution of the coefficients of SVR with the
# estimation of the distribution of the coefficients by ADA-SVR**
print(
    "ADA-SVR assumes that the normal distribution of the coefficient is",
    "center around zeros.\n",
    "Our estimation is that the maximum deviation is: {:.4f}\n".format(
        np.max(np.abs(np.mean(proba, axis=0)))
    ),
    "ADA-SVR provides that the standard deviation the normal distribution",
    "for each coefficients.\n",
    "The maximum of difference between AdA-SVR estimation and our estimation",
    "is:{:.4f}".format(np.max(np.abs(scale - np.std(proba, axis=0)) / scale)),
)
plt.show()

#############################################################################
# References
# ----------
# .. footbibliography::