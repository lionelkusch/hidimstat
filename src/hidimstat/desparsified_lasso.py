import numpy as np
from joblib import Parallel, delayed
from numpy.linalg import multi_dot
from scipy import stats
from scipy.linalg import inv
from sklearn.linear_model import Lasso

from hidimstat.noise_std import group_reid, reid
from hidimstat.stat_tools import pval_from_two_sided_pval_and_sign
from hidimstat.stat_tools import pval_from_cb


def desparsified_lasso(
    X,
    y,
    dof_ajdustement=False,
    confidence=0.95,
    max_iter=5000,
    tol=1e-3,
    alpha_max_fraction=0.01,
    tol_reid=1e-4,
    eps=1e-2,
    n_split=5,
    n_jobs=1,
    seed=0,
    verbose=0,
):
    """
    Desparsified Lasso with confidence intervals
    
    Algorithm based on Algorithm 1 of d-Lasso in :cite:`chevalier2020statistical`

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Data.

    y : ndarray, shape (n_samples,)
        Target variable.

    dof_ajdustement : bool, optional (default=False)
        If True, applies degrees of freedom adjustment as described in
        :footcite:`bellec2022biasing` and :footcite:`celentano2023lasso`.
        If False, computes original Desparsified Lasso estimator following
        :footcite:`zhang2014confidence`, :footcite:`van2014asymptotically` and
        :footcite:`javanmard2014confidence`.

    confidence : float, optional (default=0.95)
        Confidence level for intervals, must be in [0, 1].

    max_iter : int, optional (default=5000)
        Maximum iterations for Nodewise Lasso regressions.

    tol : float, optional (default=1e-3)
        Convergence tolerance for optimization.

    alpha_max_fraction : float, optional (default=0.01)
        Fraction of max lambda used for Lasso regularization.

    tol_reid : float, optional (default=1e-4)
        Tolerance for Reid estimation.

    eps : float, optional (default=1e-2) 
        Small constant used in noise estimation.

    n_split : int, optional (default=5)
        Number of splits for cross-validation in Reid procedure.

    n_jobs : int, optional (default=1)
        Number of parallel jobs. Use None for all CPUs.

    seed : int, optional (default=0)
        Random seed for reproducibility.

    verbose : int, optional (default=0)
        Verbosity level for logging.

    Returns
    -------
    beta_hat : ndarray, shape (n_features,)
        Desparsified Lasso coefficient estimates.

    cb_min : ndarray, shape (n_features,)
        Lower confidence interval bounds.

    cb_max : ndarray, shape (n_features,)
        Upper confidence interval bounds.

    Notes
    -----
    The columns of `X` and `y` are always centered, this ensures that
    the intercepts of the Nodewise Lasso problems are all equal to zero
    and the intercept of the noise model is also equal to zero. Since
    the values of the intercepts are not of interest, the centering avoids
    the consideration of unecessary additional parameters.
    Also, you may consider to center and scale `X` beforehand, notably if
    the data contained in `X` has not been prescaled from measurements.

    References
    ----------
    .. footbibliography::
    """

    X_ = np.asarray(X)

    n_samples, n_features = X_.shape

    # centering the data and the target variable
    y_ = y - np.mean(y)
    X_ = X_ - np.mean(X_, axis=0)

    # define the quantile for the confidence intervals
    quantile = stats.norm.ppf(1 - (1 - confidence) / 2)
    
    # Lasso regression and noise standard deviation estimation
    #TODO: other estimation of the noise standard deviation?
    sigma_hat, beta_lasso = reid(X_, y_, eps=eps, tol=tol_reid,
                                 max_iter=max_iter, n_split=n_split,
                                 n_jobs=n_jobs, seed=seed)

    # compute the Gram matrix
    gram = np.dot(X_.T, X_)
    gram_nodiag = np.copy(gram)
    np.fill_diagonal(gram_nodiag, 0)

    # define the alphas for the Nodewise Lasso
    #TODO why don't use the function _lambda_max instead of this?
    list_alpha_max = np.max(np.abs(gram_nodiag), axis=0) / n_samples
    alphas = alpha_max_fraction * list_alpha_max

    # Calculating precision matrix (Nodewise Lasso)
    Z, omega_diag = _compute_all_residuals(
        X_,
        alphas,
        gram,
        max_iter=max_iter,
        tol=tol,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    # Computing the degrees of freedom adjustement
    if dof_ajdustement:
        coef_max = np.max(np.abs(beta_lasso))
        support = np.sum(np.abs(beta_lasso) > 0.01 * coef_max)
        support = min(support, n_samples - 1)
        dof_factor = n_samples / (n_samples - support)
    else:
        dof_factor = 1

    # Computing Desparsified Lasso estimator and confidence intervals
    # Estimating the coefficient vector
    beta_bias = dof_factor * np.dot(y_.T, Z) / np.sum(X_ * Z, axis=0)

    # beta hat
    P = ((Z.T.dot(X_)).T / np.sum(X_ * Z, axis=0)).T
    P_nodiag = P - np.diag(np.diag(P))
    Id = np.identity(n_features)
    P_nodiag = dof_factor * P_nodiag + (dof_factor - 1) * Id
    beta_hat = beta_bias - P_nodiag.dot(beta_lasso)

    # confidence intervals
    omega_diag = omega_diag * dof_factor ** 2
    #TODO:why the double inverse of omega_diag?
    omega_invsqrt_diag = omega_diag ** (-0.5)
    confint_radius = np.abs(
        quantile * sigma_hat / (np.sqrt(n_samples) * omega_invsqrt_diag) 
    )
    cb_max = beta_hat + confint_radius
    cb_min = beta_hat - confint_radius

    return beta_hat, cb_min, cb_max


def desparsified_lasso_pvalue(cb_min, cb_max, confidence=0.95, distrib="norm", eps=1e-14):
    """
    Compute p-values for the desparsified Lasso estimator
    
    For details see: :py:func:`hidimstat.pval_from_cb`
    """ 
    pval, pval_corr, one_minus_pval, one_minus_pval_corr = (
        pval_from_cb(cb_min, cb_max, confidence=confidence, 
                     distrib=distrib, eps=eps)
    )
    
    return pval, pval_corr, one_minus_pval, one_minus_pval_corr
    

def desparsified_group_lasso(
    X,
    Y,
    cov=None,
    max_iter=5000,
    tol=1e-3,
    alpha_max_fraction=0.01,
    noise_method="AR",
    order=1,
    fit_Y=True,
    stationary=True,
    eps=True,
    tol_reid=1e-4,
    n_split=5,
    seed=0,
    n_jobs=1,
    verbose=0,
):
    """
    Desparsified Group Lasso

    Algorithm based Algorithm 1 of d-MTLasso in :cite:`chevalier2020statistical`
    

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Data.

    Y : ndarray, shape (n_samples, n_times)
        Target.

    cov : ndarray, shape (n_times, n_times), optional (default=None)
        If None, a temporal covariance matrix of the noise is estimated.
        Otherwise, `cov` is the temporal covariance matrix of the noise.

    max_iter : int, optional (default=5000)
        Maximum number of iterations for nodewise Lasso regressions.

    tol : float, optional (default=1e-3)
        Convergence tolerance for optimization.

    alpha_max_fraction : float, optional (default=0.01)
        Fraction of max lambda used for Lasso regularization.

    noise_method : {'AR', 'simple'}, optional (default='AR')
        Method to estimate noise covariance:
        - 'simple': Uses median correlation between consecutive timepoints
        - 'AR': Fits autoregressive model of specified order

    order : int, optional (default=1)
        Order of AR model when noise_method='AR'. Must be < n_times.

    fit_Y : bool, optional (default=True)
        Whether to fit Y in noise estimation.

    stationary : bool, optional (default=True)
        Whether to assume stationary noise in estimation.

    eps : float or bool, optional (default=True)
        Small constant for regularization.

    tol_reid : float, optional (default=1e-4)
        Tolerance for Reid estimation.

    n_split : int, optional (default=5)
        Number of splits for cross-validation for Reid procedure.

    seed : int, optional (default=0)
        Random seed for reproducibility.

    n_jobs : int, optional (default=1)
        Number of parallel jobs. Use None for all CPUs.

    verbose : int, optional (default=0)
        Verbosity level.

    Returns
    -------
    beta_hat : ndarray, shape (n_features, n_times)
        Desparsified group Lasso coefficient estimates.

    theta_hat : ndarray, shape (n_times, n_times)
        Estimated precision matrix.
        
    omega_diag : ndarray, shape (n_features,)
        Diagonal of covariance matrix.
        
    Notes
    -----
    The columns of `X` and the matrix `Y` are always centered, this ensures
    that the intercepts of the Nodewise Lasso problems are all equal to zero
    and the intercept of the noise model is also equal to zero. Since
    the values of the intercepts are not of interest, the centering avoids
    the consideration of unecessary additional parameters.
    Also, you may consider to center and scale `X` beforehand, notably if
    the data contained in `X` has not been prescaled from measurements.

    References
    ----------
    .. footbibliography::
    """

    X_ = np.asarray(X)

    n_samples, n_features = X_.shape
    n_times = Y.shape[1]

    if cov is not None and cov.shape != (n_times, n_times):
        raise ValueError(
            f'Shape of "cov" should be ({n_times}, {n_times}),'
            + f' the shape of "cov" was ({cov.shape}) instead'
        )

    # centering the data and the target variable
    Y_ = Y - np.mean(Y)
    X_ = X_ - np.mean(X_, axis=0)

    
    # Lasso regression and noise standard deviation estimation
    cov_hat, beta_mtl = group_reid(
        X_, Y_, method=noise_method, order=order, n_jobs=n_jobs,
        fit_Y=fit_Y, stationary=stationary, eps=eps, tol=tol_reid,
        max_iter=max_iter, n_split=n_split, seed=seed,
    )
    if cov is not None:
        cov_hat = cov
    theta_hat = n_samples * inv(cov_hat)

    # compute the Gram matrix
    gram = np.dot(X_.T, X_)
    gram_nodiag = np.copy(gram)
    np.fill_diagonal(gram_nodiag, 0)

    # define the alphas for the Nodewise Lasso
    #TODO why don't use the function _lambda_max instead of this?
    list_alpha_max = np.max(np.abs(gram_nodiag), axis=0) / n_samples
    alphas = alpha_max_fraction * list_alpha_max

    # Calculating precision matrix (Nodewise Lasso)
    Z, omega_diag = _compute_all_residuals(
        X_,
        alphas,
        gram,
        max_iter=max_iter,
        tol=tol,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    # Computing Desparsified Lasso estimator and confidence intervals
    # Estimating the coefficient vector
    beta_bias = Y_.T.dot(Z) / np.sum(X_ * Z, axis=0)

    # beta hat
    P = (np.dot(X_.T, Z) / np.sum(X_ * Z, axis=0)).T
    P_nodiag = P - np.diag(np.diag(P))
    beta_hat = beta_bias.T - P_nodiag.dot(beta_mtl.T)

    return beta_hat, theta_hat, omega_diag


def desparsified_group_lasso_pvalue(beta_hat, theta_hat, omega_diag, test="chi2"):
    """
    Compute p-values for the desparsified group Lasso estimator using chi-squared or F tests
    
    Parameters
    ----------
    beta_hat : ndarray, shape (n_features, n_times)
        Estimated parameter matrix from desparsified group Lasso.
        
    theta_hat : ndarray, shape (n_times, n_times) 
        Estimated precision matrix (inverse covariance).
        
    omega_diag : ndarray, shape (n_features,)
        Diagonal elements of the precision matrix.
        
    test : {'chi2', 'F'}, optional (default='chi2')
        Statistical test for computing p-values:
        - 'chi2': Chi-squared test (recommended for large samples)
        - 'F': F-test (better for small samples)
        
    Returns
    -------
    pval : ndarray, shape (n_features,)
        Raw p-values, numerically accurate for positive effects
        (p-values close to 0).
        
    pval_corr : ndarray, shape (n_features,)
        P-values corrected for multiple testing using
        Benjamini-Hochberg procedure.
        
    one_minus_pval : ndarray, shape (n_features,)
        1 - p-values, numerically accurate for negative effects 
        (p-values close to 1).
        
    one_minus_pval_corr : ndarray, shape (n_features,)
        1 - corrected p-values.

    Notes
    -----
    The chi-squared test assumes asymptotic normality while the F-test
    makes no such assumption and is preferable for small sample sizes.
    P-values are computed based on score statistics from the estimated
    coefficients and precision matrix.
    """
    n_features, n_times = beta_hat.shape
    n_samples = omega_diag.shape[0]
    
    # Compute the two-sided p-values
    if test == "chi2":
        chi2_scores = np.diag(multi_dot([beta_hat, theta_hat, beta_hat.T])) / omega_diag
        two_sided_pval = np.minimum(2 * stats.chi2.sf(chi2_scores, df=n_times), 1.0)
    elif test == "F":
        f_scores = (
            np.diag(multi_dot([beta_hat, theta_hat, beta_hat.T])) / omega_diag / n_times
        )
        two_sided_pval = np.minimum(
            2 * stats.f.sf(f_scores, dfd=n_samples, dfn=n_times), 1.0
        )
    else:
        raise ValueError(f"Unknown test '{test}'")
    
    # Compute the p-values
    sign_beta = np.sign(np.sum(beta_hat, axis=1))
    pval, pval_corr, one_minus_pval, one_minus_pval_corr = (
        pval_from_two_sided_pval_and_sign(two_sided_pval, sign_beta)
    )
    
    return pval, pval_corr, one_minus_pval, one_minus_pval_corr


def _compute_all_residuals(
    X, alphas, gram, max_iter=5000, tol=1e-3, n_jobs=1, verbose=0
):
    """
    Nodewise Lasso for computing residuals and precision matrix diagonal.
    
    For each feature, fits a Lasso regression against all other features
    to estimate the precision matrix and residuals needed for the
    desparsified Lasso estimator.
    
    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Input data matrix.
    
    alphas : ndarray, shape (n_features,)
        Lasso regularization parameters, one per feature.
    
    gram : ndarray, shape (n_features, n_features)
        Precomputed Gram matrix X.T @ X to speed up computations.
        
    max_iter : int, optional (default=5000)
        Maximum number of iterations for Lasso optimization.
        
    tol : float, optional (default=1e-3)
        Convergence tolerance for Lasso optimization.
        
    n_jobs : int or None, optional (default=1)
        Number of parallel jobs. None means using all processors.
        
    verbose : int, optional (default=0) 
        Controls the verbosity when fitting the models:
        0 = silent
        1 = progress bar
        >1 = more detailed output
        
    Returns
    -------
    Z : ndarray, shape (n_samples, n_features)
        Matrix of residuals from nodewise regressions.
        
    omega_diag : ndarray, shape (n_features,)
        Diagonal entries of the precision matrix estimate.
        
    Notes
    -----
    This implements the nodewise Lasso procedure from :cite:`van2014asymptotically`
    for estimating entries of the precision matrix needed in the 
    desparsified Lasso. The procedure regresses each feature against all others
    using Lasso to obtain residuals and precision matrix estimates.

    References
    ----------
    .. footbibliography::
    """

    n_samples, n_features = X.shape

    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_compute_residuals)(
            X=X,
            column_index=i,
            alpha=alphas[i],
            gram=gram,
            max_iter=max_iter,
            tol=tol,
        )
        for i in range(n_features)
    )

    # Unpacking the results
    results = np.asarray(results, dtype=object)
    Z = np.stack(results[:, 0], axis=1)
    omega_diag = np.stack(results[:, 1])

    return Z, omega_diag


def _compute_residuals(
    X, column_index, alpha, gram, max_iter=5000, tol=1e-3
):
    """
    Compute residuals and precision matrix diagonal element using nodewise Lasso regression
    
    For a given column of X, fits Lasso regression against all other columns to obtain 
    residuals and precision matrix diagonal entry needed for desparsified Lasso estimation.
    
    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Input data matrix, assumed to be centered.
        
    column_index : int
        Index of column to regress against all others.
        
    alpha : float
        Lasso regularization parameter for this regression.
        
    gram : ndarray, shape (n_features, n_features)
        Precomputed Gram matrix X.T @ X to speed up computations.
        
    max_iter : int, optional (default=5000) 
        Maximum iterations for Lasso optimization.
        
    tol : float, optional (default=1e-3)
        Convergence tolerance for optimization.
        
    Returns
    -------
    z : ndarray, shape (n_samples,)
        Residuals from regressing column_index against other columns.
        Used to construct desparsified estimator.
        
    omega_diag_i : float
        Estimated diagonal entry of precision matrix for this feature.
        Equal to n * ||z||^2 / <x_i, z>^2 where x_i is column_index of X.
        
    Notes
    -----
    Precomputes Gram matrix and uses sklearn's Lasso solver for efficiency.
    Residuals and precision estimates are needed to construct the
    bias-corrected estimator.
    """

    n_samples, n_features = X.shape
    i = column_index

    # Removing the column to regress against the others
    X_new = np.delete(X, i, axis=1)
    y_new = np.copy(X[:, i])

    # Method used for computing the residuals of the Nodewise Lasso.
    # here we use the Lasso method
    gram_ = np.delete(np.delete(gram, i, axis=0), i, axis=1)
    clf = Lasso(alpha=alpha, precompute=gram_, max_iter=max_iter, tol=tol)

    # Fitting the Lasso model and computing the residuals
    clf.fit(X_new, y_new)
    z = y_new  - clf.predict(X_new)

    # Computing the diagonal of the covariance matrix
    omega_diag_i = n_samples * np.sum(z**2) / np.dot(y_new, z) ** 2

    return z, omega_diag_i

