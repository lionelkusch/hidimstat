import numpy as np
from hidimstat._utils.regression import _alpha_max
from hidimstat._utils.scenario import multivariate_1D_simulation_AR

seed = 42


def test_alpha_max():
    """Test alpha max function"""
    n = 500
    p = 100
    snr = 5
    X, y, beta_true, non_zero = multivariate_1D_simulation_AR(n, p, snr=snr, seed=0)
    max_alpha = _alpha_max(X, y)
    max_alpha_noise = _alpha_max(X, y, use_noise_estimate=True)
    # Assert alpha_max is positive
    assert max_alpha > 0
    assert max_alpha_noise > 0

    # Assert alpha_max with noise estimate is different (usually smaller)
    assert max_alpha_noise != max_alpha

    # Test with zero target vector
    y_zero = np.zeros_like(y)
    alpha_zero = _alpha_max(X, y_zero)
    assert alpha_zero == 0
