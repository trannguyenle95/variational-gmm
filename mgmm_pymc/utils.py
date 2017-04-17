import theano.tensor as tt
import numpy as np


def lkj_cov(theta, C):
    """
    Return a covariance matrix from a standard deviation vector theta and
    a correlation matrix C.

    \Sigma = diag(theta) * C * diag(theta)

    Args:
        theta: Standard deviation vector.
        C: Correlaton matrix.

    """
    diagonal_theta = tt.diag(theta)
    return diagonal_theta.dot(C).dot(diagonal_theta)


def gaussian_log_likelihood(mu, lamb, X):
    """
    Return the log likelihood from a multivariate gaussian.

    Source: pymc3/distributions/multivariate.py
    """
    delta = X - mu
    k = lamb.shape[0]
    result = k * tt.log(2 * np.pi) + tt.log(1. / tt.nlinalg.det(lamb))
    result += (delta.dot(lamb) * delta).sum(axis=delta.ndim - 1)
    return -1 / 2. * result
