import numpy as np
from scipy.special import gammaln
from scipy.special.basic import digamma


def log_C(alpha):
    return gammaln(np.sum(alpha)) - np.sum(gammaln(alpha))


def log_B(W, nu):
    D = W.shape[1]
    q1 = -.5 * nu * np.linalg.det(W)
    q2 = (.5 * nu * D
          + .25 * D * (D - 1) * np.pi
          + np.sum(gammaln(.5 * (nu - np.arange(D)))))
    return q1 - q2


def wishart_entropy(W, nu):
    D = W.shape[1]
    q1 = (np.sum(digamma(.5 * (nu - np.arange(D))))
          + D * np.log(2)
          + np.log(np.linalg.det(W)))
    entropy = -log_B(W, nu) - .5 * (nu - D - 1) * q1 + .5 * nu * D
    return entropy
