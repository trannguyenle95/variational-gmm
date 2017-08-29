import numpy as np
from scipy.stats import invwishart


class GMMDataGenerator(object):

    def __init__(self, k, d, mu_interval=(-10, 10), mu_var=(-4, 4),
                 cov_var=(-1, 1), gamma_0=3.0, alpha_0=5.0):
        """
        Args:
            k (int): number of components per class.
            d (int): dimension of data.
        """
        self.k = k
        self.d = d
        self._mu_0 = np.zeros(d)
        self.gamma_0 = gamma_0
        self.alpha_0 = alpha_0
        self.mu_interval = mu_interval
        self.mu_var = mu_var
        self.cov_var = cov_var
        self.mu = np.zeros((k, d))
        self.cov = np.zeros((k, d, d))
        self.sample_cov()
        self.sample_mean()
        self.weights = np.random.dirichlet(self.gamma_0 * np.ones(k))

    def generate(self, n=2000):
        X = np.zeros((n, self.d))
        Z = np.zeros(n)

        # generate the component distributions
        self.pi = np.random.dirichlet(self.alpha_0 * np.ones(self.k))
        for i in range(n):
            # generate random component of this observation
            z_i = np.argmax(np.random.multinomial(1, self.pi))
            Z[i] = z_i

            # generate the features
            X[i, :] = np.random.multivariate_normal(self.mu[z_i, :],
                                                    self.cov[z_i, :])
        return X, Z.astype(int)

    def sample_cov(self):
        for j in range(self.k):
            self.cov[j, :] = invwishart.rvs(2 * self.d, np.eye(self.d))

    def sample_from_component(self, k):
        return np.random.multivariate_normal(self.mu[k, :],
                                             self.cov[k, :])

    def get_mu(self):
        return self.mu

    def get_cov(self):
        return self.cov

    def sample_mean(self):
        a, b = self.mu_interval
        std_min, std_max = self.cov_var
        mu_center = (b - a) * np.random.random(size=self.d) + a
        mu_max = mu_center + self.mu_var[1]
        mu_min = mu_center + self.mu_var[0]
        for j in range(self.k):
            mu_center_k = ((mu_max - mu_min)
                           * np.random.random(size=self.d)
                           + mu_min)
            self.mu[j, :] = np.random.multivariate_normal(
                self._mu_0 + mu_center_k, self.cov[j, :])
