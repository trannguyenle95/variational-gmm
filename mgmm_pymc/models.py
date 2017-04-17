import pymc3 as pm
import theano.tensor as tt
import numpy as np
from .utils import lkj_cov, gaussian_log_likelihood


class GaussianMixtureModel:

    def __init__(self, K, observed, **kwargs):
        """
        Args:
            gamma_0: GMM mixture dirichlet hyperparameter.
            mu_0: Gaussian prior mean hyperparameter.
            nu_0: LKJ correlation shape. For n -> oo the LKJ prior approaches
                  the identity matrix.
        """
        self.K = K
        self.observed = observed
        self.N = self.observed.shape[0]
        self.D = self.observed.shape[1]
        self.gamma_0 = kwargs.get('gamma_0', np.ones(self.K))
        self.mu_0 = kwargs.get('mu_0', np.zeros(self.D))
        self.nu_0 = kwargs.get('nu_0', 3.)
        self.mu_means = None
        self.rho_means = None
        self.lambda_means = None
        self._setup_model()

    def _mu_means(self, trace):
        mu_means = np.zeros((self.K, self.D))
        for k in range(self.K):
            mu_means[k, :] = trace.get_values('mu_{}'.format(k)).mean(axis=0)
        return mu_means

    def _rho_means(self, trace):
        return trace.get_values('rho').mean(axis=0)

    def _lamb_means(self, trace):
        lamb_means = np.zeros((self.K, self.D, self.D))
        cov_stds_m_mean = trace.get_values('cov_stds').mean(axis=0)
        for k in range(self.K):
            corr_vec_mk_mean = trace.get_values('corr_vec_{}'.format(k)).mean(
                axis=0)
            corr = np.copy(corr_vec_mk_mean[self._tri_index])
            np.fill_diagonal(corr, 1.)
            # lamb_means[k, :] = lkj_cov(cov_stds_m_mean[k], corr)
            std_diag = np.diag(cov_stds_m_mean[k])
            cov = std_diag.dot(corr).dot(std_diag)
            lamb_means[k, :] = cov
        return lamb_means

    def train(self, samples=2e4, burn_in=5e3):
        with self.model:
            step = pm.Metropolis(vars=[self.rho] + self.mu + self.lambdas)
            trace = pm.sample(samples, step=[step])[burn_in:]
            self.mu_means = self._mu_means(trace)
            self.rho_means = self._rho_means(trace)
            self.lambda_means = self._lamb_means(trace)
            return trace

    def _setup_model(self):
        def model_log_likelihood(rho, mu, lamb):
            def log_likelihood(observed):
                # Get the log-likelihood of each of the K gaussians
                likelihoods = [gaussian_log_likelihood(mu_k,
                                                       lambda_k,
                                                       observed)
                               for mu_k, lambda_k in zip(mu, lamb)]
                # Sum pi_k to every gaussian
                likelihoods = tt.stack(likelihoods).T + tt.log(rho)  # (N x K)
                # Get the log of the sum of the likelihoods for the K
                # gaussians.
                likelihoods = pm.math.logsumexp(likelihoods, axis=1)
                # Sum the log-likelihood for all observations
                return tt.sum(likelihoods)
            return log_likelihood

        observed_mu = self.observed.mean(axis=0)

        # In order to convert the upper triangular correlation values to a
        # complete correlation matrix, we need to construct an index matrix.
        # Source: pymc3/examples/LKJ_correlation.py
        n_elems = self.D * (self.D - 1) / 2.
        self._tri_index = np.zeros([self.D, self.D], dtype=int)
        self._tri_index[np.triu_indices(self.D, k=1)] = np.arange(n_elems)
        self._tri_index[np.triu_indices(self.D, k=1)[::-1]] = np.arange(n_elems)

        self.model = pm.Model()
        with self.model:
            corr_vecs = [pm.LKJCorr('corr_vec_{}'.format(k),
                                    self.nu_0,
                                    self.D)
                         for k in range(self.K)]
            corrs = [tt.fill_diagonal(corr_vecs[k][self._tri_index], 1.)
                     for k in range(self.K)]
            cov_stds = pm.Lognormal('cov_stds',
                                    np.zeros(self.D),
                                    np.ones(self.D),
                                    shape=(self.K, self.D))

            self.lambdas = [lkj_cov(cov_stds[k], corrs[k])
                            for k in range(self.K)]

            self.mu = [pm.MvNormal('mu_{}'.format(k),
                                   mu=observed_mu,
                                   tau=self.lambdas[k],
                                   shape=self.D)
                       for k in range(self.K)]

            self.rho = pm.Dirichlet('rho', self.gamma_0, shape=self.K)

            self.x = pm.DensityDist('x',
                                    model_log_likelihood(self.rho,
                                                         self.mu,
                                                         self.lambdas),
                                    observed=self.observed)

