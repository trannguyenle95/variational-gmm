import datetime
import logging
import time

import numpy as np
from scipy.misc import logsumexp
from scipy.special import gammaln
from scipy.special.basic import digamma
from scipy.stats import wishart, multivariate_normal


logger = logging.getLogger(__name__)


def log_C(alpha):
    """Return the logarithm of the C(alpha) function (eq. B.23 of Bishop)."""
    return gammaln(np.sum(alpha)) - np.sum(gammaln(alpha))


def log_B(W, nu):
    """Return the logarithm of the B(W, nu) function (eq. B.79 of Bishop).

       Parameters:
       W -- D x D symmetric positive definite matrix.
       nu -- number of degrees of freedom of the distribution, restricted to
             nu > D - 1 to ensure that the Gamma function in the normalization
             factor is well defined.
    """
    D = W.shape[1]
    q1 = -.5 * nu * np.log(np.linalg.det(W))
    q2 = (nu * D / 2. * np.log(2.)
          + D * (D - 1) / 4. * np.log(np.pi)
          + np.sum(gammaln(.5 * (nu - np.arange(D)))))
    return q1 - q2


class VariationalGMM:

    def __init__(self, n_components, n_features, **kwargs):
        self.n_components = n_components
        self.n_features = n_features

        # Hyperparameters initialization
        self.alpha_0 = kwargs.get('alpha_0', .1)
        self.beta_0 = kwargs.get('beta_0', 1e-20)
        self.nu_0 = kwargs.get('nu_0', self.n_features + 1.)
        self.m_0 = kwargs.get('m_0', np.zeros(self.n_features))
        self.W_0 = kwargs.get('W_0', np.eye(self.n_features))
        self.inv_W_0 = np.linalg.inv(self.W_0)
        self.responsibilities = None

        # model params
        self.W_k = None
        self.nu_k = None
        self.alpha_k = None
        self.beta_k = None
        self.m_k = None

    def fit(self, X, max_iter=100, elbo_epsilon=1e-5, logging_interval=5):
        logger.info('Start fitting GMM in data')
        assert self.n_features == X.shape[1]

        self.max_iter = max_iter
        self.N = np.shape(X)[0]
        self.X = X
        self.responsibilities = np.random.dirichlet(np.ones(self.n_components),
                                                    size=self.N)
        self.elbo_per_iter = []
        self.checkpoints = []
        start_time_secs = time.time()
        for i in range(self.max_iter):
            self._m_step()
            self._e_step()
            elbo = self._calculate_elbo()
            self.elbo_per_iter.append(elbo)
            self.checkpoints.append(self.get_checkpoint())

            if i % logging_interval == 0:
                logger.info('Iteration: %d', i)
                logger.info('ELBO: %f', elbo)

            if len(self.elbo_per_iter) > 1 and elbo < self.elbo_per_iter[-2]:
                logger.warning('ELBO IS DECREASING!')

            if (len(self.elbo_per_iter) > 1 and
                    abs(elbo - self.elbo_per_iter[-2]) < elbo_epsilon):
                break
        elapsed_time_secs = time.time() - start_time_secs
        time_delta = datetime.timedelta(seconds=elapsed_time_secs)
        logger.info('Finished inference. Elapsed time: %s',
                    '{} secs'.format(time_delta.seconds))
        logger.debug('Variational mu:\n%s', np.array_str(self.m_k))
        logger.debug('Variational W_k:\n%s', np.array_str(self.W_k))

    def predict(self, x):
        # tmp = (self.nu_k + 1 - self.n_features) * \
        #     self.beta_k / (self.beta_k + 1)
        # L_k = np.linalg.inv(tmp[:, np.newaxis, np.newaxis] * self.W_k)
        # assert L_k.shape == self.W_k.shape

        alpha_hat = np.sum(self.alpha_k)

        expected_lambda_inv_k = np.linalg.inv(
            self.nu_k[:, np.newaxis, np.newaxis] * self.W_k)

        predictive = .0
        for k in range(self.n_components):
            predictive += self.alpha_k[k] * \
                multivariate_normal.pdf(x,
                                        mean=self.m_k[k],
                                        cov=expected_lambda_inv_k[k])
        predictive /= alpha_hat

        return predictive

    def get_variational_parameters(self):
        parameters = dict()
        parameters['W_k'] = self.W_k
        parameters['nu_k'] = self.nu_k
        parameters['alpha_k'] = self.alpha_k
        parameters['beta_k'] = self.beta_k
        parameters['m_k'] = self.m_k
        return parameters

    def get_checkpoint(self):
        checkpoint = dict()
        checkpoint['variational_parameters'] = \
            self.get_variational_parameters()
        checkpoint['elbo'] = self.elbo_per_iter[-1]
        checkpoint['iteration'] = len(self.checkpoints)
        checkpoint['pi_k_expectation'] = self.calculate_E_pi_k()
        return checkpoint

    def get_results(self):
        out = dict()
        out['hyperparameters'] = {
            'alpha_0': self.alpha_0,
            'beta_0': self.beta_0,
            'nu_0': self.nu_0,
            'm_0': self.m_0,
            'W_0': self.W_0
        }
        out['iterations'] = self.max_iter
        out['objects'] = self.N
        out['checkpoints'] = self.checkpoints
        out['variational_parameters'] = self.get_variational_parameters()
        return out

    def calculate_E_pi_k(self):
        return (self.alpha_k + self._N_k) / (self.n_components * self.alpha_0 +
                                             self.N)

    def _m_step(self):
        # _N_k is a K vector
        self._N_k = self.responsibilities.sum(axis=0) + 1e-8
        assert self._N_k.shape == (self.n_components,)

        # nu_k is a K vector
        self.nu_k = self.nu_0 + self._N_k
        assert self.nu_k.shape == (self.n_components,)

        # alpha_k is a K vector
        self.alpha_k = self.alpha_0 + self._N_k
        assert self.alpha_k.shape == (self.n_components,)

        # _x_k_hat is a (K, D) matrix
        self._x_k_hat = (1. / self._N_k[:, np.newaxis]
                         * self.responsibilities.T.dot(self.X))
        assert self._x_k_hat.shape == (self.n_components, self.n_features)

        self._S_k = self._calculate_S_k()  # (K, D, D) tensor
        assert self._S_k.shape == (self.n_components, self.n_features,
                                   self.n_features)

        # beta_k is a K vector
        self.beta_k = self.beta_0 + self._N_k
        assert self.beta_k.shape == (self.n_components,)

        # m_k is a (K, D) matrix
        self.m_k = (1. / self.beta_k[:, np.newaxis]
                    * (self.beta_0 * self.m_0
                       + self._N_k[:, np.newaxis] * self._x_k_hat))
        assert self.m_k.shape == (self.n_components, self.n_features)

        # W_k is a (K, D, D) tensor
        self.W_k = self._calculate_W_k()
        assert self.W_k.shape == (self.n_components, self.n_features,
                                  self.n_features)

    def _e_step(self):
        # (K, N) matrix
        self._expec_mu_lambda = self._calculate_E_mu_lambda()
        assert self._expec_mu_lambda.shape == (self.n_components, self.N)

        # K vector
        self._expec_log_det_lambda = self._calculate_E_log_det_lambda()
        assert self._expec_log_det_lambda.shape == (self.n_components,)

        # K vector
        self._expec_log_pi_k = (digamma(self.alpha_k)
                                - digamma(np.sum(self.alpha_k)))
        assert self._expec_log_pi_k.shape == (self.n_components,)

        self._reestimate_responsibilities()

    def _reestimate_responsibilities(self):
        ln_rho = (self._expec_log_pi_k[:, np.newaxis]
                  + .5 * self._expec_log_det_lambda[:, np.newaxis]
                  - .5 * self.n_features * np.log(2 * np.pi)
                  - .5 * self._expec_mu_lambda)
        log_responsibilities = ln_rho - logsumexp(ln_rho, axis=0)
        self.responsibilities = np.exp(log_responsibilities)

        # Ensure that responsibilities are a (N, K) matrix
        self.responsibilities = np.transpose(self.responsibilities)
        assert self.responsibilities.shape == (self.N, self.n_components)

    def _calculate_elbo(self):
        # Eq. 10.71
        E_ln_p_X = self._calculate_expected_log_likelihood()

        # Eq. 10.72
        E_ln_p_Z = np.sum(self.responsibilities * self._expec_log_pi_k)

        # Eq. 10.73
        E_ln_p_pi = (log_C(self.alpha_0 * np.ones(self.n_components))
                     + (self.alpha_0 - 1) * np.sum(self._expec_log_pi_k))

        # Eq. 10.74
        E_ln_p_mu_lamb = self._calculate_expected_log_p_mu_lamb()

        # Eq. 10.75
        E_ln_q_Z = np.sum(self.responsibilities
                          * np.log(self.responsibilities))

        # Eq. 10.76
        E_ln_q_pi = (np.sum((self.alpha_k - 1) * self._expec_log_pi_k)
                     + log_C(self.alpha_k))

        # Eq. 10.77
        E_ln_q_mu_lamb = self._calculate_expected_log_q_mu_lamb()

        return (E_ln_p_X + E_ln_p_Z + E_ln_p_pi + E_ln_p_mu_lamb
                - E_ln_q_Z - E_ln_q_pi - E_ln_q_mu_lamb)

    def _calculate_S_k(self):
        """
        Return (K, D, D) tensor.
        Eq. 10.53 from Bishop
        """
        # (N, K, D) tensor
        normalizer = self.X[:, np.newaxis] - self._x_k_hat
        # (D, N, K) tensor
        normalizer = np.transpose(normalizer, axes=[2, 0, 1])

        # (K, D, N) tensor
        trans_normalizer = np.transpose(normalizer, axes=[2, 0, 1])

        # Multiply responsibilities for each (K, N) D vector.
        prod = self.responsibilities[np.newaxis, :] * normalizer
        prod = np.transpose(prod, axes=[2, 1, 0])  # (K, N, D) tensor

        # K dot products of dimensions (D, N) x (N, D) to get K (D, D) matrices
        # (K, D, D) tensor
        k_d_d_matrixes = np.einsum('lij, ljk -> lik',
                                   trans_normalizer,
                                   prod)

        return 1. / self._N_k[:, np.newaxis, np.newaxis] * k_d_d_matrixes

    def _calculate_W_k(self):
        """
        Return (K, D, D) tensor.
        Eq. 10.62 from Bishop
        """
        temp1 = self._x_k_hat - self.m_0
        # (K, D, D) tensor
        temp = np.einsum('ij, ik -> ijk', temp1, temp1)
        assert temp.shape == (self.n_components, self.n_features,
                              self.n_features)

        temp2 = self.beta_0 * self._N_k / (self.beta_0 + self._N_k)
        # We have the inverted W_k
        inv_W_k = (self.inv_W_0[np.newaxis, :]
                   + self._N_k[:, np.newaxis, np.newaxis] * self._S_k
                   + temp2[:, np.newaxis, np.newaxis] * temp)
        # Invert inv_W_k
        return np.linalg.inv(inv_W_k)

    def _calculate_E_mu_lambda(self):
        """
        Return (K, N) matrix.
        Eq. 10.64 from Bishop.
        """
        diff = self.X[:, np.newaxis] - self.m_k  # (N, K, D) tensor

        # (K, N, D) tensor
        diff = np.transpose(diff, axes=[1, 0, 2])

        # Take the dot product from each (N, D) vector with the (D, D) matrix.
        # The result is a (K, N, D) tensor
        first_dot = np.einsum('lij, ljk -> lik', diff, self.W_k)

        mul = np.sum(first_dot * diff, axis=2)  # (K, N) matrix.
        return (self.n_features / self.beta_k[:, np.newaxis]
                + self.nu_k[:, np.newaxis] * mul)

    def _calculate_E_log_det_lambda(self):
        """
        Return K vector.
        Eq. 10.65 from Bishop.
        """
        # The original formula is nu_k + 1 - (np.arange(D) + 1), but we can
        # simplify it.
        temp0 = self.nu_k[:, np.newaxis] - np.arange(self.n_features)
        temp1 = digamma(temp0 / 2.)
        E_log_det_lambda = (np.sum(temp1, axis=1)
                            + self.n_features * np.log(2)
                            + np.log(np.linalg.det(self.W_k)))
        return E_log_det_lambda

    def _calculate_expected_log_likelihood(self):
        """
        Return blah.
        Eq. 10.71 from Bishop.
        """
        E_ln_p_X = 0.0
        for k in range(self.n_components):
            tmp1 = self.nu_k[k] * np.trace(np.dot(self._S_k[k], self.W_k[k]))
            diff = self._x_k_hat[k] - self.m_k[k]
            tmp2 = self.nu_k[k] * np.dot(diff, self.W_k[k]).dot(diff.T)
            E_ln_p_X += self._N_k[k] * (self._expec_log_det_lambda[k]
                                        - self.n_features / self.beta_k[k]
                                        - tmp1
                                        - tmp2
                                        - self.n_features * np.log(2 * np.pi))
        return E_ln_p_X / 2.

    def _calculate_expected_log_p_mu_lamb(self):
        q1 = np.sum(self._expec_log_det_lambda)
        q2 = 0.0
        q3 = 0.0
        for k in range(self.n_components):
            q3 += self.nu_k[k] * np.trace(np.dot(self.inv_W_0, self.W_k[k]))
            diff = self.m_k[k] - self.m_0
            q2_1 = np.dot(diff, self.W_k[k]).dot(diff.T)
            q2 += (self.n_features * np.log(self.beta_0 / (2 * np.pi))
                   + self._expec_log_det_lambda[k]
                   - self.n_features * self.beta_0 / self.beta_k[k]
                   - self.beta_0 * self.nu_k[k] * q2_1)
        E_ln_p_mu_lamb = (.5 * q2
                          + self.n_components * log_B(self.W_0, self.nu_0)
                          + .5 * (self.nu_0 - self.n_features - 1) * q1
                          - .5 * q3)
        return E_ln_p_mu_lamb

    def _calculate_expected_log_q_mu_lamb(self):
        E_ln_q_mu_lamb = 0.0
        for k in range(self.n_components):
            E_ln_q_mu_lamb += (.5 * self._expec_log_det_lambda[k]
                               + (.5 * self.n_features
                                  * np.log(self.beta_k[k] / (2 * np.pi)))
                               - .5 * self.n_features
                               - wishart.entropy(scale=self.W_k[k],
                                                 df=self.nu_k[k]))
        return E_ln_q_mu_lamb
