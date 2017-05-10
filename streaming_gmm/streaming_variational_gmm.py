import numpy as np
from vb_mgmm.var_bayes_gmm import VariationalGMM


class StreamingVariationalGMM:

    def __init__(self):
        self.checkpoints = []

    def get_checkpoint(self):
        checkpoint = {
            'beta_k': self.beta_k,
            'm_k': self.m_k,
            'W_k': self.W_k,
            'nu_k': self.nu_k,
            'alpha_k': self.alpha_k
        }
        return checkpoint

    def update_with_new_data(self):
        pass

    def update_with_data_shake(self, X_old, X_new):
        alpha_k_old_data, normal_wishart_params_old_data = \
            self._get_variational_parameters(X_old)
        alpha_k_new_data, normal_wishart_params_new_data = \
            self._get_variational_parameters(X_new)
        self._dirichlet_update_parameter(alpha_k_old_data, alpha_k_new_data)
        self._normal_wishart_update_parameter(normal_wishart_params_old_data,
                                              normal_wishart_params_new_data)
        self.checkpoints.append(self.get_checkpoint())
        pass

    def _dirichlet_update_parameter(self, alpha_k_old_data, alpha_k_new_data):
        self.alpha_k = self.alpha_k - alpha_k_old_data + alpha_k_new_data
        self.alpha_k_per_iter.append(self.alpha_k)

    def _normal_wishart_update_parameter(self,
                                         normal_wishart_params_old_data,
                                         normal_wishart_params_new_data):
        self.beta_k = self._compute_update_beta_k(
            normal_wishart_params_old_data['beta_k'],
            normal_wishart_params_new_data['beta_k'])
        self.m_k = self._update_m_k(
            normal_wishart_params_old_data['beta_k_m_k'],
            normal_wishart_params_new_data['beta_k_m_k'])
        self._beta_k_m_k = self.beta_k * self.m_k
        self._inv_W_k = self._compute_update_inv_W_k(
            normal_wishart_params_old_data['inv_W_k_expression'],
            normal_wishart_params_new_data['inv_W_k_expression'])
        self._inv_W_k_expression = (self._inv_W_k
                                    + self.beta_k * np.sum(self.m_k ** 2))
        self.W_k = np.linalg.inv(self._inv_W_k)
        self.nu_k = self._compute_update_nu_k(
            normal_wishart_params_old_data['nu_k'],
            normal_wishart_params_new_data['nu_k'])

    def _compute_update_m_k(self, beta_k_m_k_old_data, beta_k_m_k_new_data):
        return 1. / self.beta_k * (self._beta_k_m_k
                                   - beta_k_m_k_old_data
                                   + beta_k_m_k_new_data)

    def _compute_update_beta_k(self, beta_k_old_data, beta_k_new_data):
        return self.beta_k - beta_k_old_data + beta_k_new_data

    def _compute_update_inv_W_k(self,
                                inv_W_k_expression_old_data,
                                inv_W_k_expression_new_data):
        return (self._inv_W_k_expression
                - inv_W_k_expression_old_data
                + inv_W_k_expression_new_data
                - self.beta_k * np.sum(self.m_k ** 2))

    def _compute_update_nu_k(self, nu_k_old_data, nu_k_new_data):
        return self.nu_k - nu_k_old_data + nu_k_new_data

    def _get_variational_parameters(self, X):
        variational_gmm = VariationalGMM(self.n_components, self.n_features)
        variational_gmm.fit(X)
        variational_parameters = variational_gmm.get_variational_parameters()
        alpha_k = variational_parameters['alpha_k']
        normal_wishart_params = {k: v
                                 for k, v in variational_parameters.items()
                                 if not k == 'alpha_k'}
        return (alpha_k, normal_wishart_params)
