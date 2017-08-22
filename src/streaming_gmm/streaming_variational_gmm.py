import numpy as np
from .variational_gmm import VariationalGMM
import logging


logger = logging.getLogger(__name__)


def sort_dictionary_by_key(dictionary):
    return [(k, dictionary[k]) for k in sorted(dictionary, key=dictionary.get)]


def get_closest_components(map_from_k_to_list, n_components):
    logger.debug('map_from_k_to_list: %s', map_from_k_to_list)
    map_from_real_to_not_identified = dict()
    already_chosen = set()
    for k in range(n_components):
        for i in range(n_components):
            if map_from_k_to_list[k][i][0] in already_chosen:
                continue
            is_i_candidate = True
            for j in range(k + 1, n_components):
                potential_k = map_from_k_to_list[k][i]
                potential_j = map_from_k_to_list[j][i]
                if (potential_j[0] == potential_k[0] and
                        potential_j[1] < potential_k[1]):
                    is_i_candidate = False
                    break

            if is_i_candidate:
                chosen = map_from_k_to_list[k][i][0]
                map_from_real_to_not_identified[k] = chosen
                already_chosen.add(chosen)
                break
    map_from_not_identified_to_real = {
        v: k for k, v in map_from_real_to_not_identified.items()}
    logger.debug('Mapped elements: %s', map_from_not_identified_to_real)
    return map_from_not_identified_to_real


def identify_components(real_m_k, not_identified_m_k):
    if real_m_k.shape != not_identified_m_k.shape:
        raise ValueError("input need to have the same shape")

    logger.debug('identify_components - real_m_k:\n%s',
                 np.array_str(real_m_k))
    logger.debug('identify_components - not iden:\n%s',
                 np.array_str(not_identified_m_k))
    n_components = real_m_k.shape[0]

    not_identified_to_real_distances = dict()
    for k in range(n_components):
        component_k_closest_real = dict()
        for j in range(n_components):
            distance = np.linalg.norm(not_identified_m_k[j] - real_m_k[k])
            component_k_closest_real[j] = distance
        not_identified_to_real_distances[k] = sort_dictionary_by_key(
            component_k_closest_real)

    return get_closest_components(not_identified_to_real_distances,
                                  n_components)


def swap_elements_at(map_from_to, numpy_array):
    tmp_array = np.zeros_like(numpy_array)
    for from_, to in map_from_to.items():
        tmp_array[to] = numpy_array[from_]
    return tmp_array


class StreamingVariationalGMM:

    def __init__(self, n_components, n_features, **kwargs):
        self.checkpoints = []
        self.batch_checkpoints = []
        self.n_components = n_components
        self.n_features = n_features

        self.mixing_coefficient_threshold = 1e-4

        # For now all components have the same hyperparameters.
        self.alpha_0 = kwargs.get('alpha_0', .1)
        self.beta_0 = kwargs.get('beta_0', 1e-20)
        self.nu_0 = kwargs.get('nu_0', self.n_features + 1.)
        self.m_0 = kwargs.get('m_0', np.zeros(self.n_features))
        self.W_0 = kwargs.get('W_0', np.eye(self.n_features))

        self.alpha_0_k = np.tile(self.alpha_0, self.n_components)
        self.beta_0_k = np.tile(self.beta_0, self.n_components)
        self.nu_0_k = np.tile(self.nu_0, self.n_components)
        self.m_0_k = np.tile(self.m_0, (self.n_components, 1))
        self.W_0_k = np.tile(self.W_0, (self.n_components, 1, 1))

        # We precompute this because we will need it to make the updates.
        self._beta_0_k_m_0_k = self._compute_beta_k_m_k(self.m_0_k,
                                                        self.beta_0_k)
        self._inv_W_0_k_expression = self._compute_inv_W_k_expression({
            'W_k': self.W_0_k,
            'm_k': self.m_0_k,
            'beta_k': self.beta_0_k})

        # The variational parameters start with the same value as the prior.
        self.alpha_k = np.copy(self.alpha_0_k)
        self.beta_k = np.copy(self.beta_0_k)
        self.nu_k = np.copy(self.nu_0_k)
        self.m_k = np.copy(self.m_0_k)
        self.W_k = np.copy(self.W_0_k)
        self._beta_k_m_k = np.copy(self._beta_0_k_m_0_k)
        self._inv_W_k_expression = np.copy(self._inv_W_0_k_expression)

    def get_checkpoint(self):
        checkpoint = dict()
        checkpoint['streaming_variational_parameters'] = {
            'beta_k': self.beta_k,
            'm_k': self.m_k,
            'W_k': self.W_k,
            'nu_k': self.nu_k,
            'alpha_k': self.alpha_k
        }
        checkpoint['batch_checkpoints'] = self.batch_checkpoints
        return checkpoint

    def update_with_new_data(self, X_new):
        alpha_k_new_data, normal_wishart_k_new_data = \
            self._get_variational_parameters(X_new)
        self._dirichlet_update_parameter(alpha_k_new_data)
        self._normal_wishart_update_parameter(normal_wishart_k_new_data)
        self.checkpoints.append(self.get_checkpoint())

    def update_with_data_shake(self, X_old, X_new):
        alpha_k_old_data, normal_wishart_params_old_data = \
            self._get_variational_parameters(X_old)
        alpha_k_new_data, normal_wishart_params_new_data = \
            self._get_variational_parameters(X_new)
        self._dirichlet_update_parameter(alpha_k_old_data, alpha_k_new_data)
        self._normal_wishart_update_parameter(normal_wishart_params_old_data,
                                              normal_wishart_params_new_data)
        self.checkpoints.append(self.get_checkpoint())

    def _get_variational_parameters(self, X):
        variational_gmm = VariationalGMM(self.n_components,
                                         self.n_features,
                                         alpha_0=self.alpha_0,
                                         beta_0=self.beta_0,
                                         nu_0=self.nu_0,
                                         m_0=self.m_0,
                                         W_0=self.W_0)
        variational_gmm.fit(X, max_iter=20)
        variational_parameters = variational_gmm.get_variational_parameters()
        self._ignore_component_with_low_mixing_coef(
            variational_parameters, variational_gmm.calculate_E_pi_k())

        return self._align_components(variational_parameters)

    def _ignore_component_with_low_mixing_coef(self, variational_parameters,
                                               mixing_coefficients):
        for k in range(self.n_components):
            if mixing_coefficients[k] < self.mixing_coefficient_threshold:
                logger.debug('Ignoring component: %d', k)
                variational_parameters['alpha_k'][k] = self.alpha_0
                variational_parameters['beta_k'][k] = self.beta_0
                variational_parameters['m_k'][k] = self.m_0
                variational_parameters['nu_k'][k] = self.nu_0
                variational_parameters['W_k'][k] = self.W_0
        return variational_parameters

    def _align_components(self, variational_parameters):
        correct_component_indexes = identify_components(
            self.m_k, variational_parameters['m_k'])

        logger.debug('correct components: %s', correct_component_indexes)

        alpha_k = variational_parameters['alpha_k']
        alpha_k = swap_elements_at(correct_component_indexes, alpha_k)

        normal_wishart_params = dict()
        for key, value in variational_parameters.items():
            if key == 'alpha_k':
                continue
            normal_wishart_params[key] = swap_elements_at(
                correct_component_indexes, value)

        return (alpha_k, normal_wishart_params)

    def _dirichlet_update_parameter(self, alpha_k_new_data,
                                    alpha_k_old_data=None):
        if alpha_k_old_data is None:
            alpha_k_old_data_delta = np.zeros_like(alpha_k_new_data)
        else:
            alpha_k_old_data_delta = alpha_k_old_data - self.alpha_0_k

        logger.debug('alpha_k_old_data_delta:\n%s',
                     np.array_str(alpha_k_old_data_delta))

        alpha_k_new_data_delta = alpha_k_new_data - self.alpha_0_k

        self.alpha_k = (self.alpha_k
                        - alpha_k_old_data_delta
                        + alpha_k_new_data_delta)

    def _normal_wishart_update_parameter(self, normal_wishart_params_new_data,
                                         normal_wishart_params_old_data=None):
        if normal_wishart_params_old_data is None:
            normal_wishart_params_old_data = {
                k: np.zeros_like(v)
                for k, v in normal_wishart_params_new_data.items()}
        self.beta_k = self._compute_update_beta_k(
            normal_wishart_params_old_data['beta_k'],
            normal_wishart_params_new_data['beta_k'])
        logger.debug('old m_k:\n%s', np.array_str(self.m_k))
        self.m_k = self._compute_update_m_k(
            normal_wishart_params_old_data,
            normal_wishart_params_new_data)
        logger.debug('new m_k:\n%s', np.array_str(self.m_k))
        self._beta_k_m_k = self._compute_beta_k_m_k(self.m_k, self.beta_k)
        self._inv_W_k = self._compute_update_inv_W_k(
            normal_wishart_params_old_data,
            normal_wishart_params_new_data)
        self.W_k = np.linalg.inv(self._inv_W_k)
        self._inv_W_k_expression = \
            self._compute_inv_W_k_expression({'W_k': self.W_k,
                                              'm_k': self.m_k,
                                              'beta_k': self.beta_k})
        self.nu_k = self._compute_update_nu_k(
            normal_wishart_params_old_data['nu_k'],
            normal_wishart_params_new_data['nu_k'])

    def _compute_update_m_k(self,
                            normal_wishart_params_old_data,
                            normal_wishart_params_new_data):
        beta_k_m_k_old_data = self._compute_beta_k_m_k(
            normal_wishart_params_old_data['m_k'],
            normal_wishart_params_old_data['beta_k'])
        if np.any(beta_k_m_k_old_data):
            beta_k_m_k_old_data_delta = (beta_k_m_k_old_data
                                         - self._beta_0_k_m_0_k)
        else:
            beta_k_m_k_old_data_delta = np.zeros_like(beta_k_m_k_old_data)

        logger.debug('beta_k_m_k_old_data_delta:\n%s',
                     np.array_str(beta_k_m_k_old_data_delta))

        beta_k_m_k_new_data = self._compute_beta_k_m_k(
            normal_wishart_params_new_data['m_k'],
            normal_wishart_params_new_data['beta_k'])
        beta_k_m_k_new_data_delta = beta_k_m_k_new_data - self._beta_0_k_m_0_k

        assert beta_k_m_k_new_data.shape == (
            self.n_components, self.n_features)
        assert beta_k_m_k_old_data.shape == (
            self.n_components, self.n_features)
        return (1. / self.beta_k[:, np.newaxis]
                * (self._beta_k_m_k - beta_k_m_k_old_data_delta
                   + beta_k_m_k_new_data_delta))

    def _compute_update_beta_k(self, beta_k_old_data, beta_k_new_data):
        if np.any(beta_k_old_data):
            beta_k_old_data_delta = beta_k_old_data - self.beta_0_k
        else:
            beta_k_old_data_delta = np.zeros_like(beta_k_old_data)
        logger.debug('beta_k_old_data_delta:\n%s',
                     np.array_str(beta_k_old_data))

        beta_k_new_data_delta = beta_k_new_data - self.beta_0_k
        return self.beta_k - beta_k_old_data_delta + beta_k_new_data_delta

    def _compute_update_inv_W_k(self,
                                normal_wishart_params_old_data,
                                normal_wishart_params_new_data):
        inv_W_k_expression_old_data = self._compute_inv_W_k_expression(
            normal_wishart_params_old_data)
        inv_W_k_expression_new_data = self._compute_inv_W_k_expression(
            normal_wishart_params_new_data)
        beta_k_m_k_m_k_T = self._compute_beta_k_m_k_m_k_T(self.m_k,
                                                          self.beta_k)

        if np.any(inv_W_k_expression_old_data):
            inv_W_k_expression_old_data_delta = \
                (inv_W_k_expression_old_data - self._inv_W_0_k_expression)
        else:
            inv_W_k_expression_old_data_delta = np.zeros_like(
                inv_W_k_expression_old_data)

        logger.debug('inv_W_k_expression_old_data_delta:\n%s',
                     np.array_str(inv_W_k_expression_old_data_delta))

        inv_W_k_expression_new_data_delta = (inv_W_k_expression_new_data
                                             - self._inv_W_0_k_expression)
        return (self._inv_W_k_expression
                - inv_W_k_expression_old_data_delta
                + inv_W_k_expression_new_data_delta
                - beta_k_m_k_m_k_T)

    def _compute_inv_W_k_expression(self, normal_wishart_params):
        W_k = normal_wishart_params['W_k']
        if np.any(W_k):
            inv_W_k = np.linalg.inv(W_k)
        else:
            inv_W_k = np.zeros_like(W_k)

        beta_k = normal_wishart_params['beta_k']
        m_k = normal_wishart_params['m_k']
        beta_k_m_k_m_k_T = self._compute_beta_k_m_k_m_k_T(m_k, beta_k)

        return inv_W_k + beta_k_m_k_m_k_T

    def _compute_beta_k_m_k_m_k_T(self, m_k, beta_k):
        m_k_T = np.reshape(m_k, (self.n_components, 1, self.n_features))
        m_k_new_axis = np.reshape(m_k, (self.n_components, self.n_features, 1))
        m_k_m_k_T = np.einsum('lij, ljk -> lik', m_k_new_axis, m_k_T)
        assert m_k_m_k_T.shape == (self.n_components, self.n_features,
                                   self.n_features)

        beta_k_m_k_m_k_T = beta_k[:, np.newaxis, np.newaxis] * m_k_m_k_T
        assert beta_k_m_k_m_k_T.shape == (self.n_components, self.n_features,
                                          self.n_features)
        return beta_k_m_k_m_k_T

    def _compute_beta_k_m_k(self, m_k, beta_k):
        assert beta_k.shape == (self.n_components,)
        assert m_k.shape == (self.n_components, self.n_features)
        return beta_k[:, np.newaxis] * m_k

    def _compute_update_nu_k(self, nu_k_old_data, nu_k_new_data):
        if np.any(nu_k_old_data):
            nu_k_old_data_delta = nu_k_old_data - self.nu_0_k
        else:
            nu_k_old_data_delta = np.zeros_like(nu_k_old_data)

        logger.debug('nu_k_old_data_delta:\n%s',
                     np.array_str(nu_k_old_data_delta))

        nu_k_new_data_delta = nu_k_new_data - self.nu_0_k
        return self.nu_k - nu_k_old_data_delta + nu_k_new_data_delta
