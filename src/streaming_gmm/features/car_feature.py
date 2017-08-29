import numpy as np


class CARFeature(object):

    def _CAR_likelihood(self, time, mag, error, params):
        sigma = params['sigma']
        tau = params['tau']

        N = np.shape(mag)

        mean_value = np.mean(mag)
        omega_0 = tau * (sigma ** 2) / 2.
        omega_last = omega_0
        x_hat_last = 0.
        a_last = 0.
        x_ast_last = mag[0] - mean_value

        likelihood = 0.

        for i in range(1, N):
            a_i = np.exp(-(time[i] - time[i - 1]) / tau)
            x_ast_i = mag[i] - mean_value
            x_hat_i = a_i * x_hat_last + (a_i * omega_last) * (x_ast_last + x_hat_last) / (omega_last + error[i - 1] ** 2)
            omega_i = omega_0 * (1 - a_i ** 2) + (a_i ** 2) * omega_last * (1 - omega_last / (omega_last + error[i - 1] ** 2))

            quocient = omega_i + error[i] ** 2
            likelihood += -1. / 2. * np.log(2 * np.pi * quocient) - 1. / 2. * (x_hat_i + x_ast_i ** 2) / quocient

        return -likelihood
