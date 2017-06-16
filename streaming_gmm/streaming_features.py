import numpy as np


def _check_arrays_size(time, magnitude, error):
    same_shape = (time.shape[0] == magnitude.shape[0]
                  and time.shape[0] == error.shape[0]
                  and magnitude.shape[0] == error.shape[0])
    if not same_shape:
        error_message = ("time, magnitude and error must have same number"
                         " of samples: {}, {} and {} respectively".format(
                             time.shape[0], magnitude.shape[0],
                             error.shape[0]))
        raise ValueError(error_message)


class StreamingFeature:

    def update(self, time, magnitude, error):
        raise NotImplementedError()


class StreamingBGLS(StreamingFeature):

    def __init__(self, plow=.5, phigh=100, ofac=.6, freq_multiplier=100):
        """
        Streaming Bayesian Generalized Lomb-Scargle periodogram.

        See: Mortier et al. 2014 "BGLS: A Bayesian formalism for the
        generalised Lomb-Scargle periodogram"

        Parameters
        ----------
        plow: float
            lowest period to consider

        phigh: float
            highest period to consider

        ofac: float
            oversampling factor
        """
        self.plow = plow
        self.phigh = phigh
        self.ofac = ofac

        self.acumulated_samples = 0

        frequency_num = .5 * self.ofac * self.phigh * freq_multiplier
        self.frequency = np.linspace(1. / self.phigh, 1. / self.plow,
                                     frequency_num)
        self.period = 1. / self.frequency

        self.omega = 2 * np.pi * self.frequency
        self.omega_time = None

        self.W = 0.
        self.Y = 0.
        self.YY_hat = 0.
        self.YC_hat = 0.
        self.YS_hat = 0.
        self.C = 0.
        self.S = 0.
        self.CC_hat = 0.
        self.SS_hat = 0.
        self._theta_up = 0.
        self._theta_down = 0.

        # Create this variables here, so we don't have to allocate memory
        # every time self.update() is called.
        self._K = np.zeros((self.frequency.shape))
        self._L = np.zeros((self.frequency.shape))
        self._M = np.zeros((self.frequency.shape))
        self._constants = np.zeros((self.frequency.shape))

    def periodogram(self):
        return self.period, self.probability

    def most_probable_period(self):
        return self.period[np.argmax(self.probability)]

    def update(self, time, magnitude, error):
        _check_arrays_size(time, magnitude, error)

        n_samples = time.shape[0]
        self.acumulated_samples += n_samples

        error_sq = error ** 2

        # eq. 8
        w_i = 1. / error_sq

        # eq. 9
        self.W += w_i.sum()

        # eq. 10
        self.Y += (w_i * magnitude).sum()

        # current_omega_time is a (frequency, time) matrix
        current_omega_time = self.omega[:, np.newaxis] * time[np.newaxis, :]
        self._theta_up += (w_i[np.newaxis, :] *
                           np.sin(2 * current_omega_time)).sum(axis=1)
        self._theta_down += (w_i[np.newaxis, :] *
                             np.cos(2 * current_omega_time)).sum(axis=1)
        theta = .5 * np.arctan2(self._theta_up, self._theta_down)
        x = current_omega_time - theta[:, np.newaxis]

        cos_x = np.cos(x)
        sin_x = np.sin(x)
        w_cos_x = w_i[np.newaxis, :] * cos_x
        w_sin_x = w_i[np.newaxis, :] * sin_x

        # eq. 14
        self.C += w_cos_x.sum(axis=1)
        # eq. 15
        self.S += w_sin_x.sum(axis=1)
        # eq. 12
        self.YC_hat += (magnitude * w_cos_x).sum(axis=1)
        # eq. 13
        self.YS_hat += (magnitude * w_sin_x).sum(axis=1)
        # eq. 16
        self.CC_hat += (w_cos_x * cos_x).sum(axis=1)
        # eq. 17
        self.SS_hat += (w_sin_x * sin_x).sum(axis=1)

        self._case_with_positives()
        self._case_with_zeros()

        exponents = self._M - (self._L ** 2) / (4. * self._K)
        log_p = np.log10(self._constants) + (exponents * np.log10(np.exp(1.)))

        # normalize to take power 10
        log_p -= log_p.max()
        prob = 10 ** log_p
        self.probability = prob / prob.sum()

    def _case_with_positives(self):
        ind = (self.CC_hat != 0) & (self.SS_hat != 0)
        tmp_var = 1. / (self.CC_hat[ind] * self.SS_hat[ind])

        self._K[ind] = (self.C[ind] ** 2 * self.SS_hat[ind] +
                        self.S[ind] ** 2 * self.CC_hat[ind] -
                        self.W * self.CC_hat[ind] * self.SS_hat[ind])
        self._K[ind] = .5 * self._K[ind] * tmp_var

        self._L[ind] = (self.Y * self.CC_hat[ind] * self.SS_hat[ind] -
                        self.C[ind] * self.YC_hat[ind] * self.SS_hat[ind] -
                        self.S[ind] * self.YS_hat[ind] * self.CC_hat[ind])
        self._L[ind] = tmp_var * self._L[ind]

        self._M[ind] = (self.YC_hat[ind] ** 2 * self.SS_hat[ind] +
                        self.YS_hat[ind] ** 2 * self.CC_hat[ind])
        self._M[ind] = .5 * tmp_var * self._M[ind]

        self._constants = np.sqrt(tmp_var[ind] / np.abs(self._K[ind]))

    def _case_with_zeros(self):
        ind = self.CC_hat == 0

        self._K[ind] = self.S[ind] ** 2 - self.W * self.SS_hat[ind]
        self._K[ind] = self._K[ind] / (2. * self.SS_hat[ind])

        self._L[ind] = (self.Y * self.CC_hat[ind] -
                        self.C[ind] * self.YC_hat[ind])
        self._L[ind] = self._L[ind] / self.CC_hat[ind]

        self._M[ind] = (self.YC_hat[ind] ** 2) / (2. * self.CC_hat[ind])

        self._constants[ind] = (1. / np.sqrt(self.CC_hat[ind] *
                                             np.abs(self._K[ind])))
