import logging

import numpy as np


logger = logging.getLogger(__name__)


def _check_arrays_size(time, magnitude, new_error):
    same_shape = (time.shape[0] == magnitude.shape[0]
                  and time.shape[0] == new_error.shape[0]
                  and magnitude.shape[0] == new_error.shape[0])
    if not same_shape:
        error_message = ("time, magnitude and new_error must have same number"
                         " of samples: {}, {} and {} respectively".format(
                             time.shape[0], magnitude.shape[0],
                             new_error.shape[0]))
        raise ValueError(error_message)


class StreamingFeature:

    def update(self, time, magnitude, new_error):
        raise NotImplementedError()


class StreamingAOV(StreamingFeature):

    def __init__(self, number_of_bins=10, plow=.01, phigh=2, step=1e-3):
        self.number_of_bins = number_of_bins
        self.periods = np.arange(plow, phigh, step=step)
        self.periods_len = self.periods.shape[0]
        self.aov_per_period = np.zeros_like(self.periods)
        self.acumulated_samples = 0

        self._magnitude_average = 0.
        self._inv_error_sq_sum = 0.
        self._magnitude_sq_sum = 0.
        self._magnitude_sum = 0.
        self._bins = np.linspace(0., 1., num=self.number_of_bins + 1)

        period_bin_map_size = (self.periods_len, self.number_of_bins)
        self._bin_average_map = np.zeros(period_bin_map_size)
        self._bin_magnitude_sq_sum_map = np.zeros(period_bin_map_size)
        self._bin_magnitude_sum_map = np.zeros(period_bin_map_size)
        self._bin_inv_error_sq_sum_map = np.zeros(period_bin_map_size)
        self._bin_acumulated_samples = np.zeros(period_bin_map_size)

    def get_period(self):
        max_aov_idx = np.argmax(self.aov_per_period)
        return self.periods[max_aov_idx]

    def update(self, new_time, new_magnitude, new_error):
        _check_arrays_size(new_time, new_magnitude, new_error)

        new_samples = new_time.shape[0]
        self.acumulated_samples += new_samples
        self._update_magnitude_avg(new_magnitude, new_error)

        for period_idx in range(self.periods_len):
            self.aov_per_period[period_idx] = self._compute_aov_of_period(
                new_time,
                new_magnitude,
                new_error,
                period_idx)

    def _compute_aov_of_period(self, new_time, new_magnitude, new_error,
                               period_idx):
        period = self.periods[period_idx]
        phi = np.mod(new_time, period) / period
        s1 = 0.
        s2 = 0.
        for bin_ in range(self.number_of_bins):
            idx_bin = np.where(np.logical_and(phi >= self._bins[bin_],
                                              phi < self._bins[bin_ + 1]))[0]
            if len(idx_bin) == 0.0:
                continue

            old_error_sum = self._bin_inv_error_sq_sum_map[period_idx][bin_]

            self._bin_acumulated_samples[period_idx][bin_] += \
                new_magnitude[idx_bin].shape[0]
            self._bin_magnitude_sq_sum_map[period_idx][bin_] += np.sum(
                new_magnitude[idx_bin] ** 2)
            self._bin_magnitude_sum_map[period_idx][bin_] += np.sum(
                new_magnitude[idx_bin])
            self._bin_inv_error_sq_sum_map[period_idx][bin_] += np.sum(
                1. / new_error[idx_bin] ** 2)

            period_bin = (period_idx, bin_)
            new_bin_average = self._compute_bin_average(old_error_sum,
                                                        period_bin,
                                                        new_magnitude[idx_bin],
                                                        new_error[idx_bin])
            s1 += (self._bin_acumulated_samples[period_idx][bin_]
                   * (new_bin_average - self._magnitude_average) ** 2)
            s2 += self._compute_s2(period_bin, new_bin_average)
            self._bin_average_map[period_idx][bin_] = new_bin_average
        s1 = s1 / (self.number_of_bins - 1)
        s2 = s2 / (self.acumulated_samples - self.number_of_bins)
        return s1 / s2

    def _update_magnitude_avg(self, new_magnitude, new_error):
        old_magnitude_weighted_sum = (self._inv_error_sq_sum
                                      * self._magnitude_average)

        new_sigma = 1 / (new_error ** 2)
        new_magnitude_weighted_sum = np.sum(new_magnitude * new_sigma)
        self._inv_error_sq_sum += np.sum(new_sigma)

        new_magnitude_average = (old_magnitude_weighted_sum
                                 + new_magnitude_weighted_sum)
        new_magnitude_average /= self._inv_error_sq_sum
        self._magnitude_average = new_magnitude_average

    def _compute_s2(self, period_bin, bin_average):
        period_idx = period_bin[0]
        bin_ = period_bin[1]
        bin_average_sq = bin_average ** 2
        sum_mag_sq = self._bin_magnitude_sq_sum_map[period_idx][bin_]
        sum_mag = self._bin_magnitude_sum_map[period_idx][bin_]
        N = self._bin_acumulated_samples[period_idx][bin_]
        return (sum_mag_sq - 2 * bin_average * sum_mag + N * bin_average_sq)

    def _compute_bin_average(self, old_error_sum,
                             period_bin,
                             new_magnitude,
                             new_error):
        period_idx = period_bin[0]
        bin_ = period_bin[1]
        old_bin_weighted_sum = (old_error_sum
                                * self._bin_average_map[period_idx][bin_])
        new_bin_weighted_sum = np.sum(new_magnitude * 1. / (new_error ** 2))

        new_bin_average = old_bin_weighted_sum + new_bin_weighted_sum
        new_bin_average /= self._bin_inv_error_sq_sum_map[period_idx][bin_]

        return new_bin_average


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
        self.number_of_updates = 0

        frequency_num = .5 * self.ofac * self.phigh * freq_multiplier
        self.frequency = np.linspace(1. / self.phigh, 1. / self.plow,
                                     frequency_num)
        self.period = 1. / self.frequency

        self.w_i = None
        self.magnitude = None
        self.omega = 2 * np.pi * self.frequency

        self.W = 0.
        self.Y = 0.
        self.YY_hat = 0.
        self.YC_hat = 0.
        self.YS_hat = 0.
        self.C = 0.
        self.S = 0.
        self.CC_hat = 0.
        self.SS_hat = 0.
        self._theta_up = np.zeros((self.frequency.shape))
        self._theta_down = np.zeros((self.frequency.shape))

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

    def is_first_update(self):
        return self.number_of_updates == 0

    def update(self, new_time, new_magnitude, new_error):
        _check_arrays_size(new_time, new_magnitude, new_error)

        n_samples = new_time.shape[0]
        self.acumulated_samples += n_samples
        logger.info('Received %d new samples', n_samples)

        # eq. 8
        new_w_i = 1. / (new_error ** 2)

        # eq. 9
        self.W += new_w_i.sum()

        # eq. 10
        self.Y += (new_w_i * new_magnitude).sum()

        theta = self._compute_theta(new_time, new_w_i)

        self._update_w_i(new_w_i)
        self._update_magnitude(new_magnitude)
        x = self.omega_time - theta[:, np.newaxis]

        cos_x = np.cos(x)
        sin_x = np.sin(x)
        w_cos_x = self.w_i[np.newaxis, :] * cos_x
        w_sin_x = self.w_i[np.newaxis, :] * sin_x

        # eq. 14
        self.C = w_cos_x.sum(axis=1)
        # eq. 15
        self.S = w_sin_x.sum(axis=1)
        # eq. 12
        self.YC_hat = (self.magnitude * w_cos_x).sum(axis=1)
        # eq. 13
        self.YS_hat = (self.magnitude * w_sin_x).sum(axis=1)
        # eq. 16
        self.CC_hat = (w_cos_x * cos_x).sum(axis=1)
        # eq. 17
        self.SS_hat = (w_sin_x * sin_x).sum(axis=1)

        self._case_with_positives()
        self._case_with_zeros()

        exponents = self._M - (self._L ** 2) / (4. * self._K)
        log_p = np.log10(self._constants) + (exponents * np.log10(np.exp(1.)))

        # normalize to take power 10
        log_p -= log_p.max()
        prob = 10 ** log_p
        self.probability = prob / prob.sum()

        self._reset_tmp_arrays()
        self.number_of_updates += 1

    def _update_w_i(self, new_w_i):
        if self.is_first_update():
            self.w_i = new_w_i
        else:
            self.w_i = np.concatenate((self.w_i, new_w_i))

    def _update_omega_time(self, new_omega_time):
        if self.is_first_update():
            self.omega_time = new_omega_time
        else:
            self.omega_time = np.concatenate(
                (self.omega_time, new_omega_time), axis=1)

    def _update_magnitude(self, new_magnitude):
        if self.is_first_update():
            self.magnitude = new_magnitude
        else:
            self.magnitude = np.concatenate((self.magnitude, new_magnitude))

    def _compute_theta(self, time, new_w_i):
        # new_omega_time is a (frequency, time) matrix
        new_omega_time = self.omega[:, np.newaxis] * time[np.newaxis, :]
        self._update_omega_time(new_omega_time)
        self._theta_up += (new_w_i[np.newaxis, :] *
                           np.sin(2 * new_omega_time)).sum(axis=1)
        self._theta_down += (new_w_i[np.newaxis, :] *
                             np.cos(2 * new_omega_time)).sum(axis=1)
        theta = .5 * np.arctan2(self._theta_up, self._theta_down)
        return theta

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

    def _reset_tmp_arrays(self):
        logger.debug('Resetting tmp arrays to zero')
        self._K.fill(0)
        self._L.fill(0)
        self._M.fill(0)
        self._constants.fill(0)
