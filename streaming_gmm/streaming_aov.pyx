#cython: boundscheck=False, wraparound=False


import numpy as np
cimport numpy as np


cdef extern from "math.h":
    double fmod(double, double)


cdef double sum_of_squares_arr(double[::1] arr):
    cdef double acumulated_sum_of_sq = 0
    for i in range(arr.shape[0]):
        acumulated_sum_of_sq += arr[i] * arr[i]
    return acumulated_sum_of_sq


cdef double sum_arr(double[::1] arr):
    cdef double acumulated_sum = 0
    for i in range(arr.shape[0]):
        acumulated_sum += arr[i]
    return acumulated_sum


cdef double sum_of_inv_squares_arr(double[::1] arr):
    cdef double acumulated_sum_of_inv_sq = 0
    cdef double tmp
    for i in range(arr.shape[0]):
        tmp = arr[i] * arr[i]
        if tmp != 0:
            acumulated_sum_of_inv_sq += (1.0 / tmp)
    return acumulated_sum_of_inv_sq


cdef double sum_of_arr1_times_inv_squares_arr2(double[::1] arr1,
                                               double[::1] arr2):
    if (arr1.shape[0] != arr2.shape[0]):
        raise ValueError("Arrays must have same size")
    cdef double acumulated_sum = 0
    cdef double square_i_arr2
    for i in range(arr1.shape[0]):
        square_i_arr2 = arr2[i] * arr2[i]
        if square_i_arr2 != 0:
            acumulated_sum += arr1[i] * 1.0 / square_i_arr2
    return acumulated_sum


cdef void phase_of_arr(double[::1] arr, double q, double[::1] out):
    for i in range(arr.shape[0]):
        out[i] = fmod(arr[i], q) / q


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


cdef class StreamingAOV:

    cdef int number_of_bins
    cdef int periods_len
    cdef int acumulated_samples
    cdef double[::1] periods
    cdef double[::1] aov_per_period

    cdef double _magnitude_average
    cdef double _inv_error_sq_sum
    cdef double _magnitude_sq_sum
    cdef double _magnitude_sum

    cdef double[::1] _bins
    cdef double[:,::1] _bin_average_map
    cdef double[:,::1] _bin_magnitude_sq_sum_map
    cdef double[:,::1] _bin_magnitude_sum_map
    cdef double[:,::1] _bin_inv_error_sq_sum_map
    cdef int[:,::1] _bin_acumulated_samples

    cdef double[::1] _phi
    cdef double[::1] _bin_mag
    cdef double[::1] _bin_error
    cdef int _bin_mag_samples
    cdef int _bin_error_samples

    def __init__(self, int number_of_bins=10, double plow=.01, double phigh=2,
                 double step=1e-3):
        self.number_of_bins = number_of_bins
        self.periods = np.arange(plow, phigh, step=step)
        self.periods_len = self.periods.shape[0]
        self.aov_per_period = np.zeros_like(self.periods)
        self.acumulated_samples = 0

        self._magnitude_average = 0
        self._inv_error_sq_sum = 0
        self._magnitude_sq_sum = 0
        self._magnitude_sum = 0
        self._bins = np.linspace(0, 1, num=self.number_of_bins + 1)

        period_bin_map_size = (self.periods_len, self.number_of_bins)
        self._bin_average_map = np.zeros(period_bin_map_size)
        self._bin_magnitude_sq_sum_map = np.zeros(period_bin_map_size)
        self._bin_magnitude_sum_map = np.zeros(period_bin_map_size)
        self._bin_inv_error_sq_sum_map = np.zeros(period_bin_map_size)
        self._bin_acumulated_samples = np.zeros(period_bin_map_size,
                                                dtype=np.int32)

    def get_period(self):
        cdef int max_aov_idx = np.argmax(np.asarray(self.aov_per_period))
        return self.periods[max_aov_idx]

    def get_periodogram(self):
        # We copy the arrays to prevent others to change things in the
        # internal array
        return (np.copy(np.asarray(self.periods)),
                np.copy(np.asarray(self.aov_per_period)))

    def update(self, double[::1] new_time,
               double[::1] new_magnitude,
               double[::1] new_error):
        _check_arrays_size(new_time, new_magnitude, new_error)

        new_samples = new_time.shape[0]
        self.acumulated_samples += new_samples
        self._update_magnitude_avg(new_magnitude, new_error)

        self._phi = np.zeros(new_time.shape[0])
        self._bin_mag = np.zeros(new_magnitude.shape[0])
        self._bin_error = np.zeros(new_error.shape[0])

        for period_idx in range(self.periods_len):
            self.aov_per_period[period_idx] = self._compute_aov_of_period(
                new_time,
                new_magnitude,
                new_error,
                period_idx)

    cdef _compute_aov_of_period(self, double[::1] new_time,
                                double[::1] new_magnitude,
                                double[::1] new_error,
                                int period_idx):
        cdef double period = self.periods[period_idx]
        cdef double s1 = 0.
        cdef double s2 = 0.

        phase_of_arr(new_time, period, self._phi)

        for bin_ in range(self.number_of_bins):
            self._select_mag_error_in_bin(bin_, new_magnitude, new_error)

            if self._bin_mag_samples == 0:
                continue

            old_error_sum = self._bin_inv_error_sq_sum_map[period_idx][bin_]

            self._bin_acumulated_samples[period_idx][bin_] += \
                self._bin_mag_samples
            self._bin_magnitude_sq_sum_map[period_idx][bin_] += \
                sum_of_squares_arr(self._bin_mag)
            self._bin_magnitude_sum_map[period_idx][bin_] += \
                sum_arr(self._bin_mag)
            self._bin_inv_error_sq_sum_map[period_idx][bin_] += \
                sum_of_inv_squares_arr(self._bin_error)

            new_bin_average = self._compute_bin_average(old_error_sum,
                                                        period_idx,
                                                        bin_)
            s1 += (self._bin_acumulated_samples[period_idx][bin_]
                   * (new_bin_average - self._magnitude_average) ** 2)
            s2 += self._compute_s2(period_idx, bin_, new_bin_average)
            self._bin_average_map[period_idx][bin_] = new_bin_average
        s1 = s1 / (self.number_of_bins - 1)
        s2 = s2 / (self.acumulated_samples - self.number_of_bins)
        return s1 / s2

    cdef void _select_mag_error_in_bin(self, int bin_,
                                       double[::1] new_magnitude,
                                       double[::1] new_error):
        self._bin_mag_samples = 0
        self._bin_error_samples = 0

        for i in range(self._phi.shape[0]):
            if (self._phi[i] >= self._bins[bin_]
                    and self._phi[i] < self._bins[bin_ + 1]):
                self._bin_mag[i] = new_magnitude[i]
                self._bin_error[i] = new_error[i]
                self._bin_mag_samples += 1
                self._bin_error_samples += 1
            else:
                self._bin_mag[i] = 0
                self._bin_error[i] = 0

    cdef void _update_magnitude_avg(self, double[::1] new_magnitude,
                                    double[::1] new_error):
        old_magnitude_weighted_sum = (self._inv_error_sq_sum
                                      * self._magnitude_average)

        new_magnitude_weighted_sum = \
         sum_of_arr1_times_inv_squares_arr2(new_magnitude, new_error)
        self._inv_error_sq_sum += sum_of_inv_squares_arr(new_error)

        new_magnitude_average = (old_magnitude_weighted_sum
                                 + new_magnitude_weighted_sum)
        new_magnitude_average /= self._inv_error_sq_sum
        self._magnitude_average = new_magnitude_average

    cdef double _compute_s2(self, int period_idx,
                            int bin_,
                            double bin_average):
        bin_average_sq = bin_average ** 2
        sum_mag_sq = self._bin_magnitude_sq_sum_map[period_idx][bin_]
        sum_mag = self._bin_magnitude_sum_map[period_idx][bin_]
        N = self._bin_acumulated_samples[period_idx][bin_]
        return (sum_mag_sq - 2 * bin_average * sum_mag + N * bin_average_sq)

    cdef double _compute_bin_average(self, double old_error_sum,
                                     int period_idx,
                                     int bin_):
        old_bin_weighted_sum = (old_error_sum
                                * self._bin_average_map[period_idx][bin_])
        new_bin_weighted_sum = \
            sum_of_arr1_times_inv_squares_arr2(self._bin_mag, self._bin_error)

        new_bin_average = old_bin_weighted_sum + new_bin_weighted_sum
        new_bin_average /= self._bin_inv_error_sq_sum_map[period_idx][bin_]

        return new_bin_average
