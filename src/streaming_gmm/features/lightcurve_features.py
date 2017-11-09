import numpy as np
import sympy as sy
import math
from scipy.optimize import minimize
from .streaming_aov import StreamingAOV


class StreamingFeature:

    def value(self):
        raise NotImplementedError("Child classes need to implement this"
                                  "method")

    def update(self, observations, observations_other_band={}):
        raise NotImplementedError("Child classes need to implement this"
                                  "method")

    @staticmethod
    def _get_aligned_obs(observations):
        return {'time': observations['aligned_time'],
                'magnitude': observations['aligned_magnitude'],
                'error': observations['aligned_error']}

    @staticmethod
    def _unpack_observations(observations):
        time = observations['time']
        magnitude = observations['magnitude']
        error = observations['error']
        return time, magnitude, error

    @staticmethod
    def _check_arrays_size(time, magnitude, error):
        same_shape = (time.shape[0] == magnitude.shape[0]
                      and time.shape[0] == error.shape[0])
        if not same_shape:
            error_message = ("time, magnitude and new_error must have same "
                             "number of samples: {}, {} and {} "
                             "respectively".format(
                                 time.shape[0], magnitude.shape[0],
                                 error.shape[0]))
            raise ValueError(error_message)


class MeanMagnitude(StreamingFeature):

    def __init__(self):
        self.acumulated_samples = 0
        self.acumulated_sum = 0.

    def __str__(self):
        return "MeanMagnitude"

    def value(self):
        return self.acumulated_sum / self.acumulated_samples

    def update(self, observations, observations_other_band={}):
        time, magnitude, error = self._unpack_observations(observations)
        self._check_arrays_size(time, magnitude, error)
        self.acumulated_samples += magnitude.shape[0]
        self.acumulated_sum += np.sum(magnitude)


class MeanVariance(StreamingFeature):

    def __init__(self):
        self.mean_magnitude = MeanMagnitude()
        self.std = Std()

    def __str__(self):
        return "MeanVariance"

    def value(self):
        return self.std.value() / self.mean_magnitude.value()

    def update(self, observations, observations_other_band={}):
        self.mean_magnitude.update(observations)
        self.std.update(observations)


class Color(StreamingFeature):

    def __init__(self):
        self._mean_magnitude_1 = MeanMagnitude()
        self._mean_magnitude_2 = MeanMagnitude()

    def __str__(self):
        return "Color"

    def value(self):
        return (self._mean_magnitude_1.value()
                - self._mean_magnitude_2.value())

    def update(self, observations, observations_other_band):
        self._mean_magnitude_1.update(observations)
        self._mean_magnitude_2.update(observations_other_band)


class CARFeature(StreamingFeature):

    def __init__(self):
        self.sigma = 10
        self.tau = 0.5
        self.b = 0
        self.mean_magnitude = MeanMagnitude()

        self._x_hat_last = 0
        self._x_ast_last = 0
        self._omega_last = 0

    def __str__(self):
        return "CAR tau"

    def value(self):
        return self.tau

    def update(self, observations, observations_other_band={}):
        time, magnitude, error = self._unpack_observations(observations)
        self.mean_magnitude.update(observations)
        self._check_arrays_size(time, magnitude, error)
        self._maximize_likelihood(time, magnitude, error)

    def _maximize_likelihood(self, time, magnitude, error):
        x0 = [self.sigma, self.tau]
        bounds = ((0, 100), (0, 100))
        res = minimize(self._CAR_likelihood, x0, args=(time, magnitude, error),
                       method='nelder-mead', bounds=bounds)

        self.sigma = res.x[0]
        self.tau = res.x[1]
        if self.sigma < 0 or self.sigma > 100:
            self.sigma = 10

        if self.tau < 0 or self.tau > 100:
            self.tau = 0.5

    def _CAR_likelihood(self, params, time, mag, error):
        sigma = params[0]
        tau = params[1]

        N = mag.shape[0]

        mean_value = self.mean_magnitude.value()
        omega_0 = tau * (sigma ** 2) / 2.
        self._omega_last = omega_0
        self._x_hat_last = 0.
        self.x_ast_last = mag[0] - mean_value

        likelihood = 0.

        for i in range(1, N):
            a_i = np.exp(-(time[i] - time[i - 1]) / tau)
            x_ast_i = mag[i] - mean_value

            quocient_last = self._omega_last + error[i - 1] ** 2
            x_hat_i = a_i * self._x_hat_last + a_i * self._omega_last * (self.x_ast_last + self._x_hat_last) / quocient_last
            omega_i = omega_0 * (1 - a_i ** 2) + (a_i ** 2) * self._omega_last * (1 - self._omega_last / quocient_last)

            quocient = omega_i + error[i] ** 2
            likelihood += -1. / 2. * (np.log(2 * np.pi) + np.log(quocient)) - 1. / 2. * (x_hat_i + x_ast_i ** 2) / quocient

            self._omega_last = omega_i
            self.x_ast_last = x_ast_i
            self._x_hat_last = x_hat_i

        return -likelihood


class Period(StreamingFeature):

    def __init__(self):
        self._streaming_aov = StreamingAOV(plow=.01, phigh=5.0, step=1e-3)

    def __str__(self):
        return "Period"

    def value(self):
        return self._streaming_aov.get_period()

    def update(self, observations, observations_other_band={}):
        time, magnitude, error = self._unpack_observations(observations)
        self._check_arrays_size(time, magnitude, error)
        self._streaming_aov.update(time, magnitude, error)


class _WeightedMean(StreamingFeature):

    def __init__(self):
        self.mean_mag_num = 0
        self.mean_mag_den = 0

    def value(self):
        if self.mean_mag_den == 0:
            return 0
        return self.mean_mag_num / self.mean_mag_den

    def update(self, observations, observations_other_band={}):
        time, magnitude, error = self._unpack_observations(observations)
        self.mean_mag_num += np.sum(magnitude / (error ** 2))
        self.mean_mag_den += np.sum(1. / (error ** 2))


class StetsonK(StreamingFeature):

    def __init__(self):
        self.weighted_mean = _WeightedMean()
        self.sigmap_num = 0
        self.sigmap_den = 0
        self.acumulated_samples = 0
        self.K = 0
        self._prev_mean_mag = 0
        self._next_sigmap_num_correction = 0
        self._next_sigmap_den_correction_1 = 0
        self._next_sigmap_den_correction_2 = 0

    def __str__(self):
        return "StetsonK"

    def value(self):
        return self.K

    def update(self, observations, observations_other_band={}):
        time, magnitude, error = self._unpack_observations(observations)

        old_mean_mag = self.weighted_mean.value()

        self.weighted_mean.update(observations, observations_other_band)
        mean_mag = self.weighted_mean.value()

        self.acumulated_samples += magnitude.shape[0]

        # TODO: approximate a new sum
        self.sigmap_num += (old_mean_mag - mean_mag) * self._next_sigmap_num_correction
        self.sigmap_num += np.sum(np.abs((magnitude - mean_mag) / error))

        self.sigmap_den += (old_mean_mag - mean_mag) * self._next_sigmap_den_correction_1
        self.sigmap_den += .5 * (old_mean_mag - mean_mag) ** 2 * self._next_sigmap_den_correction_2
        self.sigmap_den += np.sum(((magnitude - mean_mag) / error) ** 2)

        sqrt_acum_samples = np.sqrt(
            self.acumulated_samples / (self.acumulated_samples - 1))

        self.K = (1 / np.sqrt(self.acumulated_samples) * sqrt_acum_samples * self.sigmap_num / (np.sqrt(sqrt_acum_samples * self.sigmap_den)))

        self._next_sigmap_num_correction = np.sum(-np.sign(magnitude - mean_mag) / (error * np.sign(error)))
        self._next_sigmap_den_correction_1 = 2 * np.sum((-magnitude + mean_mag) / error ** 2)
        self._next_sigmap_den_correction_2 = 2 * np.sum(error ** 2)


class StetsonJ(StreamingFeature):

    def __init__(self):
        self.weighted_mean_1 = _WeightedMean()
        self.weighted_mean_2 = _WeightedMean()
        self.acumulated_samples = 0
        self._acum_sum = 0

    def __str__(self):
        return "StetsonJ"

    def value(self):
        return 1. / self.acumulated_samples * self._acum_sum

    def update(self, observations, observations_other_band={}):
        aligned_obs = self._get_aligned_obs(observations)
        aligned_obs_other_band = self._get_aligned_obs(
            observations_other_band)
        time1, magnitude1, error1 = self._unpack_observations(aligned_obs)
        time2, magnitude2, error2 = self._unpack_observations(
            aligned_obs_other_band)
        self.acumulated_samples += magnitude1.shape[0]
        self.weighted_mean_1.update(observations)
        self.weighted_mean_2.update(observations_other_band)
        mean1 = np.mean(magnitude1)
        mean2 = np.mean(magnitude2)

        samples_term = np.sqrt(
            self.acumulated_samples / (self.acumulated_samples - 1))
        sigmap = samples_term * (magnitude1 - self.weighted_mean_1.value()) / error1
        sigmaq = samples_term * (magnitude2 - self.weighted_mean_2.value()) / error2
        sigmai = sigmap * sigmaq

        self._acum_sum += np.sum(np.sign(sigmai) * np.sqrt(np.abs(sigmai)))


class StetsonL(StreamingFeature):

    def __init__(self):
        self.stetson_k = StetsonK()
        self.stetson_j = StetsonJ()

    def __str__(self):
        return "StetsonL"

    def value(self):
        k = self.stetson_k.value()
        j = self.stetson_j.value()
        return j * k / .798

    def update(self, observations, observations_other_band={}):
        aligned_obs = self._get_aligned_obs(observations)
        aligned_obs_other_band = self._get_aligned_obs(
            observations_other_band)
        self.stetson_k.update(aligned_obs, aligned_obs_other_band)

        # We don't pass the aligned observations, because stetson_j receives
        # the whole observations dict
        self.stetson_j.update(observations, observations_other_band)


class Std(StreamingFeature):

    def __init__(self):
        self.cumulative_sum = 0
        self.acumulated_samples = 0
        self.mean_magnitude = MeanMagnitude()

    def __str__(self):
        return "Std"

    def value(self):
        return np.sqrt(1 / (self.acumulated_samples - 1) * self.cumulative_sum)

    def update(self, observations, observations_other_band={}):
        time, magnitude, error = self._unpack_observations(observations)
        self.acumulated_samples += magnitude.shape[0]
        self.mean_magnitude.update(observations, observations_other_band)
        self.cumulative_sum += \
            np.sum((magnitude - self.mean_magnitude.value()) ** 2)


class RangeCS(StreamingFeature):

    def __init__(self):
        self.cumulative_sum_l = []
        self.std = Std()
        self.mean_magnitude = MeanMagnitude()
        self._max = -1 * (1 << 30)  # very small number
        self._min = -self._max  # very big number
        self.acumulated_samples = 0

    def __str__(self):
        return "RangeCS"

    def value(self):
        return 1. / (self.acumulated_samples * self.std.value()) * (self._max - self._min)

    def update(self, observations, observations_other_band={}):
        time, magnitude, error = self._unpack_observations(observations)
        N = magnitude.shape[0]

        self.acumulated_samples += N
        self.std.update(observations)
        self.mean_magnitude.update(observations)

        for i in range(N):
            if len(self.cumulative_sum_l) > 0:
                prev_sum = self.cumulative_sum_l[-1]
            else:
                prev_sum = 0
            self.cumulative_sum_l.append(prev_sum + magnitude[i])

        sorted_cum_sum = sorted(self.cumulative_sum_l)
        sorted_index = np.argsort(self.cumulative_sum_l)

        # for debugging
        max_min = []
        for i in range(len(self.cumulative_sum_l)):
            max_min.append(self.cumulative_sum_l[i] - (i + 1) * self.mean_magnitude.value())

        print("Max true: ", max(max_min))
        print("Min true: ", min(max_min))
        print("acum samples: ", self.acumulated_samples)
        print("std ", self.std.value())

        # self._max = max(max_min)
        # self._min = min(max_min)
        self._max = self.binary_search_max(sorted_cum_sum, sorted_index)
        self._min = self.binary_search_min(sorted_cum_sum, sorted_index)
        # print("Max false: ", self._max)
        # print("Min false: ", self._min)

    def binary_search_min(self, sorted_cum_sum, sorted_index):
        first = 0
        last = len(sorted_cum_sum)

        while first < last:
            mid = int((first + last) / 2)
            first_val = sorted_cum_sum[first] - (sorted_index[first] + 1) * self.mean_magnitude.value()
            mid_val = sorted_cum_sum[mid] - (sorted_index[mid] + 1) * self.mean_magnitude.value()
            if first_val <= mid_val:
                last = mid
            else:
                first = mid
        return sorted_cum_sum[first] - (sorted_index[first] + 1) * self.mean_magnitude.value()

    def binary_search_max(self, sorted_cum_sum, sorted_index):
        first = 0
        last = len(sorted_cum_sum)

        while first < last:
            mid = int((first + last) / 2)
            first_val = sorted_cum_sum[first] - (sorted_index[first] + 1) * self.mean_magnitude.value()
            mid_val = sorted_cum_sum[mid] - (sorted_index[mid] + 1) * self.mean_magnitude.value()
            if first_val >= mid_val:
                last = mid
            else:
                first = mid
        return sorted_cum_sum[first] - (sorted_index[first] + 1) * self.mean_magnitude.value()

class Skewness(StreamingFeature):

    def __init__(self):
        self.cum_sum = 0
        self.acumulated_samples = 0

    def value(self):
        pass

    def update(self, observations, observations_other_band={}):
        pass


class _BaseFluxPercentileRatio(StreamingFeature):

    def __init__(self):
        self.acumulated_samples = 0
        self.cached_mags = []

    def update(self, observations, observations_other_band={}):
        time, magnitude, error = self._unpack_observations(observations)
        self.cached_mags.append(magnitude)
        self.acumulated_samples += magnitude.shape[0]

        # TODO: do this online. Sort the new array and merge with the other
        # one that is sorted
        all_mags = np.concatenate(self.cached_mags).ravel()
        sorted_data = np.sort(all_mags)
        self._calculate_percentile(sorted_data)

    def _calculate_percentile(self, sorted_data):
        raise NotImplementedError()


class FluxPercentileRatioMid50(_BaseFluxPercentileRatio):

    def __init__(self):
        super(FluxPercentileRatioMid50, self).__init__()
        self.F_mid50 = 0

    def __str__(self):
        return "FluxPercentileRatioMid50"

    def value(self):
        return self.F_mid50

    def _calculate_percentile(self, sorted_data):
        F_25_index = int(math.ceil(0.25 * self.acumulated_samples))
        F_75_index = int(math.ceil(0.75 * self.acumulated_samples))
        F_5_index = int(math.ceil(0.05 * self.acumulated_samples))
        F_95_index = int(math.ceil(0.95 * self.acumulated_samples))

        F_25_75 = sorted_data[F_75_index] - sorted_data[F_25_index]
        F_5_95 = sorted_data[F_95_index] - sorted_data[F_5_index]
        self.F_mid50 = F_25_75 / F_5_95


class FluxPercentileRatioMid65(_BaseFluxPercentileRatio):

    def __init__(self):
        super(FluxPercentileRatioMid65, self).__init__()
        self.F_mid65 = 0

    def __str__(self):
        return "FluxPercentileRatioMid65"

    def value(self):
        return self.F_mid65

    def _calculate_percentile(self, sorted_data):
        F_175_index = int(math.ceil(0.175 * self.acumulated_samples))
        F_825_index = int(math.ceil(0.825 * self.acumulated_samples))
        F_5_index = int(math.ceil(0.05 * self.acumulated_samples))
        F_95_index = int(math.ceil(0.95 * self.acumulated_samples))

        F_175_825 = sorted_data[F_825_index] - sorted_data[F_175_index]
        F_5_95 = sorted_data[F_95_index] - sorted_data[F_5_index]
        self.F_mid65 = F_175_825 / F_5_95


class FluxPercentileRatioMid80(_BaseFluxPercentileRatio):

    def __init__(self):
        super(FluxPercentileRatioMid80, self).__init__()
        self.F_mid80 = 0

    def __str__(self):
        return "FluxPercentileRatioMid80"

    def value(self):
        return self.F_mid80

    def _calculate_percentile(self, sorted_data):
        F_10_index = int(math.ceil(0.10 * self.acumulated_samples))
        F_90_index = int(math.ceil(0.90 * self.acumulated_samples))
        F_5_index = int(math.ceil(0.05 * self.acumulated_samples))
        F_95_index = int(math.ceil(0.95 * self.acumulated_samples))

        F_10_90 = sorted_data[F_90_index] - sorted_data[F_10_index]
        F_5_95 = sorted_data[F_95_index] - sorted_data[F_5_index]
        self.F_mid80 = F_10_90 / F_5_95


class FluxPercentileRatioMid35(_BaseFluxPercentileRatio):

    def __init__(self):
        super(FluxPercentileRatioMid35, self).__init__()
        self.F_mid35 = 0

    def __str__(self):
        return "FluxPercentileRatioMid35"

    def value(self):
        return self.F_mid35

    def _calculate_percentile(self, sorted_data):
        F_325_index = int(math.ceil(0.325 * self.acumulated_samples))
        F_675_index = int(math.ceil(0.675 * self.acumulated_samples))
        F_5_index = int(math.ceil(0.05 * self.acumulated_samples))
        F_95_index = int(math.ceil(0.95 * self.acumulated_samples))

        F_325_675 = sorted_data[F_675_index] - sorted_data[F_325_index]
        F_5_95 = sorted_data[F_95_index] - sorted_data[F_5_index]
        self.F_mid35 = F_325_675 / F_5_95


class FluxPercentileRatioMid20(_BaseFluxPercentileRatio):

    def __init__(self):
        super(FluxPercentileRatioMid20, self).__init__()
        self.F_mid20 = 0

    def __str__(self):
        return "FluxPercentileRatioMid20"

    def value(self):
        return self.F_mid20

    def _calculate_percentile(self, sorted_data):
        F_60_index = int(math.ceil(0.60 * self.acumulated_samples))
        F_40_index = int(math.ceil(0.40 * self.acumulated_samples))
        F_5_index = int(math.ceil(0.05 * self.acumulated_samples))
        F_95_index = int(math.ceil(0.95 * self.acumulated_samples))

        F_40_60 = sorted_data[F_60_index] - sorted_data[F_40_index]
        F_5_95 = sorted_data[F_95_index] - sorted_data[F_5_index]
        self.F_mid20 = F_40_60 / F_5_95


class LightCurveFeatures:

    FEATURES_MAP = {
        'mean_magnitude': MeanMagnitude,
        'mean_variance': MeanVariance,
        'period': Period,
        'stetson_k': StetsonK,
        'stetson_j': StetsonJ,
        'stetson_l': StetsonL,
        'std': Std,
        'flux_percentile_ratio_mid_50': FluxPercentileRatioMid50,
        'flux_percentile_ratio_mid_65': FluxPercentileRatioMid65,
        'flux_percentile_ratio_mid_80': FluxPercentileRatioMid80,
        'flux_percentile_ratio_mid_20': FluxPercentileRatioMid20,
        'flux_percentile_ratio_mid_35': FluxPercentileRatioMid35,
        'color': Color,
        'range_cs': RangeCS,
    }

    def __init__(self, features=[]):
        if not features:
            features = list(self.FEATURES_MAP.keys())
        self.features = {key: self.FEATURES_MAP[key]() for key in features}

    def values(self):
        values_dict = {key: feature.value()
                       for key, feature in self.features.items()}
        return values_dict

    def update(self, observations, observations_other_band={}):
        for feature in self.features.values():
            feature.update(observations, observations_other_band)
