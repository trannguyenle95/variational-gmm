import numpy as np
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

    def value(self):
        return self.acumulated_sum / self.acumulated_samples

    def update(self, observations, observations_other_band={}):
        time, magnitude, error = self._unpack_observations(observations)
        self._check_arrays_size(time, magnitude, error)
        self.acumulated_samples += magnitude.shape[0]
        self.acumulated_sum += np.sum(magnitude)


class Color(StreamingFeature):

    def __init__(self):
        self._mean_magnitude_1 = MeanMagnitude()
        self._mean_magnitude_2 = MeanMagnitude()

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

        self._x_hat_last = 0
        self._x_ast_last = 0
        self._omega_last = 0

    def update(self, observations, observations_other_band={}):
        time, magnitude, error = self._unpack_observations(observations)
        self._check_arrays_size(time, magnitude, error)
        self._maximize_likelihood(time, magnitude, error)

    def _maximize_likelihood(self, time, magnitude, error):
        x0 = [self.sigma, self.tau]
        bounds = ((0, 100), (0, 100))
        res = minimize(self._CAR_likelihood, x0, args=(time, magnitude, error),
                       method='nelder-mead', bounds=bounds)

        self.sigma = res.x[0]
        self.tau = res.x[1]

    def _CAR_likelihood(self, params, time, mag, error):
        sigma = params[0]
        tau = params[1]

        N = mag.shape[0]

        mean_value = np.mean(mag)
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
            likelihood += -1. / 2. * np.log(2 * np.pi * quocient) - 1. / 2. * (x_hat_i + x_ast_i ** 2) / quocient

            self._omega_last = omega_i
            self.x_ast_last = x_ast_i
            self._x_hat_last = x_hat_i

        return -likelihood


class Period(StreamingFeature):

    def __init__(self):
        self._streaming_aov = StreamingAOV(plow=.01, phigh=5.0, step=1e-3)

    def value(self):
        return self._streaming_aov.get_period()

    def update(self, observations, observations_other_band={}):
        time, magnitude, error = self._unpack_observations(observations)
        self._check_arrays_size(time, magnitude, error)
        self._streaming_aov.update(time, magnitude, error)


class LightCurveFeatures:

    FEATURES_MAP = {
        'mean_magnitude': MeanMagnitude,
        'period': Period,
        'color': Color,
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
