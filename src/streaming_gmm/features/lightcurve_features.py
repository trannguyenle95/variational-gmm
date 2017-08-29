import numpy as np
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
        return (self._mean_magnitude_1.mean_magnitude()
                - self._mean_magnitude_2.mean_magnitude())

    def update(self, observations, observations_other_band):
        time_1, magnitude_1, error_1 = self._unpack_observations(observations)
        time_2, magnitude_2, error_2 = self._unpack_observations(
            observations_other_band)
        self._check_arrays_size(time_1, magnitude_1, error_1)
        self._check_arrays_size(time_2, magnitude_2, error_2)
        self._mean_magnitude_1.update(magnitude_1)
        self._mean_magnitude_2.update(magnitude_2)


class Period(StreamingFeature):

    def __init__(self):
        self._streaming_aov = StreamingAOV(plow=.01, phigh=100)

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
        self.features = [self.FEATURES_MAP[e]() for e in features]

    def feature_vector(self):
        values = [feature.value() for feature in self.features]
        return np.asarray(values)

    def update(self, observations, observations_other_band={}):
        for feature in self.features:
            feature.update(observations, observations_other_band)
