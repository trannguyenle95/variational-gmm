import numpy as np
import pandas as pd


def unpack_df_in_arrays(lightcurve_df):
    lightcurve_np = lightcurve_df.as_matrix()
    time = lightcurve_np[:, 0]
    mag = lightcurve_np[:, 1]
    error = lightcurve_np[:, 2]
    return time, mag, error


def pack_arrays_in_df(time, mag, error):
    lightcurve_df = pd.DataFrame()
    lightcurve_df['time'] = pd.Series(time)
    lightcurve_df['mag'] = pd.Series(mag)
    lightcurve_df['error'] = pd.Series(error)
    return lightcurve_df


def read_from_file(filepath, skiprows=1, sep=' '):
    lightcurve_df = pd.read_csv(filepath, skiprows=skiprows, sep=sep,
                                names=['time', 'mag', 'error'])
    return lightcurve_df


def remove_unreliable_observations(lightcurve_df, error_threshold=3,
                                   outlier_threshold=5):
    time, mag, error = unpack_df_in_arrays(lightcurve_df)
    mean_error = error.mean()
    mean_mag = mag.mean()
    has_low_error = error < (error_threshold * mean_error)
    is_not_outlier = (np.abs(mag - mean_mag) / np.std(mag)) < outlier_threshold
    reliables_idxs = has_low_error & is_not_outlier

    return pack_arrays_in_df(time[reliables_idxs],
                             mag[reliables_idxs],
                             error[reliables_idxs])


def _unpack_observations(observations):
    time = observations['time']
    magnitude = observations['magnitude']
    error = observations['error']
    return time, magnitude, error


def align_lightcurves(mjd, mjd2, data, data2, error, error2):
    if len(data2) > len(data):
        new_data2 = []
        new_error2 = []
        new_mjd2 = []
        new_mjd = np.copy(mjd)
        new_error = np.copy(error)
        new_data = np.copy(data)
        count = 0

        for index in range(len(data)):
            where = np.where(mjd2 == mjd[index])

            if np.array_equal(where[0], []) is False:
                new_data2.append(data2[where])
                new_error2.append(error2[where])
                new_mjd2.append(mjd2[where])
            else:
                new_mjd = np.delete(new_mjd, index - count)
                new_error = np.delete(new_error, index - count)
                new_data = np.delete(new_data, index - count)
                count = count + 1

        new_data2 = np.asarray(new_data2).flatten()
        new_error2 = np.asarray(new_error2).flatten()
    else:
        new_data = []
        new_error = []
        new_mjd = []
        new_mjd2 = np.copy(mjd2)
        new_error2 = np.copy(error2)
        new_data2 = np.copy(data2)
        count = 0
        for index in range(len(data2)):
            where = np.where(mjd == mjd2[index])

            if np.array_equal(where[0], []) is False:
                new_data.append(data[where])
                new_error.append(error[where])
                new_mjd.append(mjd[where])
            else:
                new_mjd2 = np.delete(new_mjd2, (index - count))
                new_error2 = np.delete(new_error2, (index - count))
                new_data2 = np.delete(new_data2, (index - count))
                count = count + 1

        new_data = np.asarray(new_data).flatten()
        new_mjd = np.asarray(new_mjd).flatten()
        new_error = np.asarray(new_error).flatten()

    return new_data, new_data2, new_mjd, new_error, new_error2
