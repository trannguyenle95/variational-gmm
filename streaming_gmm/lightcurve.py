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
    has_low_error = error < (outlier_threshold * mean_error)
    is_not_outlier = (np.abs(mag - mean_mag) / np.std(mag)) < outlier_threshold
    reliables_idxs = has_low_error & is_not_outlier

    return pack_arrays_in_df(time[reliables_idxs],
                             mag[reliables_idxs],
                             error[reliables_idxs])
