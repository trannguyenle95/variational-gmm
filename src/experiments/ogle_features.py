import numpy as np
import os
import logging
import sys
import argparse
import pandas as pd
import time as tm
from multiprocessing import Pool

sys.path.append('{}/../'.format(os.path.dirname(os.path.abspath(__file__))))

from streaming_gmm import lightcurve
from streaming_gmm.features.lightcurve_features import LightCurveFeatures
from streaming_gmm.streaming_lightcurve import to_chunks


LOGGING_FORMAT = '%(asctime)s|%(name)s|%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO,
                    format=LOGGING_FORMAT,
                    stream=sys.stdout)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("oglepath",
                    help="Path of the ogle dataset")
parser.add_argument("outputdir",
                    help="Path of the output dir")
parser.add_argument("--n-processes",
                    help="Number of processes for calculating the features",
                    type=int,
                    default=1)
parser.add_argument("--chunk-size",
                    help="Size of the observations chunk",
                    type=int,
                    default=20)
args = parser.parse_args()

FEATURES = ['mean_magnitude', 'mean_variance', 'period', 'stetson_k', 'std',
            'flux_percentile_ratio_mid_50', 'flux_percentile_ratio_mid_65',
            'flux_percentile_ratio_mid_80', 'flux_percentile_ratio_mid_20',
            'flux_percentile_ratio_mid_35', 'range_cs']


def _pack_observations_in_chunks(lightcurve_df, chunk_size=20):
    for time, mag, error in to_chunks(lightcurve_df, chunk_size=chunk_size):
        packed_observations = {'time': time,
                               'magnitude': mag,
                               'error': error,
                               'aligned_time': time,
                               'aligned_magnitude': mag,
                               'aligned_error': error}
        yield packed_observations


def _observations_in_chunk(chunk):
    return chunk['time'].shape[0]


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def open_ogle(path, n=500):
    columns = [0, 1, 2]
    df = pd.read_csv(path, comment='#', sep='\s+', header=None)
    df.columns = ['a', 'b', 'c']
    df = df[df.a > 0]
    df = df.sort_values(by=[df.columns[columns[0]]])

    time = np.array(df[df.columns[columns[0]]].values, dtype=float)
    magnitude = np.array(df[df.columns[columns[1]]].values, dtype=float)
    error = np.array(df[df.columns[columns[2]]].values, dtype=float)

    # Not Nan
    not_nan = np.where(~np.logical_or(np.isnan(time), np.isnan(magnitude)))[0]
    time = time[not_nan]
    magnitude = magnitude[not_nan]
    error = error[not_nan]

    if len(time) > n:
        time = time[:n]
        magnitude = magnitude[:n]
        error = error[:n]

    return lightcurve.pack_arrays_in_df(time, magnitude, error)


def calculate_one_band_features(path, chunk_size):
    lc_df = open_ogle(path)
    lc_df = lightcurve.remove_unreliable_observations(lc_df)

    lc_features = LightCurveFeatures(features=FEATURES)
    observations_seen = 0
    feature_values = []
    observations = _pack_observations_in_chunks(lc_df, chunk_size)

    for i, obs_1 in enumerate(observations):
        logger.info("New observation for %s", path)
        lc_features.update(obs_1)
        observations_seen += _observations_in_chunk(obs_1)
        feature_dict = lc_features.values()
        feature_dict['observations_seen'] = observations_seen
        feature_dict['chunk_numb'] = i
        feature_values.append(feature_dict)

    return feature_values


def calculate_features_of_file(lc_id, lc_file, lc_dir, output_dir, chunk_size):
    logger.info("Calculating features for %s", lc_id)
    try:
        features_per_chunk = calculate_one_band_features(lc_file, chunk_size)
    except Exception as e:
        features_per_chunk = None
        print("FALLOOOO", e)
        logger.exception("Couldn't calculate features for lightcurve %s",
                         lc_id)

    if features_per_chunk:
        feature_df = pd.DataFrame(features_per_chunk)
        file_path = '{}{}.csv'.format(output_dir, lc_id)
        logger.info("Saving to %s", file_path)
        feature_df.to_csv(file_path)


def calculate_features_in_dir(lc_dir, path_ogle, output_dir, n_processes=1,
                              chunk_size=20):
    lc_dir_path = '{}/{}'.format(path_ogle, lc_dir)
    files_in_lc_dir = os.listdir(lc_dir_path)
    lc_files = filter(lambda f: f.endswith('dat'), files_in_lc_dir)
    lc_id_to_file = {}
    for lc_file in lc_files:
        lc_id = lc_file[:-4]
        lc_file_path = '{}/{}'.format(lc_dir_path, lc_file)
        lc_id_to_file[lc_id] = lc_file_path

    args_for_computation = [(k, v, lc_dir, output_dir, chunk_size)
                            for k, v in lc_id_to_file.items()]
    pool = Pool(processes=n_processes)
    pool.starmap(calculate_features_of_file, args_for_computation)


print(args)
path_ogle = args.oglepath
logger.info("Calculating features in %s", path_ogle)
lc_dirs = [dir_ for dir_ in os.listdir(path_ogle)
           if not dir_.startswith('.') and dir_ != 'non_variables']

start_time = tm.time()
for lc_dir in lc_dirs:
    logger.info("Calculating features in %s", lc_dir)
    output_dir_path = '{}/streaming_{}_chunks/{}/'.format(args.outputdir,
                                                          args.chunk_size,
                                                          lc_dir)
    output_dir = os.path.dirname(output_dir_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    calculate_features_in_dir(lc_dir, path_ogle, output_dir_path,
                              args.n_processes, args.chunk_size)

elapsed_time = tm.time() - start_time
logger.info("Elapsed time: %d secs", elapsed_time)
