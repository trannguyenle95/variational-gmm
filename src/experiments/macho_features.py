"""Script used for extracting MACHO dataset features in streaming form"""

import numpy as np
import os
import logging
import sys
import argparse
import pandas as pd
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
parser.add_argument("machopath",
                    help="Path of the macho dataset")
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


def _observations_in_chunk(chunk):
    return chunk['time'].shape[0]


def _pack_observations_in_chunks(lightcurve_df, chunk_size=20):
    for time, mag, error in to_chunks(lightcurve_df, chunk_size=chunk_size):
        packed_observations = {'time': time,
                               'magnitude': mag,
                               'error': error}
        yield packed_observations


def calculate_features_in_lightcurve(lc_file_band_1_path,
                                     lc_file_band_2_path,
                                     chunk_size=20):
    lc_df_band_1 = lightcurve.read_from_file(lc_file_band_1_path, skiprows=3)
    lc_df_band_1 = lightcurve.remove_unreliable_observations(lc_df_band_1)
    lc_df_band_2 = lightcurve.read_from_file(lc_file_band_2_path, skiprows=3)
    lc_df_band_2 = lightcurve.remove_unreliable_observations(lc_df_band_2)
    lc_features = LightCurveFeatures()
    observations_seen = 0
    feature_values = []
    observations_band_1 = _pack_observations_in_chunks(lc_df_band_1,
                                                       chunk_size)
    observations_band_2 = _pack_observations_in_chunks(lc_df_band_2,
                                                       chunk_size)
    for obs_1, obs_2 in zip(observations_band_1, observations_band_2):
        logger.info("New observation for %s and %s",
                    lc_file_band_1_path, lc_file_band_2_path)
        lc_features.update(obs_1, obs_2)
        observations_seen += _observations_in_chunk(obs_1)
        feature_dict = lc_features.values()
        feature_dict['observations_seen'] = observations_seen
        feature_values.append(feature_dict)
    return feature_values


def calculate_features_of_file(lc_id, lc_files, lc_dir, output_dir,
                               chunk_size):
    lc_has_two_bands = len(lc_files) == 2
    if lc_has_two_bands:
        logger.info("Calculating features for %s", lc_id)
        features_per_chunk = calculate_features_in_lightcurve(lc_files[0],
                                                              lc_files[1],
                                                              chunk_size)
        feature_df = pd.DataFrame(features_per_chunk)
        file_path = '{}/{}csv'.format(output_dir, lc_id)
        logger.info("Saving to %s", file_path)
        feature_df.to_csv(file_path)
    else:
        logger.error("%s has %d bands", lc_id, len(lc_files))


def calculate_features_in_dir(lc_dir, path_macho, output_dir, n_processes=1,
                              chunk_size=20):
    lc_dir_path = '{}/{}'.format(path_macho, lc_dir)
    files_in_lc_dir = os.listdir(lc_dir_path)
    lc_files = filter(lambda f: f.startswith('lc_'), files_in_lc_dir)
    lc_id_to_file = {}
    for lc_file in lc_files:
        lc_id = lc_file[3:-5]
        lc_file_path = '{}/{}'.format(lc_dir_path, lc_file)
        if lc_id in lc_id_to_file:
            lc_id_to_file[lc_id].append(lc_file_path)
        else:
            lc_id_to_file[lc_id] = [lc_file_path]

    args_for_computation = [(k, v, lc_dir, output_dir, chunk_size)
                            for k, v in lc_id_to_file.items()]
    pool = Pool(processes=n_processes)
    pool.starmap(calculate_features_of_file, args_for_computation)


print(args)
path_macho = args.machopath
logger.info("Calculating features in %s", path_macho)
lc_dirs = [dir_ for dir_ in os.listdir(path_macho)
           if not dir_.startswith('.') and dir_ != 'non_variables']
for lc_dir in lc_dirs:
    logger.info("Calculating features in %s", lc_dir)
    output_dir_path = '{}/streaming_{}_chunks/{}/'.format(args.outputdir,
                                                          args.chunk_size,
                                                          lc_dir)
    output_dir = os.path.dirname(output_dir_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    calculate_features_in_dir(lc_dir, path_macho, output_dir_path,
                              args.n_processes, args.chunk_size)