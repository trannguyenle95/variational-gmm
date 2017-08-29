"""Script used for extracting MACHO dataset features in streaming form"""

import numpy as np
import os.path
import logging
import sys

sys.path.append(os.path.abspath('../'))

from streaming_gmm import lightcurve
from streaming_gmm.features.lightcurve_features import LightCurveFeatures
from streaming_gmm.streaming_lightcurve import to_chunks

PATH_MACHO = '{}/{}'.format(os.path.dirname(os.path.abspath(__file__)),
                            '../../data/macho-raw')

LOGGING_FORMAT = '%(asctime)s|%(name)s|%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO,
                    format=LOGGING_FORMAT,
                    stream=sys.stdout)
logger = logging.getLogger(__name__)


def _pack_observations_in_chunks(lightcurve_df):
    for time, mag, error in to_chunks(lightcurve_df, chunk_size=20):
        packed_observations = {'time': time,
                               'magnitude': mag,
                               'error': error}
        yield packed_observations


def calculate_features_in_lightcurve(lc_file_band_1_path, lc_file_band_2_path):
    lc_df_band_1 = lightcurve.read_from_file(lc_file_band_1_path, skiprows=3)
    lc_df_band_1 = lightcurve.remove_unreliable_observations(lc_df_band_1)
    lc_df_band_2 = lightcurve.read_from_file(lc_file_band_2_path, skiprows=3)
    lc_df_band_2 = lightcurve.remove_unreliable_observations(lc_df_band_2)
    lc_features = LightCurveFeatures()
    observations_band_1 = _pack_observations_in_chunks(lc_df_band_1)
    observations_band_2 = _pack_observations_in_chunks(lc_df_band_2)
    for obs_1, obs_2 in zip(observations_band_1, observations_band_2):
        logger.info("New observation for %s and %s",
                    lc_file_band_1_path, lc_file_band_2_path)
        #lc_features.update(obs_1, obs_2)


def calculate_features_in_dir(lc_dir):
    lc_dir_path = '{}/{}'.format(PATH_MACHO, lc_dir)
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

    for lc_id, lc_files in lc_id_to_file.items():
        lc_has_two_bands = len(lc_files) == 2
        if lc_has_two_bands:
            calculate_features_in_lightcurve(lc_files[0], lc_files[1])
        else:
            logger.error("%s has %d bands", lc_id, len(lc_files))

logger.info("Calculating features in %s", os.path.abspath(PATH_MACHO))
lc_dirs = [dir_ for dir_ in os.listdir(PATH_MACHO)
           if not dir_.startswith('.') and dir_ != 'non_variables']
for lc_dir in lc_dirs:
    logger.info("Calculating features in %s", lc_dir)
    calculate_features_in_dir(lc_dir)
