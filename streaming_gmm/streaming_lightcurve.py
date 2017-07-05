from .lightcurve import unpack_df_in_arrays


def to_chunks(lightcurve_df, chunk_size=50):
    time, mag, error = unpack_df_in_arrays(lightcurve_df)
    samples = time.shape[0]
    for i in range(0, samples, chunk_size):
        yield (time[i:i + chunk_size],
               mag[i:i + chunk_size],
               error[i:i + chunk_size])
