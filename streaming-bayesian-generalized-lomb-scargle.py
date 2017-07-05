
# coding: utf-8

# In[2]:

from streaming_gmm import lightcurve
import matplotlib.pyplot as plt
import numpy as np
import logging


get_ipython().magic('matplotlib inline')
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

LC_PATH = 'data/lc_1.3444.614.B.mjd'


# ## Batch experiment
# 
# In the following cells we load an Eclipsing Binary, so we can calculate
# its period.

# In[3]:

lightcurve_df = lightcurve.read_from_file(LC_PATH, skiprows=3)
lightcurve_df = lightcurve.remove_unreliable_observations(lightcurve_df)
time, mag, error = lightcurve.unpack_df_in_arrays(lightcurve_df)


# In[4]:

color = [1 ,0.498039, 0.313725]
p = plt.plot(time, mag, '*-', color=color, alpha = 0.6)
plt.xlabel("Time")
plt.ylabel("Magnitude")
plt.gca().invert_yaxis()


# In[5]:

def plot_folded_lightcurve(time, mag, period):
    color = [ 0.392157, 0.584314 ,0.929412]
    T = 2 * period
    new_b=np.mod(time, period) / period
    idx=np.argsort(2 * new_b)
    plt.plot(new_b, mag, '*', color=color)
    plt.xlabel("Phase")
    plt.ylabel("Magnitude")
    plt.gca().invert_yaxis()


# ### Gatspy LombScargle
# 
# In the cell below we calculate the periodogram of the lightcurve using
# the `gatspy` Lomb-Scargle algorithm. The idea is to compare this result
# to the streaming LS in batch setting for verification that the implementation
# is correct.

# In[6]:

from gatspy.periodic import LombScargleFast

model = LombScargleFast().fit(time, mag, error)
periods, power = model.periodogram_auto(nyquist_factor=100)
period = periods[np.argmax(power)]
print('Most probable period:', period)

plt.plot(periods, power)
plt.xlim((0.2, 1.4))
plt.ylim((0, 0.8))
plt.xlabel('period (days)')
plt.ylabel('Lomb-Scargle Power')
plt.show()

plot_folded_lightcurve(time, mag, period)


# ### Batch Bayesian GLS implementation
# 
# In the cell below we compute the periodogram and period using the bayesian
# generalized LS algorithm that I implemented. We should see the same period.

# In[7]:

from streaming_gmm.streaming_features import StreamingBGLS

model = StreamingBGLS(freq_multiplier=100)
model.update(time, mag, error)
periods, probability = model.periodogram()
print('Most probable period:', model.most_probable_period())

plt.plot(periods, probability)
plt.xlim((0.2, 1.4))
plt.ylim((0, 2))
plt.xlabel('period (days)')
plt.ylabel('probability %')
plt.show()

plot_folded_lightcurve(time, mag, period)


# ## Streaming experiment

# In[8]:

import time as tm


# ### Bayesian Lomb-Scargle

# In[9]:

from streaming_gmm.streaming_lightcurve import to_chunks

model = StreamingBGLS(freq_multiplier=100)

bgls_batch_times = []

start_time_sec = tm.time()
for time, mag, error in to_chunks(lightcurve_df):
    batch_start_time_sec = tm.time()
    model.update(time, mag, error)
    periods, probability = model.periodogram()
    batch_elapsed_time_sec = tm.time() - batch_start_time_sec
    bgls_batch_times.append(batch_elapsed_time_sec)
    print('Most probable period:', model.most_probable_period())
    print('Batch elapsed time:', batch_elapsed_time_sec, 'secs')
elapsed_time_sec = tm.time() - start_time_sec
print('Total time for lightcurve:', elapsed_time_sec, 'secs')


# In[10]:

model = LombScargleFast()

accum_time = None
accum_mag = None
accum_error = None

n_chunks = 0
gatspy_elapsed_time = []

start_time_sec = tm.time()
for time, mag, error in to_chunks(lightcurve_df):
    if n_chunks == 0:
        accum_time = time
        accum_mag = mag
        accum_error = error
    else:
        accum_time = np.concatenate((accum_time, time))
        accum_mag = np.concatenate((accum_mag, mag))
        accum_error = np.concatenate((accum_error, error))
    batch_start_time_sec = tm.time()
    model.fit(accum_time, accum_mag, accum_error)
    periods, power = model.periodogram_auto(nyquist_factor=100)
    period = periods[np.argmax(power)]
    batch_elapsed_time_sec = tm.time() - batch_start_time_sec
    print('Most probable period:', period)
    print('Batch elapsed time:', batch_elapsed_time_sec, 'secs')
    n_chunks += 1
elapsed_time_sec = tm.time() - start_time_sec
print('Total time for lightcurve:', elapsed_time_sec, 'secs')


# In[ ]:



