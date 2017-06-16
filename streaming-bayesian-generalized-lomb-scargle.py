
# coding: utf-8

# In[58]:

from streaming_gmm import lightcurve
import matplotlib.pyplot as plt
import numpy as np

get_ipython().magic('matplotlib inline')
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

LC_PATH = 'data/lc_1.3444.614.B.mjd'


# ## Batch experiment
# 
# In the following cells we load an Eclipsing Binary, so we can calculate
# its period.

# In[48]:

lightcurve_df = lightcurve.read_from_file(LC_PATH, skiprows=3)
lightcurve_df = lightcurve.remove_unreliable_observations(lightcurve_df)
time, mag, error = lightcurve.unpack_df_in_arrays(lightcurve_df)


# In[49]:

color = [1 ,0.498039, 0.313725]
p = plt.plot(time, mag, '*-', color=color, alpha = 0.6)
plt.xlabel("Time")
plt.ylabel("Magnitude")
plt.gca().invert_yaxis()


# In[75]:

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

# In[88]:

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

# In[89]:

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

# In[ ]:



