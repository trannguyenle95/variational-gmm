
# coding: utf-8

# # Streaming Gaussian Mixture Model
# 
# ## Update features when a new observation arrives
# 
# TODO
# 
# ## Update the classifier
# 
# ### General idea
# 
# Let $X = \{ x_1, \dotsc, x_N \}$ a collection of $N$ points. Now, lets suppose that the features of $x_j$ are updated, and lets call $x_j^{\star}$ the updated point. Lastly, lets define $X_{-j} = \{ x_1, \dotsc, x_{j-1}, x_{j+1}, \dotsc, x_N \}$.
# 
# The posterior before $x_j$ moves, and using $P(\Theta)$ as the prior is
# 
# $$P(\Theta \mid X) \propto P(X \mid \Theta)P(\Theta)$$,
# 
# where $P(X \mid \Theta) = \prod_i^N P(x_i \mid \Theta)$.
# 
# Now, after $x_j$ moves to $x_j^{\star}$, the posterior is
# 
# \begin{align*}
# P(\Theta \mid X_{-j}, x_j^{\star}) &\propto P(X_{-j}, x_j^{\star} \mid \Theta)P(\Theta) \\
# &\propto P(X_{-j} \mid \Theta)P(x_j^{\star} \mid \Theta)P(\Theta) \\
# &\propto \frac{P(X \mid \Theta)}{P(x_j \mid \Theta)}P(x_j^{\star} \mid \Theta)P(\Theta) \\
# &\propto \frac{P(x_j^{\star} \mid \Theta)}{P(x_j \mid \Theta)}P(X \mid \Theta)P(\Theta) \\
# &\propto \frac{P(x_j^{\star} \mid \Theta)}{P(x_j \mid \Theta)}P(\Theta \mid X)
# \end{align*}
# 
# ### Mixture of GMM update

# ## Synthetic data generation
# 
# In the cell below we are going to generate synthetic data from multiple GMMs (one GMM per class).

# In[2]:

import pandas as pd
import numpy as np
from scipy.stats import invwishart
import seaborn as sns
import matplotlib.pyplot as plt
import pymc3 as pm
import scipy

get_ipython().magic('matplotlib inline')
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

sns.set(rc={"figure.figsize": (6, 6)})

CLASS_COLORS = ['#66c2a5',
                '#fc8d62',
                '#8da0cb',
                '#e78ac3',
                '#a6d854',
                '#ffd92f',
                '#e5c494',
                '#b3b3b3']


# In[160]:

class GMMDataGenerator:
     
    def __init__(self, m, k, d, mu_interval=(-10, 10), mu_var=(-4, 4),
                 cov_var=(-1, 1), gamma_0=3.0, alpha_0=5.0):
        """
        Args:
            m (int): number of classes.
            k (int): number of components per class.
            d (int): dimension of data.
        """
        self.m = m
        self.k = k
        self.d = d
        self._mu_0 = np.zeros(d)
        self.gamma_0 = gamma_0
        self.alpha_0 = alpha_0
        self.mu_interval = mu_interval
        self.mu_var = mu_var
        self.cov_var = cov_var
        self.mu = np.zeros((m, k, d))
        self.cov = np.zeros((m, k, d, d))
        self._random_mean_and_cov()
        self.weights = np.random.dirichlet(self.gamma_0 * np.ones(k), size=m)
        
    def generate(self, n=2000):
        self.X = np.zeros((n, self.d))
        self.Z = np.zeros(n)
        
        # generate the class distributions
        self.pi = np.random.dirichlet(self.alpha_0 * np.ones(self.m))
        for i in range(n):
            # generate random class of this observation
            z_i = np.argmax(np.random.multinomial(1, self.pi))
            self.Z[i] = z_i
            
            # get a random component of this class
            w_im = np.argmax(np.random.multinomial(1, self.weights[z_i]))
            
            # generate the features
            self.X[i, :] = np.random.multivariate_normal(self.mu[z_i, w_im], 
                                                         self.cov[z_i, w_im])
    
    def get_obs(self):
        return self.X
    
    def get_labels(self):
        return self.Z.astype(int)
    
    def get_mu(self):
        return self.mu
    
    def get_cov(self):
        return self.cov
    
    def plot_2dproj(self, feat0=0, feat1=1, xlabel='', ylabel='', title=''):
        if not xlabel:
            xlabel = 'feature {}'.format(feat0)
        if not ylabel:
            ylabel = 'feature {}'.format(feat1)
        for m in range(self.m):
            l_class = np.where(self.Z == m)[0]
            plt.scatter(self.X[l_class, feat0], self.X[l_class, feat1], 
                        facecolor=CLASS_COLORS[m], alpha=0.5,
                        edgecolor='black', linewidth=0.15)
            plt.scatter(self.mu[m, :, feat0], self.mu[m, :, feat1],
                        facecolor=CLASS_COLORS[m], s=40)
        plt.title(title)
        #plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    
    def _random_mean_and_cov(self):
        a, b = self.mu_interval
        std_min, std_max = self.cov_var
        for i in range(self.m):
            mu_center = (b - a) * np.random.random(size=self.d) + a
            mu_max = mu_center + self.mu_var[1]
            mu_min = mu_center + self.mu_var[0]
            for j in range(self.k):
                mu_center_k = ((mu_max - mu_min) 
                               * np.random.random(size=self.d)
                               + mu_min)
                self.cov[i, j, :] = invwishart.rvs(2 * self.d, np.eye(self.d))
                self.mu[i, j, :] = np.random.multivariate_normal(
                    self._mu_0 + mu_center_k, self.cov[i, j, :])


# In[141]:

# Number of Gaussian Mixtures (classes).
M = 2

# Number of components per class.
K = 3

# Dimension of data.
D = 2

#np.random.seed()

synthetic_gmm = GMMDataGenerator(m=M, k=K, d=D)
synthetic_gmm.generate(n=2000)
synthetic_gmm.plot_2dproj()


# In[171]:

synthetic_gmm = GMMDataGenerator(m=1, k=K, d=D, mu_var=(-4, 4))
synthetic_gmm.generate(n=100)
synthetic_gmm.plot_2dproj()


# In[6]:

def plot_gmm(X, mu, W):
    assert mu.shape[0] == W.shape[0]
    dims = mu.shape[0]
    plt.scatter(X[:, 0], X[:, 1], 
                alpha=0.5,
                edgecolor='black', linewidth=0.15)
    min_x, min_y = np.amin(X, axis=0)
    max_x, max_y = np.amax(X, axis=0)
    x, y = np.mgrid[min_x:max_x:0.1, min_y:max_y:0.1]
    z = np.zeros(x.shape + (2,))
    z[:, :, 0] = x;
    z[:, :, 1] = y
    for i in range(mu.shape[0]):
        f_z = scipy.stats.multivariate_normal.pdf(z, mu[i, :], W[i, :])
        plt.contour(x, y, f_z, antialiased=True)
    plt.show()


# In[1]:

print('holaadasdsasdasdaasdas')


# In[92]:

X = synthetic_gmm.get_obs()
C = synthetic_gmm.get_labels()
observed_mu = np.zeros((M, D))
for m in range(1):
    indexs = np.where(C == m)
    observed_mu[m] = X[indexs, :].mean(axis=1)
    print(observed_mu[m])


# In[8]:

print(X[0, :].shape)


# In[9]:

print(np.squeeze(X[np.where(C == 0), :]).shape)


# In[20]:

from mgmm_pymc.models import GaussianMixtureModel

GAMMA_0 = 5. * np.ones(K)
MU_0 = np.zeros(D)
NU_0 = 3.0

gmms = [GaussianMixtureModel(K=K, 
                             observed=np.squeeze(X[np.where(C == m), :]),
                             gamma_0=GAMMA_0,
                             mu_0=MU_0,
                             nu_0=NU_0) 
        for m in range(M)]


# In[23]:

get_ipython().run_cell_magic('time', '', 'mu_means = np.zeros((M, K, D))\nlamb_means = np.zeros((M, K, D, D))\nrho_means = np.zeros((M, K))\n\ntraces = []\n\nfor m in range(M):\n    trace = gmms[m].train(samples=1e4, burn_in=2.5e3)\n    traces.append(trace)\n    mu_means[m, :] = gmms[m].mu_means\n    lamb_means[m, :] = gmms[m].lambda_means\n    rho_means[m, :] = gmms[m].rho_means\nprint(mu_means)')


# In[ ]:

pm.plots.traceplot(traces[0], ['rho']);


# In[11]:

from vbmm.varmix import run


# In[93]:

get_ipython().run_cell_magic('time', '', 'mu, W, vk = run(np.squeeze(X[np.where(C == 0), :]), K)\nW = np.asarray(W)')


# In[94]:

print(np.linalg.inv(W))


# In[95]:

print(W.shape)
plot_gmm(X, mu, np.linalg.inv(vk[:, np.newaxis, np.newaxis]*W))
print(mu)
print(synthetic_gmm.get_mu())


# In[14]:

from vb_mgmm.var_bayes_gmm import VariationalBayesGMM


# In[15]:

# We want to fail fast
np.seterr(all='raise')


# In[96]:

get_ipython().run_cell_magic('time', '', 'vbGmm = VariationalBayesGMM(K, D)\nvbGmm.fit(np.squeeze(X[np.where(C == 0), :]), max_iter=20)')


# In[97]:

#print(vbGmm.m_k)
print(synthetic_gmm.get_mu())
print(mu)


# In[98]:

plot_gmm(X, vbGmm.m_k, np.linalg.inv(vbGmm.nu_k[:, np.newaxis, np.newaxis]*vbGmm.W_k))


# In[85]:

import vb_mgmm.streaming_vbgmm


# In[ ]:



