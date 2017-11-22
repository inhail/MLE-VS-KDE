# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 09:50:34 2017

@author: user
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors.kde import KernelDensity

# read file as array
data = np.loadtxt('data.txt') # the index start from 0
x_axis = np.linspace(-30,50,num=250)

#------------Maximum likelihood-------------------------
# gaussian function: we assume that it fits gaussian distribution
def gaussian(x,mu,sigma):
    return np.exp(-np.power(x - mu, 2.) / (2 * sigma**2))/(sigma*np.sqrt(2*np.pi))

# define parameters
def theta(x):
    mu = np.mean(x)
    sigma = np.sqrt(np.dot(x - mu, (x - mu).T) / (x.shape[0]+1))
    # '(x.shape[0]+1)' let sigman being unbiased
    return mu,sigma

mu,sigma = theta(data)
index = np.arange(1,len(data)+1)
#-------------------------------------------------------------------

#---------------KDE-------------------------------------------------
data_KDE = data[:,np.newaxis]
kde = KernelDensity(kernel='gaussian',bandwidth=0.75 ).fit(data_KDE)
lon_dens = kde.score_samples(x_axis[:,np.newaxis])
kde2 = KernelDensity(kernel='gaussian',bandwidth=0.5 ).fit(data_KDE)
lon_dens2 = kde2.score_samples(x_axis[:,np.newaxis])
kde3 = KernelDensity(kernel='gaussian',bandwidth=0.3 ).fit(data_KDE)
lon_dens3 = kde3.score_samples(x_axis[:,np.newaxis])
#-------------------------------------------------------------------

# 2 subplots
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(14,6))
ax1.scatter(index,data,color = 'red')
ax1.set_title('Data Points')
ax2.plot(x_axis, gaussian(x_axis,mu,sigma), color='blue',label='MLE')
ax2.set_title('MLE')
ax2.hist(data_KDE[:, 0], bins=20,color = 'yellow', normed=True,label='Histogram')
ax2.plot(x_axis,np.exp(lon_dens),color='purple',label='KDE b=0.75')
ax2.plot(x_axis,np.exp(lon_dens2),color='green',label='KDE_b=0.5')
ax2.plot(x_axis,np.exp(lon_dens3),color='plum',label='KDE_b=0.3')
ax2.legend(fontsize=12)
fig.suptitle('HW#1')
plt.show()
fig.savefig('HW1')
