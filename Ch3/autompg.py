# Code from Chapter 3 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

# This is the start of a script for you to complete

#%%
import sys, os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import numpy as np
import linreg

auto = np.genfromtxt("C:\\work\\Tutorials\\ml-alg-perp\\MarslandMLAlgo\\Data\\auto-mpg.data", comments='"')
auto = auto[~np.isnan(auto).any(axis=1)]

#%%
# auto[:,:7] = auto[:,:7]-auto[:,:7].mean(axis=0)
# auto[:,:7] = auto[:,:7]/auto[:,:7].var(axis=0)

trainin = auto[::2,1:8]
traintgt = auto[::2,:1]

testin = auto[1::2,1:8]
testtgt = auto[1::2,:1]

# This is the training part

beta = linreg.linreg(trainin,traintgt)
testin = np.concatenate((testin,-np.ones((np.shape(testin)[0],1))),axis=1)
testout = np.dot(testin,beta)
error = np.sum((testout - testtgt)**2)
print error

# %%
