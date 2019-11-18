
# Code from Chapter 4 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

# The sinewave regression example

import pylab as pl
import numpy as np

# Set up the data
x = np.linspace(0,1,40).reshape((40,1))
t = np.sin(2*np.pi*x) + np.cos(4*np.pi*x) + np.random.randn(40).reshape((40,1))*0.2
x = (x-0.5)*2

# Split into training, testing, and validation sets
train = x[0::2,:]
test = x[1::4,:]
valid = x[3::4,:]
traintarget = t[0::2,:]
testtarget = t[1::4,:]
validtarget = t[3::4,:]

# Plot the data
pl.subplot(131)
pl.plot(x,t,'go')
pl.xlabel('x')
pl.ylabel('t')


# Perform basic training with a small MLP
import mlp
net = mlp.mlp(train,traintarget,3,outtype='linear')

# Use early stopping
net.earlystopping(train,traintarget,valid,validtarget,0.25, niterations=501)

x2 = np.concatenate((x,-np.ones((np.shape(x)[0],1))),axis=1)
t2 = net.mlpfwd(x2)

pl.subplot(132)
pl.plot(x,t2, 'ro')

pl.subplot(133)

x3 = range(np.shape(net.valerrors)[0])
pl.plot(x3,net.valerrors, 'ro')
pl.plot(x3,net.trainerrors, 'bo')

# Test out different sizes of network
# count = 0
# out = np.zeros((10,7))
# for nnodes in [1,2,3,5,10,25,50]:
#    for i in range(10):
#        net = mlp.mlp(train,traintarget,nnodes,outtype='linear')
#        out[i,count] = net.earlystopping(train,traintarget,valid,validtarget,0.25)
#    count += 1
   
# test = np.concatenate((test,-np.ones((np.shape(test)[0],1))),axis=1)
# outputs = net.mlpfwd(test)
# print 0.5*sum((outputs-testtarget)**2)

# print out
# print out.mean(axis=0)
# print out.var(axis=0)
# print out.max(axis=0)
# print out.min(axis=0)

pl.show()
