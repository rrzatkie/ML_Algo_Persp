#%%
import numpy as np


import sys, os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import mlp
import pandas as pd
import pylab as pl


# %%

data = pd.read_csv('C:\\work\Tutorials\\ml-alg-perp\\MarslandMLAlgo\\Data\\sentiment labelled sentences\\imdb_labelled.txt',sep="\t", header=None, names=["text", "sentiment"])
data = data.append(pd.read_csv("C:\\work\\Tutorials\\ml-alg-perp\\MarslandMLAlgo\\Data\\sentiment labelled sentences\\amazon_cells_labelled.txt",sep="\t", header=None, names=["text", "sentiment"]))
data = data.append(pd.read_csv("C:\\work\\Tutorials\\ml-alg-perp\\MarslandMLAlgo\\Data\\sentiment labelled sentences\\yelp_labelled.txt",sep="\t", header=None, names=["text", "sentiment"]))
# %%
data

# %%
from sklearn.feature_extraction.text import HashingVectorizer

vectorizer = HashingVectorizer(norm='l2', n_features=2)
vectorizer.fit(data.text.values)

train, validate, test = np.split(data.sample(frac=1), [int(.6*len(data)), int(.8*len(data))])

X_train = vectorizer.transform(train.text.values).toarray()
Y_train = train.sentiment.values
# %%
indices0 = np.where(Y_train==0)
indices1 = np.where(Y_train==1)

pl.ion()
pl.plot(X_train[indices0,0], X_train[indices0,1],'go')
pl.plot(X_train[indices1,0], X_train[indices1,1],'rx')
pl.show(block=True)

#%%
X_test = vectorizer.transform(test.text.values).toarray()
Y_test = test.sentiment.values

X_validate = vectorizer.transform(validate.text.values).toarray()
Y_validate = validate.sentiment.values

# %%
net = mlp.mlp(X_train,Y_train.reshape(Y_train.shape[0],1),4,outtype='softmax')
net.mlptrain(X_train,Y_train.reshape(Y_train.shape[0],1),0.25,101)

# Use early stopping
net.earlystopping(X_train,Y_train.reshape(Y_train.shape[0],1),X_validate,Y_validate.reshape(Y_validate.shape[0],1),0.25)
net.confmat(X_test, Y_test.reshape(Y_test.shape[0],1))


# %%
# Test out different sizes of network
# count = 0
# out = np.zeros((10,7))
# for nnodes in [1,2,3,5,10,25,50]:
#     for i in range(10):
#         net = mlp.mlp(X_train,Y_train.reshape(Y_train.shape[0],1),nnodes,outtype='linear')       
#         out[i,count] = net.earlystopping(X_train,Y_train.reshape(Y_train.shape[0],1),X_validate,Y_validate.reshape(Y_validate.shape[0],1),0.25)
#     count += 1
   
# X_test = np.concatenate((X_test,-np.ones((np.shape(X_test)[0],1))),axis=1)
# outputs = net.mlpfwd(X_test)
# print 0.5*sum((outputs-Y_test)**2)

# print out
# print out.mean(axis=0)
# print out.var(axis=0)
# print out.max(axis=0)
# print out.min(axis=0)

# %%
