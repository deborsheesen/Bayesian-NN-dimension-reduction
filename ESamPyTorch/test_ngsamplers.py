#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 11:32:09 2018

@author: msachs2
"""

import numpy as np
import matplotlib.pyplot as plt

import statsmodels as models
import ngsamplers as samplers

import scipy.stats as scstats
# import mini MD modules
#import models
import ngsamplers
import outputshedulers as outp
# import function to compute autocorrelation
from integrators import autocorr

np.random.seed(seed=11) #Fix seed 

data_dim = 2 # Dimension of predictor variable
Ndata1 = 10  # Number of points with class label 0
Ndata2 = 10  # Number of points with class label 1

mu1 = np.array([-4,0]) # mean of predictor variables with class label 0
mu2 =  np.array([4,0]) # mean of predictor variables with class label 1
cov1 = np.eye(data_dim) # covariance of predictor variables with class label 0
cov2 = np.eye(data_dim) # covariance of predictor variables with class label 1


# Sample data points 
X1 = np.random.multivariate_normal(mu1,cov1,size=Ndata1)
Y1 = np.zeros(Ndata1)
X2 = np.random.multivariate_normal(mu2,cov2,size=Ndata2)
Y2 = np.ones(Ndata2)

X = np.concatenate((X1,X2))
Y = np.concatenate((Y1,Y2)) 


data = [X,Y] # data set in the format used in the logistic regression model below




ndata = Y.shape[0]
color_dict= {0:'red', 1 :'blue'}
colors = [color_dict[Y[i]] for i in range(ndata)]
plt.scatter(X[:,0],X[:,1], c=colors)
plt.title('Data') 
plt.xlim([mu1[0]-3*np.sqrt(cov1[0,0]),+mu2[0]+3*np.sqrt(cov2[0,0])])
plt.ylim([mu1[1]-3*np.sqrt(cov1[1,1]),+mu2[1]+3*np.sqrt(cov2[1,1])])
plt.show()
prior = models.IsotropicGaussian(mean=np.zeros(data_dim), var=100*np.ones(data_dim))
model = models.BayesianLogisticRegression(dim=2, prior=prior)
model.set_data(training_data = [X,Y])

repmodel = models.ReplicatedStatsModel(model, # model specifying target density 
                                          nreplicas=10 # number of replicas/copies to be created
                                         )

sampler = ngsamplers.NGEnsembleQuasiNewtonNose(repmodel, 
                                               h=.1, 
                                               Tk_B=1.0, 
                                               gamma=1.0, 
                                               regparams=1.0, 
                                               B_update_mod=1,
                                               mu=1.0)
'''
sampler = ngsamplers.NGEnsembleQuasiNewton(repmodel, 
                                               h=.1, 
                                               Tk_B=1.0, 
                                               gamma=1.0, 
                                               regparams=1.0, 
                                               B_update_mod=1)
'''
op = outp.BufferedOutputsheduler(sampler, 
                                 Nsteps=1000, 
                                 varname_list=['q','p'], 
                                 modprnt=1)
sampler.run(initial_values={'q' : np.random.normal(0.0,1.0, repmodel.dim),
                            'xi' : np.ones(repmodel.dim)
                            }) # sample the system
fig, ax = plt.subplots()
ax.plot(op.traj_q[:,0])
plt.show()
fig, ax = plt.subplots()
ax.plot(op.traj_q[:,1])
plt.show()

Nx1 = 50
Nx2 = 50
xx1 = np.linspace(mu1[0]-3*np.sqrt(cov1[0,0]),+mu2[0]+3*np.sqrt(cov2[0,0]),Nx1)
xx2 = np.linspace(mu1[1]-3*np.sqrt(cov1[1,1]),+mu2[1]+3*np.sqrt(cov2[1,1]),Nx2)

#figb, axb = model.plot_prediction(op.traj_q, grid=[xx1,xx2], Neval=100 )
#figb.show()
#plt.show()