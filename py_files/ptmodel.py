# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 17:28:25 2019

@author: dsen
"""

# Reference for HMC: Figure 2 of https://arxiv.org/pdf/1206.1901.pdf

import numpy as np, torch, numpy.random as npr, torch.nn as nn, copy, timeit
from torch.distributions.bernoulli import Bernoulli 
from scipy import signal as sp


    
    
def class PTModel(object):
    
__metaclass__ = abc.ABCMeta
    
    def __init__(self, nn_model,x,y,loss):
        '''
        For every model the number of dimensions need to be specified
        '''
        self.dim = dim
        self.nn_model
        self.x = x
        self.y = y
        self.loss = loss
    
    
    def get_shapes(self) :
        shapes = []
        for param in self.nn_model.parameters() :
            shapes.append(param.shape)
        return shapes

        
        
    def update_grads(self) :
        for param in self.nn_model.parameters() :
            if not (param.grad is None) :
                param.grad.data.zero_()
        self.loss.forward(self.nn_model.forward(self.x), self.y)
        self.loss.backward(retain_graph=True)
    
    def eval_potential_energy(nn_model, x, y, prior_sigma, error_sigma) :
    N, k = np.shape(y)
    #from likelihood:
    pot_energy = N*nn.MSELoss()(nn_model(x), y).data/(2*error_sigma**2)
    # from prior:
    for param in nn_model.parameters() :
        pot_energy += (param**2).sum().data/(2*prior_sigma**2)
    
    return pot_energy.detach()
    
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight)
 
def class HMCsampler(object):
    
    def __init__(self,n_leapfrog, delta_leapfrog):
     
    def generate_momentum(shapes) :
        mom = []
        for shape in shapes :
            mom.append(torch.randn(shape))
        return mom

    def eval_kinetic_energy(mom_list) :
        return sum([(mom**2).sum().data for mom in mom_list])/2





    def update_mom(mom, nn_model, delta_leapfrog, x, y, prior_sigma, error_sigma) :
        N, k = np.shape(y)
        for (i, param) in enumerate(nn_model.parameters()) :
            log_ll_grad = N*param.grad/(2*error_sigma**2)
            log_pr_grad = param/prior_sigma**2
            mom_change = -delta_leapfrog*(torch.add(log_ll_grad,log_pr_grad))
            mom[i].data.add_(mom_change) 
        
    def update_pos(mom, nn_model, delta_leapfrog, x, y, prior_sigma, error_sigma) :
        for (i, param) in enumerate(nn_model.parameters()) :
            pos_change = delta_leapfrog*mom[i]
            param.data.add_(pos_change)
        # update gradients based on new parameters (ie, new positions):
        update_grads(nn_model, x, y)
        
        return pot_energy.detach()
    

    def leapfrog(self,nn_model) :
        
        update_grads(nn_model, x, y)
        
        mom = generate_momentum(shapes)  # Generate momentum variables
        current_mom = copy.deepcopy(mom) # keep copy of initial momentum 
        
        # half step for momentum at beginning
        update_mom(mom, nn_model, delta_leapfrog/2, x, y, prior_sigma, error_sigma)        
        
        # leapfrog steps:
        for l in range(n_leapfrog) :
            # Full step for position
            update_pos(mom, nn_model, delta_leapfrog, x, y, prior_sigma, error_sigma)
            
            # full step for momentum, except at end 
            if l < n_leapfrog-1 :
                update_mom(mom, nn_model, delta_leapfrog, x, y, prior_sigma, error_sigma)
                    
        # half step for momentum at end :
        update_mom(mom, nn_model, delta_leapfrog/2, x, y, prior_sigma, error_sigma)
            
        # Negate momentum at end to make proposal symmetric
        for i in range(len(mom)) :
            mom[i].mul_(-1)
    
        return mom, current_mom, nn_model

    def traverse(self) :
    
        current_nn_model = copy.deepcopy(nn_model)
        proposed_mom, current_mom, proposed_nn_model = leapfrog(nn_model, n_leapfrog, delta_leapfrog, shapes, x, y, prior_sigma)
        
        # Evaluate potential and kinetic energies at start and end 
        current_K = eval_kinetic_energy(current_mom)
        proposed_K = eval_kinetic_energy(proposed_mom)
        current_U = eval_potential_energy(current_nn_model, x, y, prior_sigma, error_sigma)
        proposed_U = eval_potential_energy(proposed_nn_model, x, y, prior_sigma, error_sigma)
        
        # Accept/reject
        if npr.rand() < np.exp(current_U + current_K - proposed_U - proposed_K) :
            return proposed_nn_model, 1
        else :
            return current_nn_model, 0

    
 
"""
Functions for convergence diagnostics:
"""

def compute_acf(X):
    """
    use FFT to compute AutoCorrelation Function
    """
    Y = (X-np.mean(X))/np.std(X)
    acf = sp.fftconvolve(Y,Y[::-1], mode='full') / Y.size
    acf = acf[Y.size:]
    return acf


def gewer_estimate_IAT(np_mcmc_traj, verbose=False):
    """
    use Geyer's method to estimate the Integrated Autocorrelation Time
    """
    acf = np.array( compute_acf(np_mcmc_traj) )
    #for parity issues
    max_lag = 10*np.int(acf.size/10)
    acf = acf[:max_lag]
    # let's do Geyer test
    # "gamma" must be positive and decreasing
    gamma = acf[::2] + acf[1::2]
    N = gamma.size    
    n_stop_positive = 2*np.where(gamma<0)[0][0]
    DD = gamma[1:N] - gamma[:(N-1)]
    n_stop_decreasing = 2 * np.where(DD > 0)[0][0]    
    n_stop = min(n_stop_positive, n_stop_decreasing)
    if verbose:
        print( "Lag_max = {}".format(n_stop) )
    IAT = 1 + 2 * np.sum(acf[1:n_stop])
    return IAT


"""
ESS using Geyer IAT:
"""

def find_ESS(chain, shapes, to_print=False) :
    ESS = []
    means = []
    Vars = []
    T = chain[0].size()[0]
    for (i, shape) in enumerate(shapes) :
        for j in range(shape[0]) :
            if len(list(shape)) == 1 :
                ESS.append(int(T/gewer_estimate_IAT(chain[i][:,j].numpy())))
                means.append(chain[i][:,j].mean())
                Vars.append(chain[i][:,j].var())
                if to_print :
                    print("ESS: %s/%s" % (int(T/gewer_estimate_IAT(chain[i][:,j].numpy())), T))
            else :
                for k in range(shape[1]) :
                    ESS.append(int(T/gewer_estimate_IAT(chain[i][:,j,k].numpy())))
                    means.append(chain[i][:,j,k].mean())
                    Vars.append(chain[i][:,j,k].var())
                    if to_print :
                        print("ESS: %s/%s" % (int(T/gewer_estimate_IAT(chain[i][:,j,k].numpy())), T))
                    
    return np.asarray(ESS), np.asarray(means), np.asarray(Vars)

