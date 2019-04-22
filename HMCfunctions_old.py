# Reference for HMC: Figure 2 of https://arxiv.org/pdf/1206.1901.pdf

import numpy as np, torch, numpy.random as npr, torch.nn as nn, copy, timeit
from torch.distributions.bernoulli import Bernoulli 
from scipy import signal as sp
from time import time
import matplotlib.pyplot as plt
from copy import deepcopy

from pylab import plot, show, legend

np.seterr(all='raise')


#################################################################################################
#-------------------------------------------- Model --------------------------------------------#
#################################################################################################

class model(object):
    def __init__(self, x, y, prior_sigma, error_sigma, nn_model):
        # self.radius is an instance variable
        self.x = x
        self.y = y
        self.prior_sigma = prior_sigma
        self.error_sigma = error_sigma
        self.nn_model = nn_model
        
    def shape(self) :
        shapes = []
        for param in self.nn_model.parameters() :
            shapes.append(param.shape)
        return shapes
    
    def update_grad(self) :
        for param in self.nn_model.parameters() :
            if not (param.grad is None) :
                param.grad.data.zero_()
        if not (self.x.grad is None) :
            self.x.grad.data.zero_()
        loss = nn.MSELoss()(self.nn_model(self.x), self.y)
        loss.backward(retain_graph=True)

    def init_normal(self) :
        self.set_params(torch.randn(self.n_params()))
        self.update_grad()

    def n_params(self) :
        return (sum(p.numel() for p in self.nn_model.parameters()) + np.prod(np.shape(self.x))).item()

    def get_params(self) :
        return torch.cat([torch.cat([param.view(-1) for param in self.nn_model.parameters()]), self.x.view(-1)]).data

    def grad(self) :
        return torch.cat([torch.cat([param.grad.view(-1) for param in self.nn_model.parameters()]), self.x.grad.view(-1)]).data
    
    def set_params(self, param) :
        assert len(param) == self.n_params(), "Number of parameters do not match"
        shapes = self.shape()
        counter = 0
        for i, parameter in enumerate(self.nn_model.parameters()) :
            parameter.data = param[counter:(counter+np.prod(shapes[i]))].reshape(shapes[i])
            counter += np.prod(shapes[i])  
        self.x.data = param[counter::].data.reshape(np.shape(self.x)).clone()
        self.update_grad() 
    

    
class sampler(object) :
    def __init__(self, model, stepsize, M_precond=0) :
        self.model = model
        self.momentum = torch.randn(self.model.n_params())
        self.position = self.model.get_params()
        self.stepsize = stepsize
        self.M_precond = M_precond
        if self.M_precond == 0 :
            self.M_precond = torch.eye(self.model.n_params())
        
    def generate_momentum(self) :
        self.momentum = torch.randn(self.model.n_params())
        
    def update_momentum(self, delta=1) :
        N = np.shape(self.model.y)[0]
        param, param_grad = self.model.get_params(), self.model.grad()
        log_ll_grad = N*param_grad/(2*self.model.error_sigma**2)
        log_pr_grad = param/self.model.prior_sigma**2
        momentum_change = -delta*self.stepsize*(torch.add(log_ll_grad,log_pr_grad))
        self.momentum += momentum_change  
        
    def update_position(self, delta=1) :
        pos_change = delta*self.stepsize*self.momentum/torch.diag(self.M_precond)
        param = self.model.get_params() + pos_change
        self.model.set_params(param)
        # update gradients based on new parameters (ie, new positions):
        self.model.update_grad()
        
        
    def potential_energy(self) :
        N = np.shape(self.model.y)[0]
        #from likelihood:
        pot_energy = N*nn.MSELoss()(self.model.nn_model(self.model.x), self.model.y).data/(2*self.model.error_sigma**2)
        # from prior:
        for param in self.model.nn_model.parameters() :  # for theta
            pot_energy += (param**2).sum().data/(2*self.model.prior_sigma**2)
        pot_energy += (self.model.x**2).sum().data/(2*self.model.prior_sigma**2) #for x
        return pot_energy.detach()
    
    def kinetic_energy(self) :
        return (self.momentum**2/torch.diagonal(self.M_precond)).sum().data/2
    
    def total_energy(self) :
        return self.potential_energy() + self.kinetic_energy()     
         
        
#################################################################################################
#--------------------------------------------- HMC ---------------------------------------------#
#################################################################################################        
        
class HMC(object) :
    def __init__(self, sampler, n_leapfrog, Nsteps):
        self.sampler = sampler
        self.n_leapfrog = n_leapfrog
        self.Nsteps = Nsteps
        self.chain = torch.zeros(self.Nsteps+1,self.sampler.model.n_params())
        self.minlf = 50
        self.maxlf = 500
        self.minstepsize = 1e-3
        self.n_accepted = 0


    def leapfrog(self) :
        self.sampler.model.update_grad()
        self.sampler.generate_momentum()  
        energy_start = self.sampler.total_energy()
        # half step for momentum at beginning
        self.sampler.update_momentum(1/2)        
        # leapfrog steps:
        for l in range(self.n_leapfrog) :
            # Full step for position
            self.sampler.update_position()
            # full step for momentum, except at end 
            if l < self.n_leapfrog-1 :
                self.sampler.update_momentum()
        # half step for momentum at end :
        self.sampler.update_momentum(1/2)
        # Negate momentum at end to make proposal symmetric
        self.sampler.momentum.mul_(-1)
        energy_end = self.sampler.total_energy()
        
        return self, energy_start, energy_end

    def HMC_1step(self) :

        start_nn_model = deepcopy(self.sampler.model.nn_model)
        self, energy_start, energy_end = self.leapfrog()

        # Accept/reject
        energy_diff = energy_end - energy_start
        if energy_diff > 10 :
            accepted = True 
        elif energy_diff < -10 :
            accepted = False 
        else :
            accepted = (npr.rand() < np.exp(energy_diff))
            self.n_accepted += accepted
            if not accepted :
                self.sampler.model.nn_model = start_nn_model
                
        del start_nn_model
                
        
    def limitleapfrog(self) :
        if self.n_leapfrog > self.maxlf :
            self.n_leapfrog = self.maxlf 
        elif self.n_leapfrog < self.minlf :
            self.n_leapfrog = self.minlf
        if self.sampler.stepsize < self.minstepsize :
            self.sampler.stepsize = self.minstepsize
    
    def adapt_leapfrog(self, t) :
        if self.n_accepted <= 0.2*t :
            self.n_leapfrog = int(self.n_leapfrog/1.5)
            self.sampler.stepsize /= 1.05
        elif self.n_accepted >= 0.8*t :
            self.n_leapfrog *= int(self.n_leapfrog*1.5)
            self.sampler.stepsize *= 1.05
        self.limitleapfrog()
        
    def feed(self, t) :
        self.chain[t+1] = self.sampler.model.get_params()
    
    def sample(self) :

        self.feed(0)
        start_time = time()
        for t in range(self.Nsteps) :         
            self.HMC_1step()
            #self.adapt_leapfrog(t+1)
            self.feed(t)
            if ((t+1) % (int(self.Nsteps/10)) == 0) or (t+1) == self.Nsteps :
                print("iter %6d/%d after %.2f min | accept_rate %.3f | MSE loss %.3f | stepsize %.3f | nleapfrog %i" % (
                      t+1, self.Nsteps, (time() - start_time)/60, float(self.n_accepted) / float(t+1), 
                      nn.MSELoss()(self.sampler.model.nn_model(self.sampler.model.x), self.sampler.model.y), self.sampler.stepsize, self.n_leapfrog))
                
    def ESS(self) :
        return self.Nsteps/np.asarray([gewer_estimate_IAT(self.chain[:,i].numpy()) for i in range(np.shape(self.chain)[1])])
    
    def plot(self) :
        ypred_final = self.sampler.model.nn_model(self.sampler.model.x)
        plt.plot(list(ypred_final[:,0]), list(ypred_final[:,1]), 'o', markersize=2)
        plt.grid(True)

#################################################################################################
#-------------------------------------------- BAOAB --------------------------------------------#
#################################################################################################
        
        
class BAOAB(object) :
    def __init__(self, sampler, Nsteps, beta, gamma):
        
        self.sampler = sampler
        self.Nsteps = Nsteps
        self.chain = torch.zeros(self.Nsteps+1,self.sampler.model.n_params())
        self.n_accepted = 0
        self.beta = beta 
        self.gamma = gamma
        self.alpha = np.exp(-self.gamma*self.sampler.stepsize)
        self.Rn = torch.randn(self.sampler.model.n_params())
        
    def propose(self) :
        pos, grad = self.sampler.model.get_params(), self.sampler.model.grad()
        
        # half step for momentum:
        self.sampler.update_momentum(1/2)
        p1 = deepcopy(self.sampler.momentum.data)
        
        # half step for position:
        self.sampler.update_position(1/2)
        
        # update momentum with random variables:
        self.Rn = torch.randn(self.sampler.model.n_params())
        self.sampler.momentum = self.alpha*self.sampler.momentum + np.sqrt((1-self.alpha**2)/self.beta)*self.Rn
        p2 = deepcopy(self.sampler.momentum.data)
        
        # half step for position
        self.sampler.update_position(1/2)
        
        # half step for momentum:
        self.sampler.update_momentum(1/2)
        
        return self, p1, p2
        
    def g(self, x) :
        return torch.exp(-self.beta/(2*(1-self.alpha**2)*torch.norm(x)))
        
    def MH_step(self) :
        energy_start = self.sampler.total_energy()
        model_start, momentum_start = deepcopy(self.sampler.model), deepcopy(self.sampler.momentum)
        self, p1, p2 = self.propose()
        energy_end = self.sampler.total_energy()
        accept_ratio = torch.exp(-self.beta*(energy_end-energy_start))*self.g(self.alpha*p2-p1)/self.g(self.Rn)
        #print(accept_ratio)
        u = torch.rand(1)
        
        accepted = (torch.rand(1) < accept_ratio)
        self.n_accepted += accepted
        
        if not accepted :                     # if reject
            self.sampler.model = deepcopy(model_start)   # set position to initial position
            self.sampler.momentum = deepcopy(-momentum_start)           # momentum has sign flipped
            
        del model_start, momentum_start, p1, p2
        self.sampler.model.update_grad()
        
        
    def feed(self, t) :
        self.chain[t+1] = self.sampler.model.get_params()
            
    def sample(self) :
        self.feed(0)
        start_time = time()
        for t in range(self.Nsteps) :         
            self.MH_step()
            self.feed(t)
            if ((t+1) % (int(self.Nsteps/10)) == 0) or (t+1) == self.Nsteps :
                print("iter %6d/%d after %.2f min | accept_rate %.3f | MSE loss %.3f" % (
                      t+1, self.Nsteps, (time() - start_time)/60, float(self.n_accepted) / float(t+1), 
                      nn.MSELoss()(self.sampler.model.nn_model(self.sampler.model.x), self.sampler.model.y)))
                
    def ESS(self) :
        return self.Nsteps/np.asarray([gewer_estimate_IAT(self.chain[:,i].numpy()) for i in range(np.shape(self.chain)[1])])
    
    def plot(self) :
        ypred_final = self.sampler.model.nn_model(self.sampler.model.x)
        plt.plot(list(ypred_final[:,0]), list(ypred_final[:,1]), 'o', markersize=2)
        plt.grid(True)
        
        
    
#################################################################################################
#----------------------------------------- convergence -----------------------------------------#
#################################################################################################    
 
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

"""
ESS using Geyer IAT:
"""

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
             

def init_normal(nn_model) :
    for layer in nn_model :
        if type(layer) == nn.Linear :
            nn.init.normal_(layer.weight)