# Reference for HMC: Figure 2 of https://arxiv.org/pdf/1206.1901.pdf

import numpy as np, torch, numpy.random as npr, torch.nn as nn, timeit
from copy import deepcopy
from torch.distributions.bernoulli import Bernoulli 
from scipy import signal as sp
from time import time
import abc


np.seterr(all='raise')

#def class PyTorchModel(object):

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

    def init_normal(self):
        set_params(torch.randn(self.n_params()), self)
        
        #for layer in self.nn_model :
        #    if type(layer) == nn.Linear :
        #        nn.init.normal_(layer.weight)
        #self.x.data = torch.randn(np.shape(self.x))
        #self.update_grad()

    def n_params(self) :
        return (sum(p.numel() for p in self.nn_model.parameters()) + np.prod(np.shape(self.x))).item()

    def extract_params(self) :
        return torch.cat([torch.cat([param.view(-1) for param in self.nn_model.parameters()]), self.x.view(-1)]).data

    def grad(self) :
        return torch.cat([torch.cat([param.grad.view(-1) for param in self.nn_model.parameters()]), self.x.grad.view(-1)]).data
    
    def negate_params(self) :
        set_params(-extract_params(self).data, self.model)
        
    def set_params(self, param) :
        assert len(param) == self.n_params(), "Number of parameters do not match"
        shapes = self.shape()
        counter = 0
        for i, parameter in enumerate(self.nn_model.parameters()) :
            parameter.data = param[counter:(counter+np.prod(shapes[i]))].reshape(shapes[i])
            counter += np.prod(shapes[i])  
        self.x.data = param[counter::].data.reshape(np.shape(self.x)).clone()
        self.update_grad() 
        
    def potential_energy(self) :
        N = np.shape(self.y)[0]
        #from likelihood:
        pot_energy = N*nn.MSELoss()(self.nn_model(self.x), self.y).data/(2*self.error_sigma**2)
        # from prior:
        for param in self.nn_model.parameters() :  # for theta
            pot_energy += (param**2).sum().data/(2*self.prior_sigma**2)
        pot_energy += (self.x**2).sum().data/(2*self.prior_sigma**2) #for x
        return pot_energy.detach()
    
    def kinetic_energy(self, momentum, M) :
        return (momentum**2/torch.diagonal(self.M)).sum().data/2
    
    def total_energy(self, momentum, M) :
        return self.potential_energy() + self.kinetic_energy(momentum, M) 
    
    def generate_momentum(self) :
        return torch.randn(self.n_params())
    
    
#################################################################################################
#--------------------------------------------- HMC ---------------------------------------------#
#################################################################################################


class Sampler(object) :
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, model):

        self.model = model
    
    def step(self):
        raise NotImplementedError()
     
    def run(self, initial_values=None):
        self.initialise(initial_values)
        self.feed(0)
        for t in range(self.Nsteps):
            self.step()
            self.outputsheduler.feed(t+1)
        
        
class MH(Sampler) :
    __metaclass__ = abc.ABCMeta
    
    
class KineticMH(MH) :
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, model, stepsize):
        super(KineticMH,self).__init__(model)
        self.stepsize =  stepsize
        self.mom =  (model)
        
    def update_momentum(self, delta=1) :
        N = np.shape(self.model.y)[0]
        param, param_grad = self.model.extract_params(), self.model.grad()
        log_ll_grad = N*param_grad/(2*self.model.error_sigma**2)
        log_pr_grad = param/self.model.prior_sigma**2
        mom_change = -self.stepsize*delta*(torch.add(log_ll_grad,log_pr_grad))
        self.mom += mom_change  
        
    def update_position(self) :
        pos_change = self.stepsize*self.E.mom/torch.diag(self.E.M)
        param = self.E.model.extract_params() + pos_change
        set_params(param, self.E.model)
        # update gradients based on new parameters (ie, new positions):
        self.E.model.update_grad()
        
class HMC(MH) :
    def __init__(self, model, n_leapfrog, stepsize, T, chain=0, minlf=50, maxlf=500, minstepsize=1e-3):
        # self.radius is an instance variable
        self.model = model
        self.n_leapfrog = n_leapfrog
        self.stepsize = stepsize
        self.T = T
        self.chain = chain
        self.minlf = minlf
        self.maxlf = maxlf
        self.minstepsize = minstepsize
        

    def update_momentum(self, delta=1) :
        N = np.shape(self.model.y)[0]
        param, param_grad = self.model.extract_params(), self.model.grad()
        log_ll_grad = N*param_grad/(2*self.model.error_sigma**2)
        log_pr_grad = param/self.model.prior_sigma**2
        mom_change = -self.stepsize*delta*(torch.add(log_ll_grad,log_pr_grad))
        self.mom += mom_change  
        
    def update_position(self) :
        pos_change = self.stepsize*self.E.mom/torch.diag(self.E.M)
        param = self.E.model.extract_params() + pos_change
        set_params(param, self.E.model)
        # update gradients based on new parameters (ie, new positions):
        self.E.model.update_grad()
        

    def step(self) :
        self.E.model.update_grad()
        self.E.generate_momentum()  # Generate momentum variables
        U_start, K_start = self.E.potential_energy(), self.E.kinetic_energy() 
        # half step for momentum at beginning
        self.update_momentum(1/2)        
        # leapfrog steps:
        for l in range(self.n_leapfrog) :
            # Full step for position
            self.update_position()
            # full step for momentum, except at end 
            if l < self.n_leapfrog-1 :
                self.update_momentum()
        # half step for momentum at end :
        self.update_momentum(1/2)
        # Negate momentum at end to make proposal symmetric
        self.E.mom.mul_(-1)
        U_end, K_end = self.E.potential_energy(), self.E.kinetic_energy() 
        
        return self, U_start, K_start, U_end, K_end

    def HMC_1step(self) :

        current_nnmodel = deepcopy(self.E.model.nn_model)
        self, current_U, current_K, proposed_U, proposed_K = self.leapfrog() 

        # Accept/reject
        delta = current_U + current_K - proposed_U - proposed_K
        if delta > 10 :
            accepted = True 
        elif delta < -10 :
            accepted = False 
        else :
            accepted = npr.rand() < np.exp(current_U + current_K - proposed_U - proposed_K)     

        if accepted :
            return self.E.model.nn_model, 1
        else :
            return current_nnmodel, 0
        
    def limitleapfrog(self) :
        if self.n_leapfrog > self.maxlf :
            self.n_leapfrog = self.maxlf 
        elif self.n_leapfrog < self.minlf :
            self.n_leapfrog = self.minlf
        if self.stepsize < self.minstepsize :
            self.stepsize = self.minstepsize
    
    
    def sample(self) :

        self.E.model.init_normal()
        self.E.model.update_grad()
        n_accept = 0

        self.chain = torch.zeros((self.T+1, self.E.model.n_params()))
        self.chain[0] = self.E.model.extract_params()

        start_time = time()
        for t in range(self.T) :         
            #M = torch.diag(m2 - mu**2)
            self.E.model.nn_model, accept = self.HMC_1step()
            n_accept += accept
            self.E.model.update_grad()
            self.chain[t+1] = self.E.model.extract_params()
            if n_accept <= 0.2*t :
                self.n_leapfrog = int(self.n_leapfrog/1.5)
                self.stepsize /= 1.05
            elif n_accept >= 0.8*t :
                self.n_leapfrog *= int(self.n_leapfrog*1.5)
                self.stepsize *= 1.05
            self.limitleapfrog()

            if ((t+1) % (int(self.T/10)) == 0) or (t+1) == self.T :
                accept_rate = float(n_accept) / float(t+1)
                print("iter %6d/%d after %.2f min | accept_rate %.3f | MSE loss %.3f | stepsize %.3f | nleapfrog %i" % (
                      t+1, self.T, (time() - start_time)/60, accept_rate, 
                      nn.MSELoss()(self.E.model.nn_model(self.E.model.x), self.E.model.y), self.stepsize, self.n_leapfrog))
                
                
    def ESS(self) :
        return self.T/[gewer_estimate_IAT(self.chain[i,:].numpy()) for i in range(np.shape(self.chain)[1])]
     
        
#################################################################################################
#-------------------------------------------- BAOAB --------------------------------------------#
#################################################################################################
        
        
class BAOAB(object) :
    def __init__(self, E, T, stepsize, beta, gamma, alpha=0, Rn=1, mom=1, chain=0):
        # self.radius is an instance variable
        self.T = T
        self.E = E
        self.stepsize = stepsize
        self.beta = beta 
        self.gamma = gamma
        self.alpha = np.exp(-self.gamma*self.stepsize)
        self.chain = chain
        
    def propose(self) :
        pos, grad = self.E.model.extract_params(), self.E.model.grad()
        self.E.mom -= self.stepsize/2*grad
        p1 = deepcopy(self.E.mom.data)
        pos += self.stepsize/2*self.E.mom
        self.Rn = torch.randn(self.E.model.n_params())
        self.E.mom = self.alpha*self.E.mom + np.sqrt((1-self.alpha**2)/self.beta)*self.Rn
        p2 = deepcopy(self.E.mom.data)
        pos += self.stepsize/2*self.E.mom
        set_params(pos, self.E.model)
        grad = self.E.model.grad()
        self.E.mom -= self.stepsize/2*grad
        return self, p1, p2
        
    def g(self, x) :
        return torch.exp(-self.beta/(2*(1-self.alpha**2)*torch.norm(x)))
        
    def MH_step(self) :
        H_start = self.E.potential_energy() + self.E.kinetic_energy() 
        start_model, start_mom = deepcopy(self.E.model), deepcopy(self.E.mom)
        self, p1, p2 = self.propose()
        H_proposed = self.E.potential_energy() + self.E.kinetic_energy() 
        accept_ratio = torch.exp(-self.beta*(H_proposed-H_start))*self.g(self.alpha*p2-p1)/self.g(self.Rn)
        u = torch.rand(1)
        if u < accept_ratio :
            # accept 
            xx=1
        else :
            set_params(start_model.extract_params(), self.E.model)
            self.E.mom *= -1
            
    def sample(self) :
        self.E.model.init_normal()
        self.E.model.update_grad()
        n_accept = 0

        self.chain = torch.zeros((self.T+1, self.E.model.n_params()))
        self.chain[0] = self.E.model.extract_params()

        start_time = time()
        for t in range(self.T) :         
            self.MH_step()
            self.chain[t+1] = self.E.model.extract_params()
            
            if ((t+1) % (int(self.T/10)) == 0) or (t+1) == self.T :
                accept_rate = float(n_accept) / float(t+1)
                print("iter %6d/%d after %.2f min | accept_rate %.3f | MSE loss %.3f" % (
                      t+1, self.T, (time() - start_time)/60, accept_rate, 
                      nn.MSELoss()(self.E.model.nn_model(self.E.model.x), self.E.model.y)))
                
    def ESS(self) :
        return self.T/[gewer_estimate_IAT(self.chain[i,:].numpy()) for i in range(np.shape(self.chain)[1])]
        
        
        


    
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



