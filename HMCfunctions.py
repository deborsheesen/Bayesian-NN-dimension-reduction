# Reference for HMC: Figure 2 of https://arxiv.org/pdf/1206.1901.pdf

import numpy as np, torch, numpy.random as npr, torch.nn as nn, copy, timeit
from torch.distributions.bernoulli import Bernoulli 
from scipy import signal as sp
from time import time

np.seterr(all='raise')

#def class PyTorchModel(object):

class model(object):
    def __init__(self, x, y, prior_sigma, error_sigma, nn_model):
        # self.radius is an instance variable
        self.x = x
        self.y = y
        self.prior_sigma = prior_sigma
        self.error_sigma = error_sigma
        self.nn_model = nn_model
        
    def get_shapes(self) :
        shapes = []
        for param in self.nn_model.parameters() :
            shapes.append(param.shape)
        return shapes
    
    def update_grads(self) :
        for param in self.nn_model.parameters() :
            if not (param.grad is None) :
                param.grad.data.zero_()
        loss = nn.MSELoss()(self.nn_model(self.x), self.y)
        loss.backward(retain_graph=True)

    def init_normal(self):
        if type(self.nn_model) == nn.Linear:
            self.nn_model.init.normal_(self.nn_model.weight)

    def n_params(self) :
        return sum(p.numel() for p in self.nn_model.parameters())

    def extract_params(self) :
        return torch.cat([param.view(-1) for param in self.nn_model.parameters()]).data

    def extract_grads(self) :
        return torch.cat([param.grad.view(-1) for param in self.nn_model.parameters()])
    
    
    

def set_params(param, model) :
    assert len(param) == model.n_params(), "Number of parameters do not match"
    shapes = model.get_shapes()
    counter = 0
    for i, parameter in enumerate(model.nn_model.parameters()) :
        parameter.data = param[counter:(counter+np.prod(shapes[i]))].reshape(shapes[i])
        counter += np.prod(shapes[i])  
    model.update_grads()  
        
        
        
class HMC(object) :
    def __init__(self, model, M, n_leapfrog, delta_leapfrog, T, mom=1, chain=0):
        # self.radius is an instance variable
        self.model = model
        self.M = M
        self.n_leapfrog = n_leapfrog
        self.delta_leapfrog = delta_leapfrog
        self.T = T
        self.mom = mom
        self.chain = chain
        
        
    def generate_momentum(self) :
        return torch.randn(self.model.n_params())

    def eval_kinetic_energy(self) :
        return (self.mom**2/torch.diagonal(self.M)).sum().data/2

    def eval_potential_energy(self) :
        N, k = np.shape(self.model.y)
        #from likelihood:
        pot_energy = N*nn.MSELoss()(self.model.nn_model(self.model.x), self.model.y).data/(2*self.model.error_sigma**2)
        # from prior:
        for param in self.model.nn_model.parameters() :
            pot_energy += (param**2).sum().data/(2*self.model.prior_sigma**2)

        return pot_energy.detach()

    def update_mom(self) :
        N, k = np.shape(self.model.y)
        param, param_grad = self.model.extract_params(), self.model.extract_grads()
        log_ll_grad = N*param_grad/(2*self.model.error_sigma**2)
        log_pr_grad = param/self.model.prior_sigma**2
        mom_change = -self.delta_leapfrog*(torch.add(log_ll_grad,log_pr_grad))
        return self.mom + mom_change  
        
    def update_pos(self) :
        pos_change = self.delta_leapfrog*self.mom/torch.diag(self.M)
        param = self.model.extract_params() + pos_change
        set_params(param, self.model)
        # update gradients based on new parameters (ie, new positions):
        self.model.update_grads()
        

    def leapfrog(self) :
        self.model.update_grads()
        self.mom = self.generate_momentum()  # Generate momentum variables
        U_start, K_start = self.eval_potential_energy(), self.eval_kinetic_energy() 
        # half step for momentum at beginning
        self.update_mom()        
        # leapfrog steps:
        for l in range(self.n_leapfrog) :
            # Full step for position
            self.update_pos()
            # full step for momentum, except at end 
            if l < self.n_leapfrog-1 :
                self.update_mom()
        # half step for momentum at end :
        self.update_mom()
        # Negate momentum at end to make proposal symmetric
        self.mom.mul_(-1)
        U_end, K_end = self.eval_potential_energy(), self.eval_kinetic_energy() 
        
        return self, U_start, K_start, U_end, K_end

    def HMC_1step(self) :

        current_nnmodel = copy.deepcopy(self.model.nn_model)
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
            return self.model.nn_model, 1
        else :
            return current_nnmodel, 0
    
    
    def run_HMC(self) :

        self.model.init_normal()
        self.model.update_grads()
        n_accept = 0

        self.chain = torch.zeros((self.T+1, self.model.n_params()))
        self.chain[0] = self.model.extract_params()

        start_time = time()
        for t in range(self.T) :         
            #M = torch.diag(m2 - mu**2)
            self.model.nn_model, accept = self.HMC_1step()
            n_accept += accept
            self.model.update_grads()
            self.chain[t+1] = self.model.extract_params()
            if n_accept <= 0.4*t :
                self.delta_leapfrog /= 1.5
            elif n_accept >= 0.8*t :
                self.delta_leapfrog *= 1.5

            if ((t+1) % (int(self.T/10)) == 0) or (t+1) == self.T :
                accept_rate = float(n_accept) / float(t+1)
                print("iter %6d/%d after %7.1f sec | accept_rate %.3f | MSE loss %.3f" % (
                      t+1, self.T, time() - start_time, accept_rate, 
                      nn.MSELoss()(self.model.nn_model(self.model.x), self.model.y)))

    
 
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
             





