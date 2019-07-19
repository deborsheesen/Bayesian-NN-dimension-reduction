# Reference for HMC: Figure 2 of https://arxiv.org/pdf/1206.1901.pdf

import numpy as np, torch, numpy.random as npr, torch.nn as nn, copy, timeit
from scipy.stats import invgamma
import torch.nn.functional as F, torch.autograd as autograd, torch.optim as optim
from scipy import signal as sp
from time import time
import matplotlib.pyplot as plt
from copy import deepcopy
import abc
from torch.autograd import Variable

from pylab import plot, show, legend

np.seterr(all='raise')


#################################################################################################
#-------------------------------------------- Model --------------------------------------------#
#################################################################################################

class model(object):
    def __init__(self, x, y, prior_sigma2, error_sigma2, nn_model):
        # self.radius is an instance variable
        self.x = x
        self.y = y
        self.prior_sigma2 = prior_sigma2
        self.error_sigma2 = error_sigma2
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
        loss = nn.MSELoss()(self.nn_model(self.x), self.y)
        loss.backward(retain_graph=True)

    def init_normal(self) :
        self.set_params(torch.randn(self.n_params()))
        self.update_grad()

    def n_params(self) :
        return sum(p.numel() for p in self.nn_model.parameters())

    def get_params(self) :
        return torch.cat([param.view(-1) for param in self.nn_model.parameters()])

    def grad(self) :
        return torch.cat([param.grad.view(-1) for param in self.nn_model.parameters()])
    
    def set_params(self, param) :
        assert len(param) == self.n_params(), "Number of parameters do not match"
        shapes = self.shape()
        counter = 0
        for i, parameter in enumerate(self.nn_model.parameters()) :
            parameter.data = param[counter:(counter+np.prod(shapes[i]))].reshape(shapes[i])
            counter += np.prod(shapes[i])  
        self.update_grad() 
        
        
class nn_sampler(object):
    __metaclass__ = abc.ABCMeta
    '''
    Base class for samplers
    '''
    
    def __init__(self, model, Nsteps, M_precond) :
        self.model = model
        self.Nsteps = Nsteps
        self.chain = torch.zeros(self.Nsteps+1, self.model.n_params()+1)
        self.chain[0,:self.model.n_params()] = self.model.get_params()
        self.chain[0,-1] = self.model.error_sigma2
        self.M_precond = M_precond
        if self.M_precond == None :
            self.M_precond = torch.eye(self.model.n_params())
            
    def plot(self) :
        ypred_final = self.model.nn_model(self.model.x)
        plt.plot(list(ypred_final[:,0]), list(ypred_final[:,1]), 'o', markersize=2)
        plt.grid(True)
            
    def ESS(self) :
        return self.Nsteps/np.asarray([gewer_estimate_IAT(self.chain[:,i].numpy()) for i in range(np.shape(self.chain)[1])])  
    
    def log_target(self) :
        N = np.shape(self.model.y)[0]
        param = self.model.get_params()
        log_ll = -N/(2*self.model.error_sigma2)*nn.MSELoss()(self.model.nn_model(self.model.x), self.model.y)
        log_pr = -1/(2*self.model.prior_sigma2)*sum(param**2)
        return log_ll + log_pr
            
        
        
class Gibbs_MH(nn_sampler) :
    __metaclass__ = abc.ABCMeta
    '''
    Base class for Gibbs_MH samplers
    '''    
    def __init__(self, model, Nsteps, M_precond) :
        nn_sampler.__init__(self, model, Nsteps, M_precond)
        self.n_accepted = 0
        self.accepted = []
        self.ysamples = torch.zeros(int(Nsteps/10), model.y.shape[0], model.y.shape[1])
        self.loss = np.zeros(Nsteps)
        self.alpha = 5
        self.beta = 1
        
    def propose(self) :
        raise NotImplementedError()
        
    def feed(self, t) :
        self.chain[t+1,:self.model.n_params()] = self.model.get_params()
        self.chain[t+1,-1] = self.model.error_sigma2
        
    def update_sigma(self) :
        N = np.shape(self.model.y)[0]
        a = self.alpha + N/2
        b = self.beta + N/2*nn.MSELoss()(self.model.nn_model(self.model.x), self.model.y).detach().numpy()
        self.model.error_sigma2 = invgamma.rvs(a=a,scale=b)
        
    def run(self) :
        self.feed(0)
        counter = 0
        start_time = time()
        for t in range(self.Nsteps) :  
            self.update_sigma()
            self.step()
            self.feed(t)
            
            if ((t+1) % (int(self.Nsteps/10)) == 0) :
                print("iter %d/%d after %.1f min | accept_rate %.1f percent | MSE loss %.3f" % (
                      t+1, self.Nsteps, (time() - start_time)/60, 100*float(sum(self.accepted)) / float(t+1), 
                      nn.MSELoss()(self.model.nn_model(self.model.x), self.model.y)))
            if (t+1)%10 == 0 :
                self.ysamples[counter] = self.model.nn_model(self.model.x)
                counter += 1
            self.loss[t] = nn.MSELoss()(self.model.nn_model(self.model.x), self.model.y).detach().numpy()
        
class kineticMH(Gibbs_MH) :
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, model, Nsteps, M_precond, stepsize) :
        
        Gibbs_MH.__init__(self, model, Nsteps, M_precond)
        self.momentum = torch.randn(self.model.n_params())
        self.position = self.model.get_params()
        self.stepsize = stepsize
        
    def generate_momentum(self) :
        self.momentum = torch.randn(self.model.n_params())    
        
    def potential_energy(self) :
        N = np.shape(self.model.y)[0]
        #from likelihood:
        pot_energy = N*nn.MSELoss()(self.model.nn_model(self.model.x), self.model.y).data/(2*self.model.error_sigma2)
        # from prior:
        pot_energy += sum(self.model.get_params()**2)/(2*self.model.prior_sigma2)
        return pot_energy.detach()
    
    def kinetic_energy(self) :
        return (self.momentum**2/torch.diagonal(self.M_precond)).sum().data/2
    
    def total_energy(self) :
        return self.potential_energy() + self.kinetic_energy()     
        
    
#################################################################################################
#--------------------------------------------- HMC ---------------------------------------------#
#################################################################################################   


class HMC(kineticMH) :
    
    def __init__(self, model, Nsteps, stepsize, n_leapfrog, M_precond=None) :
        kineticMH.__init__(self, model, Nsteps, M_precond, stepsize) 
        self.n_leapfrog = n_leapfrog
    
    def update_momentum(self, delta=1) :
        N = np.shape(self.model.y)[0]
        param, param_grad = self.model.get_params(), self.model.grad()
        log_ll_grad = N/(2*self.model.error_sigma2)*param_grad
        log_pr_grad = param/self.model.prior_sigma2
        momentum_change = -delta*self.stepsize*(torch.add(log_ll_grad,log_pr_grad))
        self.momentum += momentum_change  
        
    def update_position(self, delta=1) :
        pos_change = delta*self.stepsize*self.momentum/torch.diag(self.M_precond)
        param = self.model.get_params() + pos_change
        self.model.set_params(param)
        # update gradients based on new parameters (ie, new positions):
        self.model.update_grad()
        
    def propose(self) :   # using leapfrog
        self.model.update_grad()
        self.generate_momentum()  
        energy_start = self.total_energy()
        
        tot_energy = np.zeros(self.n_leapfrog+1)
        tot_energy[0] = self.total_energy()
        
        # half step for momentum at beginning
        self.update_momentum(1/2)        
        # leapfrog steps:
        for l in range(self.n_leapfrog) :
            # Full step for position
            self.update_position(1)
            # full step for momentum, except at end 
            if l < self.n_leapfrog-1 :
                self.update_momentum(1)
            tot_energy[l+1] = self.total_energy()
        # half step for momentum at end :
        self.update_momentum(1/2)
        # Negate momentum at end to make proposal symmetric
        self.momentum.mul_(-1)
        energy_end = self.total_energy()
        
        return self, energy_start, energy_end#, tot_energy

    def step(self) :

        start_nn_model = deepcopy(self.model.nn_model)
        self, energy_start, energy_end = self.propose()

        # Accept/reject
        energy_diff = energy_start - energy_end
        if energy_diff > 200 :
            accepted = True 
        elif energy_diff < -200 :
            accepted = False 
        else :
            accepted = (torch.rand(1) < torch.exp(energy_diff))
        
        self.n_accepted += accepted
        if not accepted :
            self.model.nn_model = deepcopy(start_nn_model)
            self.model.update_grad()
        self.accepted.append(int(accepted))
       
        del start_nn_model
                
#################################################################################################
#-------------------------------------------- MALA ---------------------------------------------#
#################################################################################################  
        
class MALA(kineticMH) :
    
    def __init__(self, model, Nsteps, stepsize, M_precond=0) :
        kineticMH.__init__(self, model, Nsteps, M_precond, stepsize) 
        
    def propose(self) :
        N = np.shape(self.model.y)[0]
        param, param_grad = self.model.get_params(), self.model.grad()
        log_ll_grad = -N/(2*self.model.error_sigma2)*param_grad
        log_pr_grad = -param/self.model.prior_sigma2
        log_target_grad = torch.add(log_ll_grad,log_pr_grad)
        self.generate_momentum()
        param += torch.add(self.stepsize**2*log_target_grad/2,self.stepsize*self.momentum) 
        self.model.set_params(param)
        self.model.update_grad()
        return param
    
    def step(self) :

        log_target_initial = self.log_target()
        initial, grad_initial = self.model.get_params(), self.model.grad()
        start_nn_model = deepcopy(self.model.nn_model)
        
        proposal = self.propose()
        grad_proposal = self.model.grad()
        log_target_proposal = self.log_target()
        
        log_move = -1/(2*self.stepsize**2)*sum((proposal-initial-self.stepsize**2*grad_initial/2)**2)     #\theta_n to \theta_star  
        log_move_reverse = -1/(2*self.stepsize**2)*sum((initial-proposal-self.stepsize**2*grad_proposal/2)**2)   #\theta_star to \theta_n 
        
        log_acceptance_ratio = log_target_proposal-log_target_initial+log_move_reverse-log_move
        #print("Log acceptance ratio:", log_acceptance_ratio.detach().numpy())
        
        if log_acceptance_ratio > 200 :
            accepted = True 
        elif log_acceptance_ratio < -200 :
            accepted = False 
        else :
            accepted = (torch.rand(1) < torch.exp(log_acceptance_ratio))
        
        self.n_accepted += accepted
        if not accepted :
            self.model.nn_model = deepcopy(start_nn_model)
            self.model.update_grad()
        self.accepted.append(int(accepted))
        
        del start_nn_model
    
        
#################################################################################################
#-------------------------------------------- BAOAB --------------------------------------------#
#################################################################################################
        
        
class BAOAB(kineticMH) :
    
    
    def __init__(self, model, Nsteps, stepsize, beta, gamma, M_precond=0) :
        
        kineticMH.__init__(self, model, Nsteps, M_precond, stepsize) 
        self.beta = beta 
        self.gamma = gamma
        self.alpha = np.exp(-self.gamma*self.stepsize)
        self.Rn = torch.randn(self.model.n_params())
            
        
    def propose(self) :
        pos, grad = self.model.get_params(), self.model.grad()
        
        # half step for momentum:
        self.update_momentum(1/2)
        p1 = deepcopy(self.momentum.data)
        
        # half step for position:
        self.update_position(1/2)
        
        # update momentum with random variables:
        self.Rn = torch.randn(self.model.n_params())
        self.momentum = self.alpha*self.momentum + np.sqrt((1-self.alpha**2)/self.beta)*self.Rn
        p2 = deepcopy(self.momentum.data)
        
        # half step for position
        self.update_position(1/2)
        
        # half step for momentum:
        self.update_momentum(1/2)
        
        return self, p1, p2
        
    def g(self, x) :
        return torch.exp(-self.beta/(2*(1-self.alpha**2)*torch.norm(x)))
        
    def step(self) :
        energy_start = self.total_energy()
        model_start, momentum_start = deepcopy(self.model), deepcopy(self.momentum)
        self, p1, p2 = self.propose()
        energy_end = self.total_energy()
        accept_ratio = torch.exp(-self.beta*(energy_end-energy_start))*self.g(self.alpha*p2-p1)/self.g(self.Rn)
        #print(accept_ratio)
        u = torch.rand(1)
        
        accepted = (torch.rand(1) < accept_ratio)
        self.n_accepted += accepted
        
        if not accepted :                     # if reject
            self.model = deepcopy(model_start)   # set position to initial position
            self.momentum = deepcopy(-momentum_start)           # momentum has sign flipped
            
        del model_start, momentum_start, p1, p2
        self.model.update_grad()
        
        
    
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
            nn.init.normal_(layer.bias)
            
            
#################################################################################################
#--------------------------------------------- VAE ---------------------------------------------#
#################################################################################################  
            
class inout_model(object) : 
    def __init__(self, X_dim, h_dim, Z_dim, nn_encode, nn_decode):
        # self.radius is an instance variable
        self.Wxh = 0
        self.bxh = 0
        self.Whz_mu = 0
        self.bhz_mu = 0
        self.Whz_var = 0
        self.bhz_var = 0
        self.Wzh = 0
        self.bzh = 0
        self.Whx = 0
        self.bhx = 0
        self.X_dim = X_dim
        self.h_dim = h_dim
        self.Z_dim = Z_dim
        self.params = [self.Wxh, self.bxh, self.Whz_mu, self.bhz_mu, self.Whz_var, 
                       self.bhz_var, self.Wzh, self.bzh, self.Whx, self.bhx]
        self.nn_encode = nn_encode
        self.nn_decode = nn_decode
        
    def set_params(self) :
        self.params = [self.Wxh, self.bxh, self.Whz_mu, self.bhz_mu, self.Whz_var, 
                       self.bhz_var, self.Wzh, self.bzh, self.Whx, self.bhx]
        
    def xavier_init(self, size):
        in_dim = size[0]
        xavier_stddev = 1/np.sqrt(in_dim/2)
        return Variable(torch.randn(*size) * xavier_stddev, requires_grad=True)
    
    def initialise(self) :
        self.Wxh = self.xavier_init(size=[self.X_dim, self.h_dim])
        self.bxh = Variable(torch.zeros(self.h_dim), requires_grad=True)
        self.Whz_mu = self.xavier_init(size=[self.h_dim, self.Z_dim])
        self.bhz_mu = Variable(torch.zeros(self.Z_dim), requires_grad=True)
        self.Whz_var = self.xavier_init(size=[self.h_dim, self.Z_dim])
        self.bhz_var = Variable(torch.zeros(self.Z_dim), requires_grad=True)
        self.Wzh = self.xavier_init(size=[self.Z_dim, self.h_dim])
        self.bzh = Variable(torch.zeros(self.h_dim), requires_grad=True)
        self.Whx = self.xavier_init(size=[self.h_dim, self.X_dim])
        self.bhx = Variable(torch.zeros(self.X_dim), requires_grad=True)
        self.set_params()
        
    # ============================= Q(z|X); encoding ================================
    def Q(self, X):
        #h = F.tanh(X @ self.Wxh + self.bxh.repeat(X.size(0), 1))
        h = self.nn_encode(X)
        
        z_mu = h @ self.Whz_mu + self.bhz_mu.repeat(h.size(0), 1)
        z_var = h @ self.Whz_var + self.bhz_var.repeat(h.size(0), 1)
        return z_mu, z_var
    
    def sample_z(self, mu, log_var, mb_size):
        eps = Variable(torch.randn(mb_size, self.Z_dim))
        return mu + torch.exp(log_var/2) * eps
    
    # ============================= P(X|z); decoding ================================
    def P(self, z):
        #h = F.tanh(z @ self.Wzh + self.bzh.repeat(z.size(0), 1))
        #X = h @ self.Whx + self.bhx.repeat(h.size(0), 1)
        X = self.nn_decode(z)
        return X
    
    # =============================== Optimising ====================================
    def solve(self, loss, lr) :
        loss.backward(retain_graph=True)
        solver = optim.Adam(self.params, lr=lr)
        solver.step()
        self.set_params()
        
    def housekeeping(self) :
        for p in self.params:
            if p.grad is not None:
                data = p.grad.data
                p.grad = Variable(data.new().resize_as_(data).zero_())
                
                
                