# Reference for HMC: Figure 2 of https://arxiv.org/pdf/1206.1901.pdf

import numpy as np, torch, numpy.random as npr, torch.nn as nn, copy, timeit
from torch.distributions.bernoulli import Bernoulli 
from scipy import signal as sp
from time import time

np.seterr(all='raise')

#def class PyTorchModel(object):

class model(object):
    def __init__(self, x, y, prior_sigma, error_sigma):
        # self.radius is an instance variable
        self.x = x
        self.y = y
        self.prior_sigma = prior_sigma
        self.error_sigma = error_sigma
        
    

def get_shapes(nn_model) :
    shapes = []
    for param in nn_model.parameters() :
        shapes.append(param.shape)
    return shapes

def update_grads(nn_model, model) :
    for param in nn_model.parameters() :
        if not (param.grad is None) :
            param.grad.data.zero_()
    loss = nn.MSELoss()(nn_model(x), y)
    loss.backward(retain_graph=True)

def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight)
        
        
        
def n_params(nn_model) :
    return sum(p.numel() for p in nn_model.parameters())

def extract_params(nn_model) :
    return torch.cat([param.view(-1) for param in nn_model.parameters()]).data

def extract_grads(nn_model) :
    return torch.cat([param.grad.view(-1) for param in nn_model.parameters()])

def set_params(param, nn_model, data) :
    assert len(param) == n_params(nn_model), "Number of parameters do not match"
    shapes = get_shapes(nn_model)
    counter = 0
    for i, parameter in enumerate(nn_model.parameters()) :
        parameter.data = param[counter:(counter+np.prod(shapes[i]))].reshape(shapes[i])
        counter += np.prod(shapes[i])  
    update_grads(nn_model, data)   
        
        
def generate_momentum(nn_model) :
    return torch.randn(n_params(nn_model))

def eval_kinetic_energy(mom, M) :
    return (mom**2/torch.diagonal(M)).sum().data/2

def eval_potential_energy(data, nn_model, prior_sigma, error_sigma) :
    N, k = np.shape(y)
    #from likelihood:
    pot_energy = N*nn.MSELoss()(nn_model(x), y).data/(2*error_sigma**2)
    # from prior:
    for param in nn_model.parameters() :
        pot_energy += (param**2).sum().data/(2*prior_sigma**2)
    
    return pot_energy.detach()

def update_mom(data, mom, nn_model, delta_leapfrog, prior_sigma, error_sigma) :
    N, k = np.shape(y)
    param, param_grad = extract_params(nn_model), extract_grads(nn_model)
    log_ll_grad = N*param_grad/(2*error_sigma**2)
    log_pr_grad = param/prior_sigma**2
    mom_change = -delta_leapfrog*(torch.add(log_ll_grad,log_pr_grad))
    return mom + mom_change  
        
def update_pos(data, mom, nn_model, delta_leapfrog, prior_sigma, error_sigma, M) :
    pos_change = delta_leapfrog*mom/torch.diag(M)
    param = extract_params(nn_model) + pos_change
    set_params(param, nn_model, data)
    # update gradients based on new parameters (ie, new positions):
    update_grads(nn_model, data)

def leapfrog(data, nn_model, n_leapfrog, delta_leapfrog, prior_sigma, error_sigma, M) :
    
    update_grads(nn_model, data)
    
    mom = generate_momentum(nn_model)  # Generate momentum variables
    current_mom = copy.deepcopy(mom) # keep copy of initial momentum 
    
    # half step for momentum at beginning
    update_mom(data, mom, nn_model, delta_leapfrog/2, prior_sigma, error_sigma)        
    
    # leapfrog steps:
    for l in range(n_leapfrog) :
        # Full step for position
        update_pos(data, mom, nn_model, delta_leapfrog, prior_sigma, error_sigma, M)
        
        # full step for momentum, except at end 
        if l < n_leapfrog-1 :
            update_mom(data, mom, nn_model, delta_leapfrog, prior_sigma, error_sigma)
                
    # half step for momentum at end :
    update_mom(data, mom, nn_model, delta_leapfrog/2, prior_sigma, error_sigma)
        
    # Negate momentum at end to make proposal symmetric
    mom.mul_(-1)

    return mom, current_mom, nn_model

def HMC_1step(data, nn_model, n_leapfrog, delta_leapfrog, prior_sigma, error_sigma, M) :
    
    
    
    current_nn_model = copy.deepcopy(nn_model)
    proposed_mom, current_mom, proposed_nn_model = leapfrog(data,
                                                            nn_model, 
                                                            n_leapfrog, 
                                                            delta_leapfrog, 
                                                            prior_sigma, 
                                                            error_sigma, 
                                                            M)
    
    # Evaluate potential and kinetic energies at start and end 
    current_K = eval_kinetic_energy(current_mom, M)
    proposed_K = eval_kinetic_energy(proposed_mom, M)
    current_U = eval_potential_energy(data, current_nn_model, prior_sigma, error_sigma)
    proposed_U = eval_potential_energy(data, proposed_nn_model, prior_sigma, error_sigma)
    
    # Accept/reject
    
    delta = current_U + current_K - proposed_U - proposed_K
    if delta > 10 :
        accepted = True 
    elif delta < -10 :
        accepted = False 
    else :
        accepted = npr.rand() < np.exp(current_U + current_K - proposed_U - proposed_K)     
        
    if accepted :
        return proposed_nn_model, 1
    else :
        return current_nn_model, 0
    
    
def run_HMC(data, T, nn_model, n_leapfrog, delta_leapfrog, prior_sigma, error_sigma, M) :
    
    nn_model.apply(init_normal)
    update_grads(nn_model, data)
    n_accept = 0
    
    mu, m2 = extract_params(nn_model), torch.ones(n_params(nn_model))
    
    chain = torch.zeros((T+1, n_params(nn_model)))
    chain[0] = extract_params(nn_model)
    
    start_time = time()
    for t in range(T) :         
        #M = torch.diag(m2 - mu**2)
        nn_model, accept = HMC_1step(data, 
                                     nn_model, 
                                     n_leapfrog, 
                                     delta_leapfrog,  
                                     prior_sigma,
                                     error_sigma,
                                     M)
        n_accept += accept
        update_grads(nn_model, data)
        chain[t+1] = extract_params(nn_model)
        if n_accept <= 0.4*t :
            delta_leapfrog /= 1.5
        elif n_accept >= 0.8*t :
            delta_leapfrog *= 1.5
            
        mu = (t*mu + chain[t+1])/(t+1)
        m2 = (t*m2 + chain[t+1]**2)/(t+1)
        
        if ((t+1) % (int(T/10)) == 0) or (t+1) == T :
            accept_rate = float(n_accept) / float(t+1)
            print("iter %6d/%d after %7.1f sec | accept_rate %.3f" % (
                t+1, T, time() - start_time, accept_rate))
            
    return chain

    
 
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
             





