# Reference for HMC: Figure 2 of https://arxiv.org/pdf/1206.1901.pdf

import numpy as np, torch, numpy.random as npr, torch.nn as nn, copy, timeit
from torch.distributions.bernoulli import Bernoulli 
from scipy import signal as sp

def get_shapes(nn_model) :
    shapes = []
    for param in nn_model.parameters() :
        shapes.append(param.shape)
    return shapes

def update_grads(nn_model, x, y) :
    for param in nn_model.parameters() :
        if not (param.grad is None) :
            param.grad.data.zero_()
    loss = nn.MSELoss()(nn_model(x), y)
    loss.backward(retain_graph=True)

def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight)
        
def generate_momentum(shapes) :
    mom = []
    for shape in shapes :
        mom.append(torch.randn(shape))
    return mom

def eval_kinetic_energy(mom_list) :
    return sum([(mom**2).sum().data for mom in mom_list])

def leapfrog(nn_model, n_leapfrog, delta_leapfrog, shapes, x, y) :
    
    update_grads(nn_model, x, y)
    
    mom = generate_momentum(shapes)  # Generate momentum variables
    current_mom = copy.deepcopy(mom) # keep copy of initial momentum 
    
    # half step for momentum at beginning
    for (i, param) in enumerate(nn_model.parameters()) :
        mom_change = -delta_leapfrog/2*(torch.add(param.grad,param))
        mom[i].data.add_(mom_change)          
    
    # leapfrog steps:
    for l in range(n_leapfrog) :
        # Full step for position
        for (i, param) in enumerate(nn_model.parameters()) :
            #param.data.add_(delta_leapfrog*torch.mul(M_inv[i],mom[i]))
            pos_change = delta_leapfrog*mom[i]
            param.data.add_(pos_change)
        
        # update gradients based on new parameters (ie, new positions):
        update_grads(nn_model, x, y)
        
        # full step for momentum, except at end 
        if l != (n_leapfrog-1) :
            for (i, param) in enumerate(nn_model.parameters()) :
                mom_change = -delta_leapfrog*(torch.add(param.grad,param))
                mom[i].data.add_(mom_change)                # N(0,1) prior for params (?)
                
    # half step for momentum at end :
    for (i, param) in enumerate(nn_model.parameters()) :
        mom_change = -delta_leapfrog/2*(torch.add(param.grad,param))
        mom[i].data.add_(mom_change)           # N(0,1) prior for params (?)
        # Negate momentum at end to make proposal symmetric
        mom[i].mul_(-1)

    return mom, current_mom, nn_model

def HMC_1step(nn_model, n_leapfrog, delta_leapfrog, shapes, x, y, criterion, sigma) :
    
    current_nn_model = copy.deepcopy(nn_model)
    proposed_mom, current_mom, proposed_nn_model = leapfrog(nn_model, n_leapfrog, delta_leapfrog, shapes, x, y)
    
    # Evaluate potential and kinetic energies at start and end 
    current_K = eval_kinetic_energy(current_mom)
    proposed_K = eval_kinetic_energy(proposed_mom)
    current_U = nn.MSELoss()(current_nn_model(x), y).data/2
    proposed_U = nn.MSELoss()(proposed_nn_model(x), y).data/2
    
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
    Y = ( X - np.mean(X) ) / np.std(X)
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
    T = chain[0].size()[0]
    for (i, shape) in enumerate(shapes) :
        for j in range(shape[0]) :
            if len(list(shape)) == 1 :
                ESS.append(int(T/gewer_estimate_IAT(chain[i][:,j].numpy())))
                if to_print :
                    print("ESS: %s/%s" % (int(T/gewer_estimate_IAT(chain[i][:,j].numpy())), T))
            else :
                for k in range(shape[1]) :
                    ESS.append(int(T/gewer_estimate_IAT(chain[i][:,j,k].numpy())))
                    if to_print :
                        print("ESS: %s/%s" % (int(T/gewer_estimate_IAT(chain[i][:,j,k].numpy())), T))
                    
    return np.asarray(ESS)
