import numpy as np, torch, numpy.random as npr, torch.nn as nn, copy, timeit
from torch.distributions.bernoulli import Bernoulli 

def get_shapes(nn_model) :
    shapes = []
    for param in nn_model.parameters() :
        shapes.append(param.shape)
    return shapes

def update_grads(nn_model, x, y) :
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

def eval_kinetic_energy(mom) :
    energy = 0
    for i in mom :
        energy += (i**2).sum().data
    return energy

def leapfrog(nn_model, n_leapfrog, delta_leapfrog, shapes, x, y) :
    
    update_grads(nn_model, x, y)
    
    mom = generate_momentum(shapes)  # Simulate momentum variables
    current_mom = copy.deepcopy(mom) # keep copy of initial momentum 
    
    # half step for momentum at beginning
    for (i, param) in enumerate(nn_model.parameters()) :
        mom[i].data.add_(-delta_leapfrog/2*param.grad)          # prior for params?
    
    # leapfrog steps:
    for l in range(n_leapfrog) :
        # Full step for position
        for (i, param) in enumerate(nn_model.parameters()) :
            #param.data.add_(delta_leapfrog*torch.mul(M_inv[i],mom[i]))
            param.data.add_(-delta_leapfrog*mom[i])   
        
        # update gradients based on new parameters (new positions):
        update_grads(nn_model, x, y)
        
        # full step for momentum, except at end 
        if l != (n_leapfrog-1) :
            for (i, param) in enumerate(nn_model.parameters()) :
                mom[i].data.add_(-delta_leapfrog*param.grad)    # prior for params?
                
    # half step for momentum at end :
    for (i, param) in enumerate(nn_model.parameters()) :
        mom[i].data.add_(-delta_leapfrog/2*param.grad)          # prior for params?
        # Negate momentum at end to make proposal symmetric
        mom[i].mul_(-1)

    return mom, current_mom, nn_model

def HMC_1step(nn_model, n_leapfrog, delta_leapfrog, shapes, x, y, criterion, sigma) :
    
    current_nn_model = copy.deepcopy(nn_model)
    proposed_mom, current_mom, proposed_nn_model = leapfrog(nn_model, n_leapfrog, delta_leapfrog, shapes, x, y)
    
    # Evaluate potential and kinetic energies at start and end 
    current_K = eval_kinetic_energy(current_mom)/2
    proposed_K = eval_kinetic_energy(proposed_mom)/2
    current_U = nn.MSELoss()(current_nn_model(x), y).data/2
    proposed_U = nn.MSELoss()(proposed_nn_model(x), y).data/2
    
    # Accept/reject
    if npr.rand() < np.exp(current_U + current_K - proposed_U - proposed_K) :
        return proposed_nn_model, 1
    else :
        return current_nn_model, 0 
    
    
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight)
        
        
