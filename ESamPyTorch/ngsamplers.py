#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 11:25:01 2018

@author: msachs2
"""

import numpy as np
import abc
import integrators



class NGThermostat(integrators.Thermostat):
    __metaclass__ = abc.ABCMeta
            
    def traverse(self):
        raise NotImplementedError()
        
    def update_dyn_values(self):  

        self.model.update_mb()    
        self.force = self.model.comp_force(self.q)
            
    def print_summary(self):
        pass



class NGKineticThermostat(NGThermostat):
    __metaclass__ = abc.ABCMeta
    """
    Base class for kinetic thermostats (thermostat methods possesing a momentum
    variable)
    """
    def __init__(self, model, h, Tk_B=1.0):
        '''
        :param Tk_B: temperature paramameter
        
        for other parameters see parent class
        '''
        super(NGKineticThermostat,self).__init__(model, h, Tk_B)
        self.p = np.zeros(self.model.dim)
        
class SGLD(NGThermostat):
    '''
    Base class for numerical integrators for Brownian dynamics 
    '''
    def __init__(self, model, h, Tk_B=1.0):
        self.zeta = None
        super(SGLD,self).__init__(model, h, Tk_B)

    def set_Tk_B(self, Tk_B):
        self.Tk_B = Tk_B
        self.zeta = np.sqrt(2.0 * self.h * self.Tk_B)
             
    def traverse(self):
        self.q = self.q + self.h * self.force + self.zeta * np.random.normal(0, 1,self.model.dim)
        self.model.apply_boundary_conditions(self.q)
        
        self.model.update_mb()
        self.force = self.model.comp_force(self.q)
        #self.model.grad_log_posterior(self.q, self.model.mbsampler.get_minibatch())


        
class NGLangevinThermostat(NGKineticThermostat):
    __metaclass__ = abc.ABCMeta
    """
    Base class for thermostats implementing the underdamped Langevin equation 
    """
    
    def __init__(self, model, h, Tk_B=1.0, gamma=1.0):
        """ Init function for the class
        '''
        :param gamma:   friction coefficient
                        
        for other parameters see parent class
        '''
        """
        super(NGLangevinThermostat,self).__init__(model, h, Tk_B)
        self.gamma = gamma
        
class NGLangevinBAOSplitting(NGLangevinThermostat):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, model, h, Tk_B=1.0, gamma=1.0):
        super(NGLangevinBAOSplitting,self).__init__(model, h, Tk_B,  gamma)
        
        self.alpha = np.exp(-self.h * self.gamma)
        self.zeta = np.sqrt((1.0-self.alpha**2)*self.Tk_B)
        self.alpha2 = np.exp(-.5 * self.h * self.gamma)
        self.zeta2 = np.sqrt((1.0-self.alpha2**2)*self.Tk_B)
        
        
class NGEnsembleQuasiNewton(NGLangevinBAOSplitting):
    """ Sampler implementing ensemble Quasi Newton method 
    implementation is witout local weighting of estimates of the walker covariance
    """
    
    def __init__(self, repmodel, h, Tk_B=1.0, gamma=1.0, regparams=1.0, B_update_mod=1, lamb_max=None):

        
        super(NGEnsembleQuasiNewton, self).__init__(repmodel, h, Tk_B, gamma)
        self.pmdim = self.model.pmdim
        self.Bmatrix = np.zeros([self.model.nreplicas, self.pmdim, self.pmdim ])
        for i in range(self.model.nreplicas):
            self.Bmatrix[i,:,:] = np.eye( self.pmdim)
        self.regparams = regparams
        self.B_update_mod = B_update_mod
        self.substep_counter = 0
        self.lamb_max = lamb_max
    
    def traverse(self):
        
        nreplicas = self.model.nreplicas
           
        # update preconditioner
        if self.substep_counter % self.B_update_mod == 0:
            self.update_Bmatrix()
            
        # B-step
        for i in range(nreplicas):
            self.p[i*self.pmdim:(i+1)*self.pmdim] += .5 * self.h * np.matmul(np.transpose(self.Bmatrix[i,:,:]), self.force[i*self.pmdim:(i+1)*self.pmdim])

        # A-step
        for i in range(nreplicas):
            self.q[i*self.pmdim:(i+1)*self.pmdim] += .5 * self.h  * np.matmul(self.Bmatrix[i,:,:], self.p[i*self.pmdim:(i+1)*self.pmdim])
                    
        # O-step
        self.p = self.alpha * self.p + self.zeta * np.random.normal(0., 1., self.model.dim)
         
        # A-step
        for i in range(nreplicas):
            self.q[i*self.pmdim:(i+1)*self.pmdim] += .5 * self.h * np.matmul(self.Bmatrix[i,:,:], self.p[i*self.pmdim:(i+1)*self.pmdim]) 

        # update force
        self.model.apply_boundary_conditions(self.q)
        self.model.update_mb()
        self.force = self.model.comp_force(self.q)
        
        # B-step   
        for i in range(nreplicas):
            self.p[i*self.pmdim:(i+1)*self.pmdim] += .5 * self.h * np.matmul(np.transpose(self.Bmatrix[i,:,:]), self.force[i*self.pmdim:(i+1)*self.pmdim])

            
        self.substep_counter+=1
        
    def update_Bmatrix(self):
        if self.model.nreplicas > 1:
            indices = [i for i in range(self.model.nreplicas)]
            for r in range(self.model.nreplicas):
                mask =  np.array(indices[:r] + indices[(r + 1):])
                #print(np.cov(self.q.reshape([self.model.nreplicas,self.pmdim])[mask,:],rowvar=False) + self.regparams * np.eye(self.pmdim))
                self.Bmatrix[r,:,:] = np.linalg.cholesky(
                        np.cov(self.q.reshape([self.model.nreplicas,self.pmdim])[mask,:],rowvar=False) + self.regparams * np.eye(self.pmdim)
                                                        )
                if self.lamb_max is not None:
                    lamb = np.linalg.norm(self.Bmatrix[r,:,:],2) * self.h
                    if lamb > self.lamb_max:
                        self.Bmatrix[r,:,:] *= (self.lamb_max/lamb)
                                                
                        
class NGEnsembleQuasiNewtonNose(NGEnsembleQuasiNewton):
    
    def __init__(self, repmodel, h, Tk_B=1.0, gamma=1.0, regparams=1.0, B_update_mod=1,  lamb_max=None, mu=1.0):
        super(NGEnsembleQuasiNewtonNose, self).__init__(repmodel, h, Tk_B, gamma, regparams, B_update_mod=B_update_mod, lamb_max=lamb_max)
        self.xi = np.zeros(self.model.dim)
        self.mu = mu
        
    
    def traverse(self):
        
        nreplicas = self.model.nreplicas
           
         # update preconditioner
        if self.substep_counter % self.B_update_mod == 0:
            self.update_Bmatrix()
            
        # B-step
        for i in range(nreplicas):
            self.p[i*self.pmdim:(i+1)*self.pmdim] += .5 * self.h * np.matmul(np.transpose(self.Bmatrix[i,:,:]), self.force[i*self.pmdim:(i+1)*self.pmdim])
        
        # A-step
        for i in range(nreplicas):
            self.q[i*self.pmdim:(i+1)*self.pmdim] += .5 * self.h  * np.matmul(self.Bmatrix[i,:,:], self.p[i*self.pmdim:(i+1)*self.pmdim])


        # D-step:
        self.xi += .5 * self.h * (self.p**2 - self.Tk_B) / self.mu
        
        # O-step
        #if np.abs(self.gamma + self.xi) < 0.001:
        alpha = np.exp(- (self.gamma + self.xi) * self.h)
        self.p = alpha * self.p + np.sqrt(self.gamma * (1.0 - alpha**2) / (self.gamma + self.xi)  ) * np.random.normal(0, 1, self.model.dim) 
        #else:
        #self.p += self.p * self.h * self.xi + np.sqrt(2.0 * self.Tk_B * self.h) *  np.random.normal(0., 1., self.model.dim)
        
        # D-step:
        self.xi += .5 * self.h * (self.p**2 - self.Tk_B) / self.mu
        
        # A-step
        for i in range(nreplicas):
            self.q[i*self.pmdim:(i+1)*self.pmdim] += .5 * self.h * np.matmul(self.Bmatrix[i,:,:], self.p[i*self.pmdim:(i+1)*self.pmdim]) 

        # update force
        self.model.apply_boundary_conditions(self.q)
        self.model.update_mb()
        self.force = self.model.comp_force(self.q)
        
        # B-step   
        for i in range(nreplicas):
            self.p[i*self.pmdim:(i+1)*self.pmdim] += .5 * self.h * np.matmul(np.transpose(self.Bmatrix[i,:,:]), self.force[i*self.pmdim:(i+1)*self.pmdim])
        
        
                       
        self.substep_counter+=1

class NGEnsembleQuasiNewtonNOGIM(NGEnsembleQuasiNewton):
    
    def __init__(self, repmodel, h, Tk_B=1.0, gamma=1.0, regparams=1.0, B_update_mod=1,  lamb_max=None, diag_only=False):
        super(NGEnsembleQuasiNewtonNOGIM, self).__init__(repmodel, h, Tk_B, gamma, regparams, B_update_mod=B_update_mod, lamb_max=lamb_max)

        self.Sigma_est = [ np.zeros([self.pmdim,self.pmdim]) for i in range(self.model.nreplicas)]
        self.lambh = np.sqrt((1.0 + np.exp(-self.gamma * self.h)) / (1.0 - np.exp(-self.gamma * self.h)))
        self.G = np.zeros([self.pmdim,self.pmdim])
        self.noise_vec = np.zeros([self.model.nreplicas,self.pmdim])
        self.diag_only = diag_only
    
    def traverse(self):
        
        nreplicas = self.model.nreplicas
           
    
        # update preconditioner
        if self.substep_counter % self.B_update_mod == 0:
            self.update_Bmatrix()
            
        # A-step
        for i in range(nreplicas):
            self.q[i*self.pmdim:(i+1)*self.pmdim] += .5 * self.h  * np.matmul(self.Bmatrix[i,:,:], self.p[i*self.pmdim:(i+1)*self.pmdim])
        
   
        # update force
        self.model.apply_boundary_conditions(self.q)
        self.model.update_mb()
        self.force = self.model.comp_force(self.q)
     

        self.noise_vec = self.lambh * np.random.normal(0,1.0,self.noise_vec.shape)
        # B-step
        for i in range(nreplicas):
            self.p[i*self.pmdim:(i+1)*self.pmdim] += (
                    .5 * self.h * np.matmul(np.transpose(self.Bmatrix[i,:,:]), self.force[i*self.pmdim:(i+1)*self.pmdim]) 
                    + self.noise_vec[i,:].flatten() 
                    )
      
        # O-step
        for i in range(nreplicas):
            self.Sigma_est[i] = self.model.model_list[i].cov_grad_force(self.q, mb=None, precond=np.transpose(self.Bmatrix[i,:,:]), diag_only=self.diag_only)
            #self.Sigma_est[i] = self.model.model_list[i].cov_grad_force(self.q, mb=None, precond=None, diag_only=False)
        
        if self.diag_only:
            for i in range(nreplicas):
                self.p[i*self.pmdim:(i+1)*self.pmdim] *= ((1 - self.lambh**2) - .25 * self.h**2 * self.Sigma_est[i]) /((1+self.lambh**2) + .25 * self.h**2 * self.Sigma_est[i])
        else:
            for i in range(nreplicas):
                self.G = (1+self.lambh**2)*np.eye(self.pmdim) + .25 * self.h**2 * self.Sigma_est[i]
                self.p[i*self.pmdim:(i+1)*self.pmdim] = np.linalg.solve(self.G, self.p[i*self.pmdim:(i+1)*self.pmdim])
                self.G = (1 - self.lambh**2)*np.eye(self.pmdim) - .25 * self.h**2 * self.Sigma_est[i]
                self.p[i*self.pmdim:(i+1)*self.pmdim] = np.dot(self.G,self.p[i*self.pmdim:(i+1)*self.pmdim])
         
        # B-step
        for i in range(nreplicas):
            self.p[i*self.pmdim:(i+1)*self.pmdim] += (
                    .5 * self.h * np.matmul(np.transpose(self.Bmatrix[i,:,:]), self.force[i*self.pmdim:(i+1)*self.pmdim]) 
                    + self.noise_vec[i,:].flatten() 
                    )

        # A-step
        for i in range(nreplicas):
            self.q[i*self.pmdim:(i+1)*self.pmdim] += .5 * self.h * np.matmul(self.Bmatrix[i,:,:], self.p[i*self.pmdim:(i+1)*self.pmdim]) 

           
        self.substep_counter+=1


class NGEnsembleQuasiNewtonNoseNOGIM(NGEnsembleQuasiNewtonNose):
    
    def __init__(self, repmodel, h, Tk_B=1.0, gamma=1.0, regparams=1.0, B_update_mod=1,  lamb_max=None, diag_only=False, mu=1.0):
        super(NGEnsembleQuasiNewtonNoseNOGIM, self).__init__(repmodel, h, Tk_B, gamma, regparams, B_update_mod=B_update_mod, lamb_max=lamb_max, mu=mu)

        self.Sigma_est = [ np.zeros([self.pmdim,self.pmdim]) for i in range(self.model.nreplicas)]
        self.lambh = np.sqrt((1.0 + np.exp(-self.gamma * self.h)) / (1.0 - np.exp(-self.gamma * self.h)))
        self.G = np.zeros([self.pmdim,self.pmdim])
        self.noise_vec = np.zeros([self.model.nreplicas,self.pmdim])
        self.diag_only = diag_only
    
    def traverse(self):
        
        nreplicas = self.model.nreplicas
           
    
        # D-step:
        self.xi += .5 * self.h * (self.p**2 - self.Tk_B) / self.mu
        self.p *= np.exp(-self.xi*.5 * self.h)
        
        # A-step
        for i in range(nreplicas):
            self.q[i*self.pmdim:(i+1)*self.pmdim] += .5 * self.h  * np.matmul(self.Bmatrix[i,:,:], self.p[i*self.pmdim:(i+1)*self.pmdim])
        
       
        
        # update preconditioner
        if self.substep_counter % self.B_update_mod == 0:
            self.update_Bmatrix()
   
        # update force
        self.model.apply_boundary_conditions(self.q)
        self.model.update_mb()
        self.force = self.model.comp_force(self.q)
     

        self.noise_vec = self.lambh * np.random.normal(0,1.0,self.noise_vec.shape)
        # B-step
        for i in range(nreplicas):
            self.p[i*self.pmdim:(i+1)*self.pmdim] += (
                    .5 * self.h * np.matmul(np.transpose(self.Bmatrix[i,:,:]), self.force[i*self.pmdim:(i+1)*self.pmdim]) 
                    + self.noise_vec[i,:].flatten() 
                    )
      
        # O-step
        for i in range(nreplicas):
            self.Sigma_est[i] = self.model.model_list[i].cov_grad_force(self.q, mb=None, precond=self.Bmatrix[i,:,:], diag_only=self.diag_only)
            #self.Sigma_est[i] = self.model.model_list[i].cov_grad_force(self.q, mb=None, precond=np.transpose(self.Bmatrix[i,:,:]), diag_only=self.diag_only)

            #self.Sigma_est[i] = self.model.model_list[i].cov_grad_force(self.q, mb=None, precond=None, diag_only=False)
        
        if self.diag_only:
            for i in range(nreplicas):
                self.p[i*self.pmdim:(i+1)*self.pmdim] *= ((1 - self.lambh**2) - .25 * self.h**2 * self.Sigma_est[i]) /((1+self.lambh**2) + .25 * self.h**2 * self.Sigma_est[i])
        else:
            for i in range(nreplicas):
                self.G = (1+self.lambh**2)*np.eye(self.pmdim) + .25 * self.h**2 * self.Sigma_est[i]
                self.p[i*self.pmdim:(i+1)*self.pmdim] = np.linalg.solve(self.G, self.p[i*self.pmdim:(i+1)*self.pmdim])
                self.G = (1 - self.lambh**2)*np.eye(self.pmdim) - .25 * self.h**2 * self.Sigma_est[i]
                self.p[i*self.pmdim:(i+1)*self.pmdim] = np.dot(self.G,self.p[i*self.pmdim:(i+1)*self.pmdim])
            
        # B-step
        for i in range(nreplicas):
            self.p[i*self.pmdim:(i+1)*self.pmdim] += (
                    .5 * self.h * np.matmul(np.transpose(self.Bmatrix[i,:,:]), self.force[i*self.pmdim:(i+1)*self.pmdim]) 
                    + self.noise_vec[i,:].flatten() 
                    )
            
        
        
        # A-step
        for i in range(nreplicas):
            self.q[i*self.pmdim:(i+1)*self.pmdim] += .5 * self.h * np.matmul(self.Bmatrix[i,:,:], self.p[i*self.pmdim:(i+1)*self.pmdim]) 

        # D-step:
        self.p *= np.exp(-self.xi*.5 * self.h)
        self.xi += .5 * self.h * (self.p**2 - self.Tk_B) / self.mu
        
        self.substep_counter+=1
        
class NGEnsembleQuasiNewtonNoseODABADO(NGEnsembleQuasiNewtonNose):

    def traverse(self):
        
        
        
        nreplicas = self.model.nreplicas
          
        # O-step
        alpha = np.exp(- (self.gamma + self.xi) * .5 * self.h)
        self.p = alpha * self.p + np.sqrt(self.gamma * (1.0 - alpha**2) / (self.gamma + self.xi)  ) * np.random.normal(0, 1, self.model.dim) 
        
        # D-step:
        self.xi += .5 * self.h * (self.p**2 - self.Tk_B) / self.mu
  
    
        # A-step
        for i in range(nreplicas):
            self.q[i*self.pmdim:(i+1)*self.pmdim] += .5 * self.h  * np.matmul(self.Bmatrix[i,:,:], self.p[i*self.pmdim:(i+1)*self.pmdim])

        # update preconditioner
        if self.substep_counter % self.B_update_mod == 0:
            self.update_Bmatrix()
        
         # update force
        self.model.apply_boundary_conditions(self.q)
        self.model.update_mb()
        self.force = self.model.comp_force(self.q)
        
        # B-step
        for i in range(nreplicas):
            self.p[i*self.pmdim:(i+1)*self.pmdim] += self.h * np.matmul(np.transpose(self.Bmatrix[i,:,:]), self.force[i*self.pmdim:(i+1)*self.pmdim])
        
        # A-step
        for i in range(nreplicas):
            self.q[i*self.pmdim:(i+1)*self.pmdim] += .5 * self.h * np.matmul(self.Bmatrix[i,:,:], self.p[i*self.pmdim:(i+1)*self.pmdim]) 
         
        # D-step:
        self.xi += .5 * self.h * (self.p**2 - self.Tk_B) / self.mu

        # O-step
        alpha = np.exp(- (self.gamma + self.xi) * .5 * self.h)
        self.p = alpha * self.p + np.sqrt(self.gamma * (1.0 - alpha**2) / (self.gamma + self.xi)  ) * np.random.normal(0, 1, self.model.dim) 
                          
        self.substep_counter+=1
        
class MBSampler(object):
    """
    
    """
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, num_data_pts, mbsize):
        self.num_data_pts= num_data_pts    # Number of datapoints in the complete dataset
        self.mbsize = mbsize  # Number of datapoints in the minibatch
        self.mb = np.random.randint(self.num_data_pts, size=self.mbsize)
        
    def traverse(self):
        raise NotImplementedError()
        
class MBSampler_MRiid(MBSampler):
    """
    Minibatch sampler which is designed such that at each time step all entries
    in the minibatch are replaced by indicies sampled uniformly and 
    independently from the index set of the complete dataset    
    """
    def traverse(self):
        self.mb =  np.random.randint(self.num_data_pts, size=self.mbsize)
        
class ReplicatedMBSampler(MBSampler):
    
    def __init__(self, mbsampler, nreplicas):
        
        self.mbsampler = mbsampler
        self.nreplicas = nreplicas
    
        