#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 11:03:26 2018

@author: msachs2
"""

import models
import numpy as np
import abc
import pickle

class StatsModel(models.Model):
    '''
    Base class for force fields. Each derived class must implement 
    the functions com_force and comp_potential.
    '''
    __metaclass__ = abc.ABCMeta
    
    
    def __init__(self, dim, prior, data=None, mbsampler=None):
        '''
        For every statistical model the number of dimensions and prior need to be specified
        '''
        super(StatsModel, self).__init__(dim)
        self.prior = prior
        self.data = data
        self.mbsampler = mbsampler
    
    def update_mb(self):
        if self.mbsampler is not None:
            self.mbsampler.traverse()
    
                
    def get_minibatch(self):
        if self.mbsampler is None:
            return None
        else:
            return self.mbsampler.mb    
        
    def comp_force(self, params):
        '''
        returns the force for the provided position vector  
        
        param: q: position vector, numpy array of length self.dim
            
        '''
        return self.grad_log_posterior(params, mb=self.get_minibatch())

    def comp_potential(self, params):
        '''
        returns the potential energy for the provided position vector  
        
        param: q: position vector, numpy array of length self.dim
            
        '''
        return self.log_posterior(params, mb=self.get_minibatch())
            

    def apply_boundary_conditions(self, params):
        '''
        Function which updates the position variable q according to the 
        boundary condition of the positional domain 
        '''
        pass
    
    def internal(self, params):
        return params
        
    def external(self, params):
        return params
    
    @abc.abstractmethod
    def generate_data(self, N):
        raise NotImplementedError()
        
    @abc.abstractmethod
    def log_like(self, mb):
        raise NotImplementedError()
        
    @abc.abstractmethod
    def grad_log_like(self, params, mb):
        raise NotImplementedError()
        
    def posterior(self, params, mb=None ):
        return np.exp(self.log_posterior(params, mb))
    
    def log_posterior(self, params, mb=None):
        # Optional step to turn params from vector into preferred format
        # e.g. a matrix in the case of multiclass logistic regression
        params = self.internal(params)
        
        if mb is None:
            mb = range(self.num_data_pts)
        
        n = len(mb)
        
        log_like = self.prior.log_density(params)
        for ii in mb:
            log_like += (float(self.num_data_pts)/n) * self.log_like(params, ii)
            
        log_like = self.external(log_like)
            
        return log_like
        
    
    def grad_log_posterior(self, params, mb=None):
        # Optional step to turn params from vector into preferred format
        # e.g. a matrix in the case of multiclass logistic regression
        params = self.internal(params)
        
        if mb is None:
            mb = range(self.num_data_pts)
        
        n = len(mb)
        
        grad = self.prior.grad_log_density(params)
        for ii in mb:
            grad += (float(self.num_data_pts)/n) * self.grad_log_like(params, ii)
            
        grad = self.external(grad)
            
        return grad
    

    def cov_grad_force(self, params, mb=None, precond=None, diag_only=False):
        # Calculates the empireical covariance of gradient noise
        params = self.internal(params)
        
        if mb is None:
            mb = range(self.num_data_pts)
        
        all_grad = np.zeros([len(mb),self.dim])
        
        if precond is None:
            for i in range(len(mb)):
                all_grad[i,:] = (float(self.num_data_pts)) * self.grad_log_like(params, mb[i])
        else:
            for i in range(len(mb)):
                all_grad[i,:] = (float(self.num_data_pts)) * np.matmul(precond,self.grad_log_like(params, mb[i]))
        
        if diag_only:
            return np.var(all_grad, rowvar=False)/len(mb)
        else:
            return np.cov(all_grad, rowvar=False)/len(mb)

    @abc.abstractmethod
    def output(self, params, input_value):
        raise NotImplementedError()
    


        
class GaussianMean(StatsModel):
    """
    Implements a multivariate Gaussian distribution with unknown mean
    """
    
    def __init__(self, prior, mu, cov=None):
        # Parent constructor
        super(GaussianMean, self).__init__(dim=len(mu), prior=prior)
    
        if cov is None:
            cov = np.eye(len(mu))
            
        self.true_mu = mu
        self.p = np.size(mu)
        self.cov = cov
        self.cov_inv = np.linalg.inv(cov)
        
    def generate_data(self, N):
        self.data = [np.random.multivariate_normal(self.true_mu, self.cov) for j in range(N)]
        self.num_data_pts = N
        
    def log_like(self, params, ix):
        return -np.dot(params - self.data[ix], np.dot(self.cov_inv, params - self.data[ix]))/2.0
        
    def grad_log_like(self, params, ix):
        return -np.dot(self.cov_inv, params - self.data[ix])
            
    def output(self, params, input_value):
        raise NotImplementedError('Model is neither regression nor classification: output method not implemented')


class Gaussian1DMeanAndVariance(StatsModel):
    """
    Implements a simple inference problem of a 1-dimension Gaussian distribution 
    with unknown mean and unknown variance 
    """
    
    def __init__(self, prior, mu=None, var=None):
        # Parent constructor
        super(Gaussian1DMeanAndVariance, self).__init__(dim=2, prior=prior)
    
        if var is None:
            var = 1
        if mu is None:
            mu = 0     
        self.true_mu = mu
        self.true_var = var
        self.true_pecision = 1.0/var
        
        
        
    def generate_data(self, N):
        self.data = [np.random.normal(self.true_mu, self.true_var) for j in range(N)]
        self.num_data_pts = N

    def log_like(self, params, ix):
        mu, prec = params
        return -.5 * prec*(mu - self.data[ix])**2 
        
    def grad_log_like(self, params, ix):
        mu, prec = params
        return np.array([ - prec * ( mu - self.data[ix]), .5 / prec - .5 * (mu - self.data[ix])**2])
            
    def output(self, params, input_value):
        raise NotImplementedError('Model is neither regression nor classification: output method not implemented')

        
        
class BayesianLinearRegression(StatsModel):
    """
    Implements a standard Bayesian linear regression model
    """
    
    def __init__(self, prior, true_beta, noise_var=None):
        # True linear regression parameters
        self.true_beta = true_beta
        # Number of regression parameters
        self.p = len(self.true_beta)
        # Observation noise variance
        if noise_var is None:
            noise_var = 1.0
        self.noise_var = noise_var
        self.noise_stddev = np.sqrt(self.noise_var)
        
        # Prior distribution over linear regression parameters
        super(BayesianLinearRegression, self).__init__(dim=len(true_beta), prior=prior )

    def generate_data(self, N, X=None):
        # If X data points are not specified, generate:
        if X is None:
            X = [np.random.multivariate_normal(np.zeros(self.p), np.eye(self.p))]
        # Add intercept term
        for ii in range(len(X)):
            X[ii] = np.hstack((np.array([1]), X[ii]))
        
        # Generate data points through the relationship y = X beta + eps
        self.data = [ [X[ii], np.dot(X[ii], self.true_beta) + np.random.normal(0, self.noise_stddev)]for ii in range(N)]
        
        self.num_data_pts = N

    def log_like(self, params, ix):
        est_output = np.dot(params, self.data[ix][0])
        return -(est_output - self.data[ix][1])**2 / (2.0*self.noise_var)
        
    def grad_log_like(self, params, ix):
        est_output = np.dot(params, self.data[ix][0])
        return -(est_output - self.data[ix][1]) / (self.noise_var)

    def output(self, params, input_value):
        # Return estimated output with given parameter values - remember that
        # intercept term must be added to inputs
        return np.dot(params, np.hstack((np.array([1]), input_value)))

class Classifier(StatsModel):
    __metaclass__ = abc.ABCMeta

    def __init__(self, dim, prior, n_classes):
        self.n_classes = n_classes
        super(Classifier, self).__init__(dim=dim, prior=prior )
     
        
    @abc.abstractmethod    
    def predict(self, params, x):
        '''
        returns the probability vector [P(y_i| params, x)]_{ i = 0, ..., n_classes-1}  
        '''
        raise NotImplementedError()
    
        
    def ll_prediction(self, prob, y): 
        return np.log(prob[np.argmax(y)])
    
    def cc_prediction(self, prob, y): 
        return y[np.argmax(prob)]
    
    def log_like_data_pt(self, params, data_pt):
        x,y = data_pt
        prob = self.predict(params, x)
        return self.ll_prediction( prob, y)
        
    def log_like(self, params, ix):
        return self.log_like_data_pt(params, self.data[ix])
        
    def predict_from_sample(self, params_traj, x): # params_traj  is a T \times self.dim  matrix, where each column represents one sample of params 
        '''
        returns the probability vector 1/T sum_{t=0}^{T-1}[P(y_i| params[:,t], x)]_{ i = 0, ..., n_classes-1}  
        '''
        T = params_traj.shape[0]
        prob = np.zeros(self.n_classes)
        for t in range(T):
            params = params_traj[t,:]            
            prob += self.predict(params, x)
        prob /= T
        
        return prob
                  
    def predict_from_sample_traj(self, traj_params, modprnt):
        data = self.test_data
        av_log_like_traj = []
        class_pred_traj = []
        prob_sum = np.zeros([len(data),self.n_classes])
        t = 0
        for params in traj_params:
            for i in range(len(data)):
                x, y = data[i]
                prob_sum[i,:] += self.predict( params, x)
            if t % modprnt == 0:
                ll = 0
                cc = 0
                prob = [prob_sum[i]/(t+1) for i in range(len(data))]
                for i in range(len(data)):   
                    x,y = data[i]
                    ll += self.ll_prediction(prob[i], y ) #Calculates the m
                    cc += self.cc_prediction(prob[i], y) 
                av_log_like_traj+= [ll/len(data)]
                class_pred_traj += [cc/len(data)]
            t+=1
        return av_log_like_traj, class_pred_traj
    
    def set_data(self, training_data = None, test_data=None):
        if training_data is not None:
            self.data = training_data
            self.num_data_pts = len(training_data)
            self.n_classes = len(training_data[0][1])
            print("Training data set")
        if test_data is not None:
            self.test_data = test_data
            print("Test data set")
    
class BayesianLogisticRegression(Classifier):
    def __init__(self, dim, prior, path=None):
        super(BayesianLogisticRegression, self).__init__(dim=dim, prior=prior, n_classes=2 )
        if path is not None:
            self.load_data(path)
    
    def predict(self, params, x):
        prob0 = self._logistic(np.dot(params, x))
        return np.array([prob0,1-prob0])
        
    def grad_log_like(self, params, ix):
        x, y = self.data
        prob0 = self._logistic(np.dot(params, x[ix]))
        return x[ix]*(y[ix] - prob0)
        
    def _logistic(self, x):
        expx = np.exp(x)
        if np.isinf(expx):
            prob0 = 1
        else:
            prob0 = expx/(1.0+expx)
        return prob0
    
    def _d_logistic(self, x):
        '''
@Mark:  Not sure whether this function can be deleted. It doesn't seem to be 
        used within the class definition
        '''
        return self._logistic(x)*(1 - self._logistic(x))
    
        
    def load_data(self, path):
        data = pickle.load( open(path, "rb" ) )
        self.data = data['training_data']
        self.test_data = data['test_data']
        self.num_data_pts = len(self.data)
        
    
    def generate_data(self, N, true_beta, intercept=False, X=None):
        self.true_beta = true_beta
        # If X data points are not specified, generate:
        if X is None:
            if intercept == True:
                X = [np.random.multivariate_normal(np.zeros(self.dim-1), np.eye(self.dim-1)) for i in range(N)]
                # Add intercept term
                for ii in range(len(X)):
                    X[ii] = np.hstack((np.array([1]), X[ii]))
            else:
                X = [np.random.multivariate_normal(np.zeros(self.dim), np.eye(self.dim)) for i in range(N)]
            
        # Generate data points through the relationship y = X beta + eps
        self.data = [ [X[ii], np.eye(2,dtype=int)[np.random.binomial(1, self._logistic(np.dot(X[ii], self.true_beta))),:]] for ii in range(N)]
        
        self.num_data_pts = N
           
    def output(self):
        pass
        
        
    
        
        
class BayesianMulticlassLogisticRegression(Classifier):
    def __init__(self, prior, n_pts=None, training_data=None):
        if n_pts is not None:
            self.load_data(n_pts)
        elif training_data is not None:
            self.data = training_data
            n_pts = len(training_data)
        #Missing : Handle non compatible input        
        self.num_data_pts = n_pts
        self.p = len(self.data[0][0])
        self.n_classes = len(self.data[0][1])
        print(self.num_data_pts)
        dim = len(self.data[0][0]) * len(self.data[0][1])
        super(BayesianMulticlassLogisticRegression, self).__init__(dim=dim, prior=prior)
        
    def internal(self, params):
        return np.reshape(params, (self.p, self.n_classes))
    
    def external(self, params):
        return np.reshape(params, (self.dim))
        
    def load_data(self, n_pts):
        self.data = load_mnist(n_pts)
        return len(self.data[0][0])*len(self.data[0][1])
       
    def _softmax(self, outputs):
        exp = np.exp(outputs)
        return exp/np.sum(exp)
    
    def grad_log_like(self, params, ix):
        x, y = self.data[ix]
        probs = self._softmax(np.dot(x, params))
        return np.outer(x, y - probs)
        
    def predict(self, params, x):
        params = self.internal(params)
        probs = self._softmax(np.dot(x, params))
        return probs
        
                    
    def make_prediction(self, params, data_pt):
        params = self.internal(params)
        x, y = data_pt
        probs = self.predict(params, x)
        print(probs)
        import matplotlib.pyplot as plt
        plt.pcolor(np.reshape(x, (28,28)), cmap='Greys')
    
    def generate_data(self, Ndata):
        pass
    
    def output(self, params, input_value):
        pass
        
class BayesianNeuralNetwork(Classifier):
    def __init__(self, topology, prior_var, n_pts):
        self.W = [np.zeros((topology[i], topology[i-1])) for i in range(1,len(topology))]
        self.b = [np.zeros(topology[i]) for i in range(1,len(topology))]
        self.h = [np.zeros(topology[i]) for i in range(len(topology))]
        self.a = [np.zeros(topology[i]) for i in range(len(topology))]
        self.n_layers = len(topology)
        self.topology = topology
        # Specify prior variance on weights - assume normally distributed
        # flat pseudoprior on biases
        self.prior_var = prior_var
        # data stuff
        self.dimW = [w.size for w in self.W]
        self.Wlocs = np.hstack((0, np.cumsum(self.dimW)))
        self.dimb = [b.size for b in self.b]
        self.blocs = self.Wlocs[-1] + np.hstack((0, np.cumsum(self.dimb)))
        self.dim = np.sum(self.dimW) + np.sum(self.dimb)
        self.load_data(n_pts)
    def generate_data(self,Ndata):
        pass
    
    def grad_log_prior(self, W, b):
        p_W = [-mat/self.prior_var for mat in W]
        p_b = [np.zeros(np.shape(bias)) for bias in b]
        return p_W, p_b
    
    def load_data(self, n_pts):
        self.data = load_mnist(n_pts)
        self.num_data_pts = len(self.data)
    
    def act_fn(self, x, layer):
        if layer == self.n_layers-1:
            # Softmax for final layer
            exp = np.exp(x)
            return exp/np.sum(exp)
        else:
            # Sigmoids for other layers
            return np.tanh(x)
            
    def d_act_fn(self, x):
        return 1.0 - np.tanh(x)**2
        
    def feedfwd(self, input_data, W, b):
        # Add input data
        self.a[0] = input_data
        self.h[0] = input_data
        # Feed forward through layers sequentially
        for ii in range(1,self.n_layers):
            self.h[ii] = np.dot(W[ii-1], self.a[ii-1]) + b[ii-1]
            self.a[ii] = self.act_fn(self.h[ii], ii)
        return self.a[-1]
                
    def grad_log_like(self, x, y, W, b):
        # Do feedfwd
        output = self.feedfwd(x, W, b)
        
        # Calculates gradient of cross-entropy wrt W, b via backpropagation
        dW = [np.zeros((np.shape(w))) for w in self.W]
        db = [np.zeros((np.shape(b))) for b in self.b]
        dh = [np.zeros((np.shape(h))) for h in self.h]
        da = [np.zeros((np.shape(a))) for a in self.a]

        dh[-1] = y - output
        for tt in range(self.n_layers-2, -1, -1):
            dW[tt] = np.outer(dh[tt+1], self.a[tt])
            db[tt] = dh[tt+1]
            da[tt] = np.dot(dh[tt+1], self.W[tt])
            dh[tt] = np.multiply(da[tt], self.d_act_fn(self.h[tt]))
            
        return dW, db
        
    def grad_log_posterior(self, params, ixs):
        # internal into weights and biases
        W, b = self.internal(params)
        
        # Calculate grad log prior
        glp_W, glp_b = self.grad_log_prior(W, b)
        n = len(ixs)
    
        # Calculate grad log like at data points
        for i in ixs:
            ll_W, ll_b = self.grad_log_like(self.data[i][0], self.data[i][1], W, b)
            for j in range(len(glp_W)):
                glp_W[j] += float(self.num_data_pts)/float(n)*ll_W[j]
            for j in range(len(glp_b)):
                glp_b[j] += float(self.num_data_pts)/float(n)*ll_b[j]
    
        grad = self.external(glp_W, glp_b)
        
        # Return vector
        return grad
        
    def internal(self, params):
        W = []
        b = []
        for i in range(1,self.n_layers):
            W.append(np.reshape(params[self.Wlocs[i-1]:self.Wlocs[i]], (self.topology[i], self.topology[i-1])))
            b.append(params[self.blocs[i-1]:self.blocs[i]])        
        return W, b

    def external(self, W, b):
        params = np.zeros(self.dim)
        for i in range(1,len(self.Wlocs)):
            params[self.Wlocs[i-1]:self.Wlocs[i]] = np.reshape(W[i-1], (self.dimW[i-1]))
        for i in range(1, len(self.blocs)):
            params[self.blocs[i-1]:self.blocs[i]] = np.reshape(b[i-1], (self.dimb[i-1]))
        
        return params
        
    def predict(self, params, x):
        W, b = self.internal(params)
        probs = self.feedfwd(x, W, b)
        return probs
    
    def make_prediction(self, params, data_pt):
        W, b = self.internal(params)
        x, y = data_pt
        probs = self.feedfwd(x, W, b)
        import matplotlib.pyplot as plt
        plt.pcolor(np.reshape(x, (28,28)), cmap='Greys')
        plt.show()
        return probs
        
    def output(self, params, input_value):
        pass

def load_mnist(n_pts=None,indices=None):
    import sklearn.datasets
    mnist = sklearn.datasets.fetch_mldata('MNIST original', data_home='./data/')
    # select random training data pts
    if indices is None and n_pts is not None:
        indices = np.random.choice(60000, n_pts)
    #Nead to take care of non compatible inputs: elif(indices is None and n_pts is None)
    data = [[mnist.data[i,:], one_hot(10, int(mnist.target[i]))] for i in indices]
    return data

def one_hot(n, m):
    output = np.zeros(n)
    output[m] = 1
    return output
        
        
    
class Distribution(object):
    """
    ABC for classes to be used to hold information on prior distribution
    """
    
    __metaclass__ = abc.ABCMeta
    def __init__(self):
        pass
    
    @abc.abstractmethod
    def log_density(self, params):
        raise NotImplementedError()
    
    @abc.abstractmethod
    def grad_log_density(self, params):
        raise NotImplementedError()
        
    def density(self, params):
        return np.exp(self.log_density(params))
        
        
class ProductDistribution(Distribution):

    def __init__(self, distribution_list ):
        self.d_list = distribution_list
        
    def log_density(self, x):
        log_pd  = 0
        for d in self.d_list:
            log_pd += d.log_density(x)
        return log_pd
        
    def grad_log_density(self, x):
        grad_log_pd  = 0
        for d in self.d_list:
            grad_log_pd += d.grad_log_density(x)
        return grad_log_pd 

class Uniform(Distribution):
    
    #needs some testing/extensions
    def __init__(self,a=None,b=None):
        #if a != None
        #self.h = 1.0/(b-a)
        #self.a = a 
        pass

    def log_density(self, x):
        return 0.0
        
    def grad_log_density(self, x):
        return np.zeros(x.shape)
        
    def laplace_log_density(self, x):
        return 0.0
    
      
class MVGaussian(Distribution):
    
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov
        self.cov_inv = np.linalg.inv(self.cov)
        self.cov_det = np.linalg.det(self.cov)

    def log_density(self, x):
        return -np.dot(x - self.mean, np.dot(self.cov_inv, x - self.mean))/2.0
        
    def grad_log_density(self, x):
        return -np.dot(self.cov_inv, x - self.mean)
        
    def laplace_log_density(self, x):
        return -np.trace(self.cov_inv)
    
    def draw_sample(self):
        return np.random.multivariate_normal(self.mean, self.cov, 1)
        
class IsotropicGaussian(Distribution):
    
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var
        self.var_inv = 1.0/self.var
        self.var_det = self.var

    def log_density(self, x):
        return -((x - self.mean)**2)*self.var_inv * .5
        
    def grad_log_density(self, x):
        return - self.var_inv * ( x - self.mean)
        
    def laplace_log_density(self, x):
        return - self.var_inv
    
    def draw_sample(self):
        return np.random.normal(self.mean, self.var)
        
class Laplace(Distribution):
    """
    Just axis-aligned for now
    """
    
    def __init__(self, mean):
        self.mean = mean
        
    def log_density(self, x):
        return np.sum(-np.abs(x - self.mean))
        
    def grad_log_density(self, x):
        return -np.sign(x - self.mean)
       
    def laplace_log_density(self, x):
        return 0

class Quartic_DW(Distribution):
    
    def __init__(self, a, b, c=0.0 ):
        self.a = a
        self.b = b
        self.c = c
        
    def draw_sample(self,left_trunk=None, right_trunk=None):
        if left_trunk is None:
            left_trunk = -4*self.a
        if right_trunk is None:
            right_trunk = 4*self.a
        x = np.random.uniform(left_trunk, right_trunk )
        while np.random.uniform() >  np.exp( self.log_density(x)):
            x = np.random.uniform(left_trunk, right_trunk )
        return x
        
    def log_density(self, x):
        return -np.sum((self.b*(-self.a + x**2)**2)/self.a + self.c*x)
        
    def grad_log_density(self, x):
        return -(4*self.b*x*(-self.a + x**2))/self.a - self.c
       
    def laplace_log_density(self, x):
        return -np.sum((8*self.b*x**2)/self.a + (4*self.b*(-self.a + x**2))/self.a)


class MultiModalGaussian(Distribution):
    
    def __init__(self, means, covs, wts):
        """
        means is a m x n np array, where m=# Gaussians, n=dimension of problem
        covs is a m x n x n np array
        """
        self.dim = np.shape(means)[1]
        self.gaussians = []
        self.wts = wts
        for i in range(len(means)):
            self.gaussians.append(Gaussian(means[i], covs[i]))
    
    def density(self, x):
        gaussian_densities = [np.exp(-0.5*np.dot(x-g.mean, np.dot(g.cov_inv,x-g.mean)))/np.sqrt(g.cov_det) for g in self.gaussians]/np.power(2*np.pi, self.dim/float(2))
        return np.dot(self.wts, gaussian_densities)
        
    def log_density(self, x):
        return np.log(self.density(x))

    def grad_log_density(self, x):
        #print("x=",x)
        log_dens_vec =[np.exp(-0.5*np.dot(x-g.mean, np.dot(g.cov_inv,x-g.mean)))/np.sqrt(g.cov_det) for g in self.gaussians]/np.power(2*np.pi, self.dim/float(2))
        log_dens = np.sum(log_dens_vec)
    
        #y = np.sum(np.multiply([-np.dot(g.cov_det,x-g.mean)*np.exp(-0.5*np.dot(x-g.mean, np.dot(g.cov_inv,x-g.mean)))/np.sqrt(g.cov_det) for g in self.gaussians] \
        #               /np.power(2*np.pi, self.dim/float(2)), self.wts), axis=0)
        #y = - log_dens_vec * [np.dot(g.cov_inv,x-g.mean) for g in self.gaussians] / log_dens
        #print("y = ",y)
        y = 0
        for k in range(len(self.gaussians)):
            y += log_dens_vec[k] * np.dot(self.gaussians[k].cov_inv,x-self.gaussians[k].mean) 
        return - y / log_dens

class WarpedGaussian(Distribution):
    
    def __init__(self, a,b,c, beta=1.0):
        self.a = a
        self.b = b
        self.c = c
        self.beta = beta
        self.Z = np.sqrt(self.a) * np.pi / self.beta
    
    def density(self,x):
        return np.exp(self.beta*self.log_density(x))/self.Z

    def log_density(self, x):   
        return - (x[0]**2/self.a+(x[1]+self.b*x[0]**2-self.c)**2)
        
    def grad_log_density(self, x):
        grad = np.zeros(2)
        grad[0] = - ( 2 * x[0]/ self.a + 4 * self.b * x[0]* (-self.c +self.b * np.power(x[0],2) + x[1]))
        grad[1] = - ( 2 * (-self.c + self.b * np.power(x[0], 2) + x[1]) )
        return grad
        
class NormalGamma(Distribution):
    
    # alpha shape parameter
    # beta rate parameter 
    def __init__(self, mu, prec, alpha, beta):
        self.mu = mu
        self.prec = prec
        self.alpha = alpha
        self.beta = beta
        self.nconst = np.log(beta**alpha*np.sqrt(prec)/(scipy.special.gamma(alpha)*np.sqrt(2*np.pi)))
        
    def log_density(self, x): #Const ommitted
        return self.nconst + (self.alpha -.5) * np.log(x[1]) - self.beta * x[1] - .5 * self.prec * x[1]*(x[0]-self.mu)**2
        
    def grad_log_density(self, x):
        dx0 = -self.prec * x[1]*(x[0]-self.mu)
        dx1 = (self.alpha -.5)/x[1] - self.beta - .5 * self.prec  * (x[0]-self.mu)**2
        return np.array([dx0,dx1])
        
    def marginal_prec(self):
        marginal =  Gamma(self.alpha, self.beta )
        return marginal

class Gamma(Distribution):
    
    # alpha shape parameter
    # beta rate parameter 
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.logbaG = alpha*np.log(beta) - np.log( scipy.special.gamma(self.alpha))

    def density(self, x):
        return np.prod(scipy.stats.gamma.pdf(x, self.alpha, scale=1.0/self.beta))
        #1.0 / scipy.special.gamma(self.alpha) * x**(self.alpha-1)*np.exp(-self.beta*x)
    def draw_sample(self):
        return np.random.gamma(shape=self.alpha, scale=1.0/self.beta)
        
    def log_density(self, x):
        return np.sum(self.logbaG +(self.alpha-1)*np.log(x) - self.beta * x)
        
    def grad_log_density(self, x):
        return (self.alpha-1)/x - self.beta

    def laplace_log_density(self, x):
        return np.sum(-(self.alpha-1)/x**2)


class IsotropicGaussianMatrix(Distribution):
    
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var
        self.var_inv = 1.0/self.var
        
    def log_density(self, x):
        return -np.dot(x - self.mean, np.multiply(self.var_inv, x - self.mean))/2.0
        
    def grad_log_density(self, x):
        return -np.multiply(self.var_inv, x - self.mean)

 
if __name__ == "__main__":
    '''
    Example execution - standard SGLD for Gaussian mean inference
    '''
    # Specify prior over mean of Gaussian
    prior = Gaussian(mean=np.array([0, 0]),
                     cov=5*np.eye(2))
    # Set up the likelihood model
    model = GaussianMean(prior, mu=np.array([-2.0,2]), cov=np.array([[5.0, 2],[2,4]]))
    # Generate some data
    model.generate_data(30)
    
    # Now query the model for stochastic gradients
    # The call below give the estimate for the gradient at parameter value
    # [0, 0], using the data points with indices 0, 2, 6, and 13
    model.grad_log_posterior(np.array([0,0]), [0, 2, 6, 13])
  
class ReplicatedStatsModel(StatsModel):
    """ Baseclass for models used in samplers using multiple replicas.
    """    
    
    def __init__(self, statsmodel, nreplicas=1):
        self.model_list = [statsmodel for i in range(nreplicas)]
        self.pmdim =  statsmodel.dim
        self.dim = self.pmdim  * nreplicas
        self.nreplicas = nreplicas
        
        if hasattr(statsmodel, 'data'):
            self.data = statsmodel.data
            self.num_data_pts = statsmodel.num_data_pts
        #self.q = np.repeat(np.reshape(model.q, (-1, model.dim)),self.nreplicas, axis=0)
        #self.force = np.repeat(np.reshape(model.f, (-1, model.dim)),self.nreplicas, axis=0)
        #if model.p is not None:
        #    self.p = np.repeat(np.reshape(model.p, (-1, model.dim)),self.nreplicas, axis=0)
    
    def update_mb(self):
        for i in range(self.nreplicas):
            if self.model_list[i].mbsampler is not None:
                self.model_list[i].mbsampler.traverse()
        
    def comp_potential(self, q):
        """ returns the potential at the current position
        """
        pot = 0.0
        for i in range(self.nreplicas):
            pot+= self.model_list[i].comp_potential(q[i*self.pmdim:((i+1)*self.pmdim)])
            
        return pot
        
    def comp_force(self, q):
        """ updates the force internally from the current position
        """
        force = np.zeros( self.nreplicas * self.pmdim )
        for i in range(self.nreplicas):
            force[i*self.pmdim:((i+1)*self.pmdim)] = self.model_list[i].comp_force(q[i*self.pmdim:((i+1)*self.pmdim)])
            
        return force
    
    
    def apply_boundary_conditions(self, q):
        pass # In this implementation the class ReplicatedModel is assumed to have no speficied boundary conditions



    
class SyntheticNoiseModel(StatsModel):
    """
    Target model where gradient log density is corrupted by an additive noise process  
    """
    def __init__(self, grad_log_target, dim, noise_generator=None, cov_noise_generator=None):
        
        self.grad_log_target = grad_log_target
        self.dim = dim
        self.noise_generator = noise_generator
        self.mbsampler = None
        self.cov_noise_generator = cov_noise_generator
        self.G =None
        
    def generate_data(self, N):
        pass
    
    def log_like(self, mb=None):
        pass
        
    def grad_log_like(self, params, mb=None):
        pass #raise NotImplementedError? 

    def comp_force_exact(self, params):
        return self.grad_log_target(params)
    
    def comp_force(self, params):
        if self.noise_generator is not None:
            return self.grad_log_target(params) + self.noise_generator.pull(self.dim)
        else:
            return self.grad_log_target(params)
    
    def cov_grad_force(self, params, mb=None, precond=None, diag_only=False):
        
        params = self.internal(params)
        if self.G is None:
            if precond is not None:
                self.G = np.matmul(precond,np.matmul(self.noise_generator.cov,np.transpose(precond)))
            else:
                self.G = self.noise_generator.cov

        if self.cov_noise_generator is not None:
            if diag_only:
                return np.diag(self.G) + self.cov_noise_generator.pull(dim=self.dim)
            else:
                return self.G  + self.cov_noise_generator.pull(dim=self.dim**2).reshape([self.dim,self.dim])
        else:      
            if diag_only:
                return np.diag(self.G)
            else:
                return self.G


class NoiseStream(object):
    __metaclass__ = abc.ABCMeta
            
    def pull(self):
        raise NotImplementedError()

class BinaryNoise(NoiseStream):
    
    def __init__(self, var):
        self.var = var;
        self.std = np.sqrt(var);

    def pull(self, dim):
        return self.std*(2.0 * np.random.binomial(1, .5, size=dim)-1)


class UniformNoise(NoiseStream):
    
    def __init__(self, var):
        self.a = np.sqrt(12.0*var/4.0);
        self.std = np.sqrt(var);

    def pull(self, dim):
        return np.random.uniform(-self.a,self.a, size=dim)

class MVGaussianNoise(NoiseStream):
    
    def __init__(self, mean, cov, cov_est_noise=0.0):
        self.mean = mean
        self.cov = cov
        self.cov_est_noise = cov_est_noise
        
    def estimate_cov(self):
        return self.cov + np.random.normal(0.0, 1.0, self.cov.shape)
        

    def pull(self, dim):
        return np.random.multivariate_normal(self.mean, self.cov)
        