import numpy           as np
import pandas          as pd
import seaborn         as sns
import scipy.stats     as stats

from matplotlib        import pyplot as plt
from sklearn.cluster   import KMeans
from tqdm.notebook     import tqdm
from scipy.special     import digamma
from scipy.optimize    import fsolve

class GMM:
    
    '''
    1D Gaussian Mixture Model of 3 components - Optimised by expectation maximisation
    
    Methods
    -------
    
    calculate_expectation()
    optimise()
        Maximises the expctation
    fit()
        Calls the EM methods for n iterations
    print_params
        Prints the model parameters
    sample
        Generate samples from the fitted model
    '''
    
    def __init__(self, data,
                 epochs     = 10,
                 seed       = 0):
        
        if seed is not None: np.random.seed(seed)

        data = data.reshape(-1,1)
            
        self.n, self.m    = data.shape
        self.X            = data
        self.clusters     = 3
        self.epochs       = epochs
        
        self.comp_weights = np.zeros(self.clusters)
        self.means        = np.zeros(self.clusters)
        self.vars         = np.ones(self.clusters)
        self.weights      = np.zeros((self.n, self.clusters))
        
        self.left_alpha   = 1
        self.right_alpha  = 1
        self.left_beta    = 1
        self.right_beta   = 1

        self.preds        = np.zeros(self.X.shape)
        self.left_perc    = self.X < np.percentile(self.X, 20)
        self.right_perc   = self.X > np.percentile(self.X, 80)

        self.preds[self.left_perc]  = 1      
        self.preds[self.right_perc] = 2

        self.labels       = np.unique(self.preds)
        self.comp_weights = np.array([np.mean((self.preds == i)*1) for i in range(self.clusters)])
        self.means        = np.array([np.mean(self.X[self.preds == i], axis = 0) for i in range(self.clusters)])
        self.vars         = np.array([np.var(self.X[self.preds == i], axis = 0) for i in range(self.clusters)])
             
        return
        
    def calculate_expectation(self):
        
        ''' Calculate the likelihood of the data under the Gaussians '''

        self.pdf0 = stats.multivariate_normal(self.means[0], self.vars[0], allow_singular=True).pdf(self.X).flatten()

        self.pdf1 = stats.multivariate_normal(self.means[1], self.vars[1], allow_singular=True).pdf(self.X).flatten()
        
        self.pdf2 = stats.multivariate_normal(self.means[2], self.vars[2], allow_singular=True).pdf(self.X).flatten()

        self.weights[:, 0] = self.comp_weights[0] * self.pdf0
        self.weights[:, 1] = self.comp_weights[1] * self.pdf1
        self.weights[:, 2] = self.comp_weights[2] * self.pdf2

        self.weights /= self.weights.sum(axis = 1)[:,None]
        
        return
    
    def maximise_expectation(self):
        
        self.comp_weights  = np.mean(self.weights, axis = 0)
        
        self.means = np.array([(self.weights[:,i].T @ self.X) / np.sum(self.weights[:,i]) for i in range(self.clusters)]).flatten()
        
        for k, kitem in enumerate(range(self.clusters)):

            new_demeaned = self.X - self.means[k]
    
            self.vars[k] = new_demeaned.T @ (self.weights[:,k]*np.eye(self.weights[:,k].shape[0])) @ new_demeaned/np.sum(self.weights, axis = 0)[k]
    
    def fit(self):
        
        for epoch in range(self.epochs):

            self.calculate_expectation()
            self.maximise_expectation()
            
        return self
    
    def sample(self, n_samples = 1000, plot = True):
        
        ''' Sample from our fitted mixture model and plot the histogram of samples '''
        
        samples    = []
        
        dist_split = np.random.multinomial(n_samples, self.comp_weights)
        
        for i in range(self.clusters):
                
            sample = np.random.normal(self.means[i], np.sqrt(self.vars[i]), dist_split[i])
                
            samples = np.append(samples, sample)

        samples = np.array(samples)

        if plot:
            
            sns.distplot(samples, bins = 100, label = 'Sampled')
            sns.distplot(self.X, bins = 100, label = 'Original')
            plt.legend()
        
        return samples
    
    def print_params(self):
            
        print(r'Left Gaussian:    mu    = {} and sigma^2 = {},               comp_weight = {}'.format(np.round(self.means[1], 3), np.round(self.vars[1], 3), self.comp_weights[1]))
        print()
        print(r'Right Gaussian:   mu    = {} and sigma^2 = {},               comp_weight = {}'.format(np.round(self.means[2], 3), np.round(self.vars[2], 3), self.comp_weights[2]))
        print()
        print(r'Central Gaussian: mu    = {} and sigma^2 = {},               comp_weight = {}'.format(np.round(self.means[0], 3), np.round(self.vars[0], 3), self.comp_weights[0]))

    
##############################################################################################################################
        
class GaussGammaMM:
    
    '''
    1D Gaussian-Gamma Mixture Model of 3 components - Optimised by expectation maximisation
    
    Methods
    -------
    
    calculate_expectation()
    optimise()
        Maximises the expctation
    fit()
        Calls the EM methods for n iterations
    print_params
        Prints the model parameters
    sample
        Generate samples from the fitted model
    '''
    
    def __init__(self, data,
                 clusters   = 3,
                 epochs     = 10,
                 seed       = 0):
        
        if seed is not None: np.random.seed(seed)
        
        data = data.flatten()
        
        self.n            = data.shape[0]
        self.X            = data
        self.clusters     = clusters
        self.epochs       = epochs
        
        self.comp_weights = np.zeros(self.clusters)
        self.means        = np.zeros(self.clusters)
        self.vars         = np.ones(self.clusters)
        self.weights      = np.zeros((self.n, self.clusters))
        
        self.left_alpha   = 2
        self.right_alpha  = 2
        self.left_beta    = 0.5
        self.right_beta   = 0.5

        self.preds        = np.zeros(self.X.shape)
        self.left_perc    = self.X < np.percentile(self.X, 5)
        self.right_perc   = self.X > np.percentile(self.X, 95)

        self.preds[self.left_perc]  = 1      
        self.preds[self.right_perc] = 2

        self.labels       = np.unique(self.preds)
        self.comp_weights = np.array([np.mean((self.preds == i)*1) for i in range(self.clusters)])
        self.means        = np.array([np.mean(self.X[self.preds == i], axis = 0) for i in range(self.clusters)])
        self.vars         = np.array([np.var(self.X[self.preds == i], axis = 0) for i in range(self.clusters)])
             
        return
        
    def calculate_expectation(self):
        
        self.pdf0 = stats.multivariate_normal(self.means[0], self.vars[0], allow_singular=True).pdf(self.X).flatten()
        
        self.pdf1 = stats.gamma(a = self.left_alpha,  scale = 1/self.left_beta,  loc  = 0).pdf(-(self.X)).flatten()

        print(self.pdf1)
        
        self.pdf2 = stats.gamma(a = self.right_alpha, scale = 1/self.right_beta, loc  = 0).pdf(self.X).flatten() 
    
        self.weights[:, 0] = self.comp_weights[0] * self.pdf0
        self.weights[:, 1] = self.comp_weights[1] * self.pdf1
        self.weights[:, 2] = self.comp_weights[2] * self.pdf2

        self.weights /= self.weights.sum(axis = 1)[:,None]
        
        return
    
    def maximise_expectation(self):
        
        self.comp_weights  = np.mean(self.weights, axis = 0)
        
        self.means = np.array([(self.weights[:,i].T @ self.X) / np.sum(self.weights[:,i]) for i in range(self.clusters)]).flatten()
        
        for k, kitem in enumerate(range(self.clusters)):

            new_demeaned = self.X - self.means[k]
    
            self.vars[k] = new_demeaned.T @ (self.weights[:,k]*np.eye(self.weights[:,k].shape[0])) @ new_demeaned/np.sum(self.weights, axis = 0)[k]
        
        left_mean  = -self.means[1] #Scalar
        left_var   = self.vars[1]   #Scalar
        right_mean = self.means[2]  #Scalar
        right_var  = self.vars[2]   #Scalar
        
        self.left_alpha = (left_mean**2)/left_var
        self.left_beta  = left_var/left_mean
        self.right_alpha = (right_mean**2)/right_var
        self.right_beta  = right_var/right_mean
        
        print(self.left_alpha, self.right_alpha)
        
        print(self.left_beta, self.right_beta)
    
    def fit(self):
        
        for epoch in range(self.epochs):

            self.calculate_expectation()
            self.maximise_expectation()
            
        return self
    
    def sample(self, n_samples = 1000, plot = True):
        
        samples    = []
        
        dist_split = np.random.multinomial(n_samples, self.comp_weights)
        
        for i in range(self.clusters):
                
            if   i == 1: sample = -stats.gamma(a = self.left_alpha, scale = self.left_beta, loc   = -self.means[0]).rvs(dist_split[i])

            elif i == 2: sample = stats.gamma(a = self.right_alpha, scale = self.right_beta, loc  = self.means[0]).rvs(dist_split[i])
            
            else: sample = np.random.normal(self.means[i], np.sqrt(self.vars[i]), dist_split[i])
                
            samples = np.append(samples, sample)

        samples = np.array(samples)

        if plot:
            
            sns.distplot(samples, bins = 100, label = 'Sampled')
            sns.distplot(self.X, bins = 100, label = 'Original')
            plt.legend()
        
        return samples
    
    def print_params(self):
            
        print(r'Left Gamma:       alpha = {} and beta    = {}, mean = {},  comp_weight = {}'.format(np.round(self.left_alpha, 3), np.round(self.left_beta, 3), np.round(self.means[1], 3), self.comp_weights[1]))
        print()
        print(r'Right Gamma:      alpha = {} and beta    = {}, mean = {},  comp_weight = {}'.format(np.round(self.right_alpha, 3), np.round(self.right_beta, 3), np.round(self.means[2], 3), self.comp_weights[2]))
        print()
        print(r'Central Gaussian: mu    = {} and sigma^2 = {},               comp_weight = {}'.format(np.round(self.means[0], 3), np.round(self.vars[0], 3), self.comp_weights[0]))