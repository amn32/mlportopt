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
    
    '''1D Gaussian Mixture Model - Optimised by expectation maximisation'''
    
    def __init__(self, data, 
                 clusters       = 3, 
                 epochs         = 10, 
                 seed           = True, 
                 initialisation = 'Kmeans',
                 bias           = 0.4,
                 constrained    = 'True',
                 constraints    = [0.1,0.8,0.1],
                 verbose        = False):
        
        '''
        - clusters: number of Gaussians to include
        - epochs:   number of iterations
        - initialisation: 'Kmeans' or 'Fixed' initialisation of the clusters. Fixed sections off tail data as a proportion (bias) of the range from the min/max.
        - bias: proportion of the range that the tail Gaussians are assigned to.
        - constrained: Boolean indicator for whether the mixing proportions should be constrained
        - constraints: Maximum mixing proportions for the tails (the centre Gaussian is assigned the difference).
        '''
        
        if seed: np.random.seed(0)
        
        self.n           = data.shape[0]
        self.m           = data.shape[1]
        self.X           = data
        self.clusters    = clusters
        self.epochs      = epochs
        self.pi          = np.zeros(self.clusters)
        self.means       = np.zeros(self.clusters)
        self.vars        = np.zeros(self.clusters)
        self.weights     = np.zeros((self.n, self.clusters))
        self.constrained = constrained
        self.constraints = constraints
        self.bias        = bias
        self.n_params    = self.clusters*3 - 1
        self.verbose     = verbose
        
        ### Initialising the clusters ###
        
        if initialisation == 'Kmeans':
        
            self.preds    = KMeans(self.clusters).fit(self.X).predict(self.X)
            
        elif initialisation == 'Fixed':
            
            self.preds = np.zeros(self.X.shape)
            
            min_val    = np.min(self.X)
            max_val    = np.max(self.X)
            range_     = max_val - min_val
            
            self.left_val   = min_val + self.bias*range_
            self.right_val  = max_val - self.bias*range_

            self.preds[self.X < self.left_val]  = 1      
            self.preds[self.X > self.right_val] = 2
        
        self.labels   = np.unique(self.preds)
        
        ### Initialising cluster mean and variances ###
        
        for i, item in enumerate(self.labels):
            
            km_assigned     = np.where(self.preds == item)[0]
            self.pi[i]      = len(km_assigned)/self.n
            self.means[i]   = np.mean(self.X[km_assigned, :], axis = 0)
            self.vars[i]    = np.var((self.X[km_assigned, :] - self.means[i]))*self.pi[i]
         
        ### Constraining initialisation ###
        
        if self.constrained: self.reparam_pi()

        return
    
    def reparam_pi(self):
        
        '''Limits the tail mixing proportions with the remainder given to the central dist'''
        
        left   = np.where(self.labels == 1)[0][0]
        right  = np.where(self.labels == 2)[0][0]
        middle = np.where(self.labels == 0)[0][0]

        if self.pi[left]  > self.constraints[0]:
            
            self.pi[left] = self.constraints[0]
        
        if self.pi[right]  > self.constraints[2]:
            
            self.pi[right] = self.constraints[2]
            
        self.pi[middle] = 1 - self.pi[left] - self.pi[right]
                    
    def calculate_expectation(self):
        
        ''' Calculate the likelihood of the data under the Gaussians '''
        
        if self.constrained: self.reparam_pi()
            
        for j, jitem in enumerate(range(self.clusters)):
            
            pdf                = stats.multivariate_normal(self.means[j], self.vars[j], allow_singular=True).pdf(self.X).flatten()

            self.weights[:, j] = self.pi[j] * pdf

        self.weights /= self.weights.sum(axis = 1)[:,None]
        
        return
        
    def optimise(self):
        
        ''' Optimisation of the Gaussian parameters '''
         
        self.pi = np.mean(self.weights, axis = 0)

        if self.constrained: self.reparam_pi()

        self.means = (self.weights.T @ self.X) / np.sum(self.weights, axis = 0)[:,None]
        
        if np.any(self.vars < 1e-6):
            
            return #This might be a bit of a hack - discuss with Tom

        for k, kitem in enumerate(range(self.clusters)):

            new_demeaned = self.X - self.means[k]

            self.vars[k] = (new_demeaned.T @ (self.weights[:,k]*np.eye(self.weights[:,k].shape[0])) @ new_demeaned)/np.sum(self.weights, axis = 0)[k]        
    
        return
        
    def fit(self, plot = False):
        
        for epoch in range(self.epochs):
            
            if self.verbose: print(f'Epoch {epoch} out of {self.epochs}\r', end="")
            
            self.calculate_expectation()
            self.optimise()
            
        if plot: sns.distplot(self.X, bins = 100, label = 'Original')
            
        return self
    
    def print_params(self):
                    
        for i in range(self.clusters):

            print(r'Gaussian {}: mu    = {} and sigma^2 = {}, pi = {}'.format(i, np.round(self.means[i][0], 3), np.round(self.vars[i], 3), self.pi[i]))
            print()
                
    def predict(self, X):
        
        preds = [self.pi[i] * stats.multivariate_normal(self.means[i], self.vars[i], allow_singular=True).pdf(X).flatten() for i in range(self.clusters)]

        return np.argmax(preds, axis = 0)
    
    def sample(self, n_samples = 1000, plot = True):
        
        ''' Sample from our fitted mixture model and plot the histogram of samples '''
        
        samples    = []
        
        dist_split = np.random.multinomial(n_samples, self.pi)
        
        for i in range(self.clusters):
        
            sample = np.random.normal(self.means[i], np.sqrt(self.vars[i]), dist_split[i])
            
            samples = np.append(samples, sample)

        samples = np.array(samples)
            
        if plot:
            
            sns.distplot(samples, bins = 100, label = 'Sampled')
            sns.distplot(self.X, bins = 100, label = 'Original')
            plt.legend()
        
        return samples
    
##############################################################################################################################
        