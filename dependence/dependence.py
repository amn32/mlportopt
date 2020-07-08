import ot
import sys
import scipy
import numpy             as np
import matplotlib.pyplot as plt
import scipy.stats       as stats

from sklearn.metrics     import mutual_info_score
from tqdm.notebook       import tqdm
from copulae             import (NormalCopula, ClaytonCopula, EmpiricalCopula, GumbelCopula, StudentCopula)

from mlportopt.mixturemodels.mixturemodels       import *
from mlportopt.preprocessing.preprocessing       import *

class Dependence:
    
    def __init__(self, data, n_samples = 1000, rmt_denoise = None):
        
        self.data        = data
        
        self.n, self.m   = self.data.shape
        
        self.rmt_denoise = rmt_denoise
        
        self.n_samples   = n_samples # used for MC simulations in various metrics
        
        self.copula_dict = {'deheuvels': None, 'gaussian':NormalCopula, 'student':StudentCopula, 'clayton': ClaytonCopula, 'gumbel':GumbelCopula}
        
        self.sim_measures  = ['CE', 'corr', 'VI', 'MI']
        self.dist_measures = ['Wasserstein', 'CD']
        
        self.copulas       = list(self.copula_dict.keys())
        
    def MIcoeff(self, x, y):
    
        n    = x.shape[0]

        corr = np.corrcoef(x, y)[0,1]

        bins = int(np.round(0.5*np.sqrt(2)*np.sqrt(1 + np.sqrt(1+(24*n)/(1 - corr**2))), 0))

        entropy_x = stats.entropy(np.histogram(x, bins)[0])
        entropy_y = stats.entropy(np.histogram(y, bins)[0])

        MI   = mutual_info_score(None, None, contingency = np.histogram2d(x, y, bins)[0])

        MI /= min(entropy_x, entropy_y)

        return MI

    def VIcoeff(self, x, y):

        n    = x.shape[0]

        corr = np.corrcoef(x, y)[0,1]

        bins = int(np.round(0.5*np.sqrt(2)*np.sqrt(1 + np.sqrt(1+(24*n)/(1 - corr**2))), 0))

        entropy_x = stats.entropy(np.histogram(x, bins)[0])
        entropy_y = stats.entropy(np.histogram(y, bins)[0])

        MI   = mutual_info_score(None, None, contingency = np.histogram2d(x, y, bins)[0])

        VI   = entropy_x + entropy_y - 2*MI

        VI  /= entropy_x + entropy_y - MI

        return VI
    
    def CEcoeff(self, X,Y):
    
        n     = X.shape[0]

        corr  = np.corrcoef(X, Y)[0,1]

        bins  = int(np.round(0.5*np.sqrt(2)*np.sqrt(1 + np.sqrt(1+(24*n)/(1 - corr**2))), 0))

        Xrank = stats.rankdata(X,'ordinal') / len(X)
        Yrank = stats.rankdata(Y,'ordinal') / len(Y)

        empirical = np.histogram2d(Xrank, Yrank, bins = bins, density = True)[0]

        frequency = np.histogram2d(Xrank, Yrank, bins = bins, density = False)[0]

        bin_area  = (frequency/(empirical+1e-9)/n)[0,0]

        entropy_x = stats.entropy(np.histogram(X, bins)[0])
        entropy_y = stats.entropy(np.histogram(Y, bins)[0])

        c   = empirical.flatten() + 1e-9

        CE  = sum(c*np.log(c)*bin_area)

        CE /= min(entropy_x, entropy_y)

        return CE
    
    def CDcoeff(self, X, Y, copula = None):
        
        if copula is not None:
            
            self.copula = copula
        
        np.random.seed(0)

        copula        = self.get_copula(X, Y)
        
        n             = copula.shape[0]
        initial_probs = np.ones(n)/n
        
        p_copula  = np.array((np.arange(n), np.arange(n))).T/n
        r_copula  = np.array((np.random.uniform(size = n), np.random.uniform(size = n))).T

        c_probs   = initial_probs.copy()
        p_probs   = initial_probs.copy()
        r_probs   = initial_probs.copy()

        p_dist    = ot.dist(copula, p_copula) #Ground distance matrix between locations (just sq - euclidean)
        r_dist    = ot.dist(copula, r_copula)

        p_tport   = ot.emd(c_probs, p_probs, p_dist) # Optimal Transport Distance matrix
        r_tport   = ot.emd(c_probs, r_probs, r_dist)

        m1        = np.trace(np.dot(np.transpose(p_tport), p_dist))
        m2        = np.trace(np.dot(np.transpose(r_tport), r_dist))

        return 1 - m1/(m1+m2)
    
    def get_copula(self, X, Y):
        
        if self.copula == 'deheuvels':
        
            X_     = stats.rankdata(X) / len(X)
            Y_     = stats.rankdata(Y) / len(Y)
            
            copula    = np.array((X_,Y_)).T
            
            self.stored_copula = copula.copy()
            
        else:
            
            if self.copula == 'student':
            
                cop = self.copula_dict[self.copula](2, 3)
            
            else:
            
                cop = self.copula_dict[self.copula](2)
        
            cop.fit(np.array((X,Y)).T)

            self.cop_params = {self.copula:cop.params}

            samples = cop.random(self.n_samples)

            X_, Y_  = samples[:,0], samples[:,1]

            copula  = samples.copy()
            
            self.stored_copula = copula.copy()
        
        return copula
    
    def corrcoef(self, x, y):

        return np.corrcoef(x, y)[0, 1]
    
    def GMMsample(self, x, y):
        
        x = x.reshape(-1,1)
        y = y.reshape(-1,1)
        
        gmm1 = GMM(x, clusters  = 3,      epochs      = 100, 
                    initialisation  ='Fixed', bias        = 0.2, 
                    constrained     = True,   constraints = [0.2, 0.6, 0.2])
        
        gmm2 = GMM(y, clusters  = 3,      epochs      = 100, 
                    initialisation  ='Fixed', bias        = 0.2, 
                    constrained     = True,   constraints = [0.2, 0.6, 0.2])
        
        preds1  = gmm1.fit().predict(x)
        preds2  = gmm2.fit().predict(y)
        
        sample1 = gmm1.sample(1000, plot = False)
        sample2 = gmm2.sample(1000, plot = False)
        
        return sample1, sample2
    
    def wasserstein_dist(self, sam1, sam2):
        
        return stats.wasserstein_distance(sam1, sam2)
    
    def GMM_wass(self, x, y):
        
        sam1, sam2 = self.GMMsample(x,y)
        
        return self.wasserstein_dist(sam1, sam2)
    
    def fit(self, metric = 'MI', copula = None, distance = True, condensed = True):
        
        if metric == 'corr': 
            
            sim = np.corrcoef(self.data)
            
            if distance: 
            
                return self.to_dist(sim, condensed)
            
            else: 
            
                return sim 
        
        self.copula = copula
        n           = self.data.shape[0]
        sim         = np.empty((n,n))
        
        if metric == 'MI':   func = self.MIcoeff
        if metric == 'VI':   func = self.VIcoeff
        if metric == 'CE':   func = self.CEcoeff
        if metric == 'CD':   func = self.CDcoeff
        if metric == 'Wasserstein': func = self.GMM_wass
        

        for i in range(n):
            
            for j in range(n):
                
                print(f'Estimating dependence measure for asset {i} and asset {j} out of {n} x {n}\r', end="")
                
                if i == j: 
                    
                    sim[i,j] = 1
                
                elif j>i:
                    
                    sim[i,j] = func(self.data[i,:],self.data[j,:])
                    
                else:
                    
                    sim[i,j] = sim[j,i]
                    
        if self.rmt_denoise is not None and metric in self.sim_measures:
            
            rmt  = RMT(sim, ismat = True)
            rmt.optimise(q = self.m/self.n, verbose = True)
            cov, sim = rmt(self.rmt_denoise)
            
            for i in range(n):
                for j in range(n):
                
                    if i == j: 

                        sim[i,j] = 1

        if metric in self.dist_measures:
                        
            return self.to_dist(sim, condensed, tform = False)
                        
        elif distance: 
            
            return self.to_dist(sim, condensed)
        
        else: 
            
            return sim   
        
    def to_dist(self, similarity_matrix, condensed = True, tform = True):
        
        dist = similarity_matrix.copy()
        
        if tform: dist = np.sqrt(0.5*(1 - similarity_matrix))
        
        n    = dist.shape[0]
        
        for i in range(n):
            for j in range(n):

                if i == j: 

                    dist[i,j] = 0
                    
                if i < j:
                    
                    dist[i,j] = dist[j,i]
        
        if condensed:

            return scipy.spatial.distance.squareform(dist)
        
        else:
        
            return dist
    
    def pair_summary(self, X, Y):
        
        dep_dict = {}
        
        sub_dep  = Dependence(np.array((X,Y)))
        
        dep_dict['Pearson Correlation']       = sub_dep.fit(metric = 'corr', dist = False)[0,1]
        dep_dict['Variation of Information']  = sub_dep.fit(metric = 'VI', dist = False)[0,1]
        dep_dict['Mutual Information']        = sub_dep.fit(metric = 'MI', dist = False)[0,1]
        dep_dict['Copula Entropy']            = sub_dep.fit(metric = 'CE', dist = False)[0,1]
        
        for i in self.copulas:
            
            dep_dict[f'Copula OTD - {i}'] = sub_dep.fit(metric = 'CD', copula = i, dist = False)[0,1]
            
        return dep_dict
    
    def plot_copula(self, X, Y, copula):
        
        self.copula = copula
        
        copula = self.get_copula(X, Y)
        
        plt.scatter(copula[:, 0], copula[:,1])
        
        return

