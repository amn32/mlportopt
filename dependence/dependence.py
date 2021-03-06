import ot
import sys
import scipy
import numpy             as np
import matplotlib.pyplot as plt
import matplotlib.pylab  as pl
import scipy.stats       as stats
import seaborn           as sns

from sklearn.metrics     import mutual_info_score
from tqdm.notebook       import tqdm
from copulae             import (NormalCopula, ClaytonCopula, EmpiricalCopula, GumbelCopula, StudentCopula)

from mlportopt.mixturemodels.mixturemodels       import *
from mlportopt.preprocessing.preprocessing       import *

class Distance:
    
    '''
    Takes a similarity matrix and returns a distance matrix
    '''
    
    def __init__(self, similarity_matrix, dist_measure = 'angular'):

        '''
        Parameters
        ----------
        similarity_matrix: ndarray
        dist_measure: str
            The desired function that maps similarity (-1,1) to distance (0,1) (Default is 'angular') 
            [Options: 'angular', 'corr_dist', 'acute_angular', 'abs_corr', 'other' (Uses set metric for measure or if already distance, does not transform)]
        '''
        
        self.sim          = similarity_matrix
        self.dist_measure = dist_measure

        ### Distance Functions ###
        
        self.angular_dist     = lambda sim: np.arccos(sim)
        self.corr_dist        = lambda sim: np.sqrt(0.5*(1-sim))
        self.acute_angular    = lambda sim: 0.5*np.pi - np.abs(0.5*np.pi - np.arccos(sim))
        self.abs_corr_dist    = lambda sim: np.sqrt(1-sim**2)
        
        self.null_transform   = lambda sim: sim
        
        self.valid_distances  = {'angular':self.angular_dist,
                                 'corr_dist':self.corr_dist,
                                 'acute_angular':self.acute_angular, 
                                 'abs_corr':self.abs_corr_dist,
                                 'other': self.null_transform}
        
        return
    
    def __call__(self, condensed = False):
        
        '''
        Parameters
        ----------
        condensed: Bool
            Boolean indicator to determine whether a condensed matrix is returned
        '''
        
        dist = self.valid_distances[self.dist_measure](self.sim)
        
        for i in range(dist.shape[0]):
            for j in range(dist.shape[0]):
                
                if   i == j: dist[i,j] = 0
                elif  i < j: dist[i,j] = dist[j,i] 
                    
        if condensed:
            
            return scipy.spatial.distance.squareform(dist)
        
        else:
        
            return dist

class Dependence:
    
    '''
    Compute a similarity (or distance) matrix
    
    Methods
    -------
    
    MICoeff(x, y)
        Calculate the (normalised) mutual information coefficient between x and y
    VICoeff(x,y)
        Calculate the (normalised) variation of information coefficient between x and y
    CEcoeff(x,y)
        Calculate the entropy of the empricial copula of x and y. Theoretically approximate to MI.
    CDcoeff(X, Y, copula = None)
        Calculate the optimal transport solution between a fitted copula and two reference copulas (Random and Positive Dependence), normalised to a coefficient.
    get_copula(X,Y)
        Calculate the copula of x and y. If copula specified in fit() is not the empricial copula, a copula is estimated and samples returned. Else the empricial (deheuvels) copula is returned.
    corrcoef(x,y)
        Calaculate the Pearson correlation coefficient
    GMM_sample(x,y)
        Fit Gaussian Mixture Models to x and y and return samples from the fitted distributions
    wasserstein_dist(sam1, sam2)
        Compute the Wasserstein distance between the two samples
    GMM_wass(x, y)
        Fit GMMs, sample and return the Wasserstein distance between the two distributions
    fit(metric = 'MI', copula = None, dist_measure = None, condensed = True)
        Fit method
    to_dist(similarity_matrix, dist_measure, condensed = True, tform = True)
        Map similairity matrix to distance matrix
    pair_summary(X, Y)
        Calculate the pairwise dependence for all metrics
    plot_copula(X,Y,copula)
        Plot the fitted copula
    '''
    
    
    def __init__(self, data, n_samples = 1000, rmt_denoise = None, bins = None):
        
        '''
        Parameters
        ----------
        data: ndarray
        n_samples: int
            Number of samples to generate in mixture model and copula methods (Default is 1000)
        rmt_denoise: str
            If not None, the Random Matrix Theoretical approach to clean similarity matrices [Options 'fixed','shrinkage','targeted_shrinkage'] (Default is None)
        bins: int
            Number of bins to use in density estimation. If None uses Hacine-Gharbi optimal number (Default is None)
        '''
        
        if len(data.shape) == 1:
            
            data = data.reshape(-1,1)
        
        self.data        = data
        
        self.n, self.m   = self.data.shape
        
        self.rmt_denoise = rmt_denoise
        
        self.n_samples   = n_samples
        
        self.bins        = bins
        
        self.copula_dict = {'deheuvels': None, 
                            'gaussian':  NormalCopula, 
                            'student':   StudentCopula, 
                            'clayton':   ClaytonCopula, 
                            'gumbel':    GumbelCopula}
        
        self.sim_measures  = ['CE', 'corr', 'VI', 'MI']
        self.distances     = ['Wasserstein', 'CD']
        
        self.copulas       = list(self.copula_dict.keys())
        
    def MIcoeff(self, x, y):
    
        n    = x.shape[0]

        corr = np.corrcoef(x, y)[0,1]
        
        bins = int(np.round(0.5*np.sqrt(2)*np.sqrt(1 + np.sqrt(1+(24*n)/(1 - corr**2))), 0))

        if self.bins is not None: bins = self.bins
        
        entropy_x = stats.entropy(np.histogram(x, bins)[0])
        entropy_y = stats.entropy(np.histogram(y, bins)[0])

        MI   = mutual_info_score(None, None, contingency = np.histogram2d(x, y, bins)[0])
        
        original_MI = MI
        
        MI_dist     = 1-MI/max(entropy_x, entropy_y)
        
        MI /= min(entropy_x, entropy_y)

        return [MI, MI_dist]

    def VIcoeff(self, x, y):

        n    = x.shape[0]

        corr = np.corrcoef(x, y)[0,1]
        
        bins = int(np.round(0.5*np.sqrt(2)*np.sqrt(1 + np.sqrt(1+(24*n)/(1 - corr**2))), 0))

        if self.bins is not None: bins = self.bins
        
        entropy_x = stats.entropy(np.histogram(x, bins)[0])
        entropy_y = stats.entropy(np.histogram(y, bins)[0])

        MI   = mutual_info_score(None, None, contingency = np.histogram2d(x, y, bins)[0])

        VI   = entropy_x + entropy_y - 2*MI

        VI  /= entropy_x + entropy_y - MI

        return [VI, VI]
    
    def CEcoeff(self, x, y):
    
        n     = x.shape[0]

        corr  = np.corrcoef(x, y)[0,1]
        
        bins  = int(np.round(0.5*np.sqrt(2)*np.sqrt(1 + np.sqrt(1+(24*n)/(1 - corr**2))), 0))

        if self.bins is not None: bins = self.bins
        
        Xrank = stats.rankdata(x,'ordinal') / len(x)
        Yrank = stats.rankdata(y,'ordinal') / len(y)
        
        empirical = np.histogram2d(Xrank, Yrank, bins = bins, density = True)[0]

        frequency = np.histogram2d(Xrank, Yrank, bins = bins, density = False)[0]

        bin_area  = (frequency/(empirical+1e-9)/n)[0,0]

        entropy_x = stats.entropy(np.histogram(x, bins)[0])
        entropy_y = stats.entropy(np.histogram(y, bins)[0])

        c   = empirical.flatten() + 1e-9

        CE  = sum(c*np.log(c)*bin_area)
        
        original_CE = CE
        
        CE_dist     = 1 - CE/max(entropy_x, entropy_y)

        CE /= min(entropy_x, entropy_y)

        return [CE, CE_dist]
    
    def CDcoeff(self, X, Y, copula = None):
        
        '''
        Parameters
        ----------
        copula: str
            Chosen copula [Options: 'deheuvels', 'gaussian','student','clayton','gumbel'] (Default is None)
        '''
        
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
        
        if m1<1e-3: m1 = 0

        return [m1/(m1+m2), m1/(m1+m2)]
    
    def get_copula(self, X, Y):
        
        if self.copula == 'deheuvels':
        
            X_     = stats.rankdata(X, 'ordinal') / len(X)
            Y_     = stats.rankdata(Y, 'ordinal') / len(Y)
            
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
        
        gmm1 = GMM(x).fit()
        
        gmm2 = GMM(y).fit()

        sample1 = gmm1.sample(1000, plot = False)
        sample2 = gmm2.sample(1000, plot = False)
        
        return sample1, sample2
    
    def GMM_wass(self, x, y):
        
        sam1, sam2 = self.GMMsample(x,y)
        
        return [stats.wasserstein_distance(sam1, sam2), stats.wasserstein_distance(sam1, sam2)]
    
    def fit(self, metric = 'MI', copula = None, dist_measure = None, condensed = True):
        
        '''
        Parameters
        ----------
        metric: str
            Chosen metric [Options: 'MI', 'VI', 'CE', 'CD', 'corr', 'Wasserstein'] (Default is 'MI')
        copula: str
            Chosen copula [Options: 'deheuvels', 'gaussian','student','clayton','gumbel'] (Default is None)
        dist_measure: str
            Chosen distance measure [Options: 'angular', 'abs_angular', 'sq_angular', 'MI', 'CE'] (Default is None)
        condensed: 
            Boolean indicator to determine whether a condensed matrix is returned if dist_measure not None (Default is True)
        '''
        
        n           = self.data.shape[0]
        
        ########### Vectorised version for correlation (Using Numpy) #########
        
        if metric == 'corr': 

            sim = np.corrcoef(self.data)
            
        ########### For all the other similarity measures ####################    
        
        else:
        
            self.copula = copula
            sim         = np.empty((n,n))
            distance_c  = np.empty((n,n))

            if metric == 'MI':          func = self.MIcoeff
            if metric == 'VI':          func = self.VIcoeff
            if metric == 'CE':          func = self.CEcoeff
            if metric == 'CD':          func = self.CDcoeff
            if metric == 'Wasserstein': func = self.GMM_wass
        

            for i in range(n):
                for j in range(n):

                    print(f'Estimating dependence measure for asset {i} and asset {j} out of {n} x {n}\r', end="")

                    if i == j:  
                        
                        sim[i,j]        = 1
                        distance_c[i,j] = 0
                        
                    elif j>i:   
                        
                        sim_list         = func(self.data[i,:],self.data[j,:])
                        sim[i,j]         = sim_list[0]   
                        distance_c[i,j]  = sim_list[1] 
                    
                    else:       
                        
                        sim[i,j]         = sim[j,i]
                        distance_c[i,j]  = distance_c[j,i]
                    
        ############# Denosiing methods of the constructed similarity matrix ############
                    
        if self.rmt_denoise is not None and metric in self.sim_measures:
            
            rmt  = RMT(sim, ismat = True)
            rmt.optimise(q = self.m/self.n, verbose = True)
            cov, sim = rmt(self.rmt_denoise)
            
            for i in range(n):
                for j in range(n):
                
                    if i == j: sim[i,j] = 1
 
        ############# Transforming (if not already) to a distance metric ################
        
        if dist_measure is not None: 
            
            if dist_measure == 'other':

                return self.to_dist(distance_c, dist_measure, condensed, True)
            
            else:

                return self.to_dist(sim, dist_measure, condensed, True)

        else: 

            return sim 
        
    def to_dist(self, similarity_matrix, dist_measure, condensed = True, tform = True):
        
        '''
        Parameters
        ----------
        similarity_matrix: ndarray
        dist_measure: str
            Chosen distance measure [Options: 'angular', 'corr_dist', 'acute_angular', 'abs_corr']
        condensed: Bool
            Boolean indicator to determine whether a condensed matrix is returned if dist_measure not None (Default is True)
        tform: Bool
            Boolean indicator to determine whether to transform to a distance measue (Default is True)
        '''
        
        dist = similarity_matrix.copy()
        
        if tform: 
            
            dist = Distance(similarity_matrix, dist_measure)(condensed)
        
        return dist
    
    def pair_summary(self, X, Y):
        
        dep_dict = {}
        
        sub_dep  = Dependence(np.array((X,Y)))
        
        dep_dict['Pearson Correlation']       = sub_dep.fit(metric = 'corr', dist = False)[0,1]
        dep_dict['Variation of Information']  = sub_dep.fit(metric = 'VI',   dist = False)[0,1]
        dep_dict['Mutual Information']        = sub_dep.fit(metric = 'MI',   dist = False)[0,1]
        dep_dict['Copula Entropy']            = sub_dep.fit(metric = 'CE',   dist = False)[0,1]
        
        for i in self.copulas:
            
            dep_dict[f'Copula OTD - {i}'] = sub_dep.fit(metric = 'CD', copula = i, dist = False)[0,1]
            
        return dep_dict
    
    def plot_copula(self, X, Y, copula):
        
        self.copula = copula
        
        copula = self.get_copula(X, Y)
        
        plt.scatter(copula[:, 0], copula[:,1])
        
        return

