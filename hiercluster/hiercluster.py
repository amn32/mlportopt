import numpy                   as np
import scipy.stats             as stats
import matplotlib.pyplot       as plt
from   scipy.special           import gamma, gammaln
from   scipy.cluster.hierarchy import linkage, dendrogram

from mlportopt.preprocessing import *
from mlportopt.util          import *
from mlportopt.dependence    import *

import warnings
warnings.filterwarnings("ignore", message="overflow encountered in exp")

class scipy_linkage:
    
    '''Wrapper Class to hierarchcially cluster data through the scipy.linkage function, using mlportopt dependence measures'''
    
    def __init__(self, data, metric = 'corr', method = 'single', distance = 'angular', copula = None, rmt_denoise = None):
        
        '''
        Parameters
        ----------
        data: ndarray
        metric: str
            Dependence metric of choice [Options: 'MI', 'VI','CE','CD','corr','Waaserstein'] (Default is 'corr')
        method: str
            Linkage method [Options: 'single', 'average', 'complete', 'ward'] (Default is 'single')
        copula: str
            Chosen copula [Options: 'deheuvels', 'gaussian','student','clayton','gumbel'] (Default is None)
        dist_measure: str
            Chosen distance measure [Options: 'angular', 'abs_angular', 'sq_angular'] (Default is 'angular')
        rmt_denoise: str
            Boolean indicator to use Random Matrix Theory toolkit to clean similarity matrices [Options 'fixed','shrinkage','targeted_shrinkage'] (Default is None)

        '''
        self.data    = data
        self.method  = method
        
        dep          = Dependence(self.data, rmt_denoise = rmt_denoise)
        self.dist    = dep.fit(metric = metric, dist_measure = distance, condensed = True, copula = copula)
        
        self.linkage = linkage(self.dist, self.method)
        
    def plot_dendrogram(self):
        
        dn  = dendrogram(self.linkage)
        plt.show()
        
###########################################################################################################

class BHC:
    
    '''Bayesian Hierarchical Agglomerative Clustering - splits determined through likelihood tests
    
    Methods
    -------
    
    log_likelihood(x)
        Evaluate the log likelihood at x
    reinitialise()
        Reinitialise parameters after merges
    compute_odds
        Compute the (log) odds of cluster merges
    fit(reweight = True)
        Perform BHC by comparing all possible merges until all data are merged into one cluster
    plot_dend()
        Plot the fitted dendrogram
    '''
    
    def __init__(self, data, alpha = 0.001, kappa = 0.001, beta = 1, gamma = None, cov_scale = 1, rmt_denoise = None):
        
        '''
        Parameters
        ----------
        data: ndarray
        alpha: float
            Scaling factor on the precision of the Gaussian prior on the mean (Default is 0.001)
        beta: float
            Wishart prior hyperparameter (Default is 1)
        gamma: float
            Gaussian Prior on the mean (Default is None)
        kappa: float
            Clustering hyperparameter (Default is 0.001)
        rmt_denoise: str
            Boolean indicator to use Random Matrix Theory toolkit to clean similarity matrices [Options 'fixed','shrinkage','targeted_shrinkage'] (Default is None)
        '''
        
        self.X           = data
        self.n           = data.shape[0]
        self.m           = data.shape[1]
        self.alpha       = alpha
        self.beta        = beta
        self.gamma       = gamma
        self.kappa       = kappa
        self.cov_scale   = cov_scale
        self.leaf_ind    = list(np.arange(self.n))
        self.log_kappa   = np.log(kappa)
        self.rmt_denoise = rmt_denoise
        
        self.PP = []
        
        self.mu          = np.mean(self.X, axis = 0).reshape(-1,1)
        self.cov         = np.cov((self.X/self.beta), rowvar = False)
        self.n_0         = [1 for _ in range(self.n)]
        self.x_0         = [(i,) for i in range(self.n)]
        self.d_0         = [self.log_kappa for _ in range(self.n)]
        self.ml_0        = [self.log_likelihood(self.X[i,:].reshape(1,-1)) for i in range(self.n)]
        
        self.clust_pairs = []
        self.clusters    = []
        self.log_kappas  = []
        self.x           = []
        self.d           = []
        self.odds        = []
        self.probs       = []  
        self.step        = 0
        
        self.linkage     = []
              
    def log_likelihood(self, x):
        
        n, m = x.shape
        
        self.n_m          = n + m
        self.feat         = np.sum(x, axis = 0).reshape(-1,1)
        
        self.alpha_factor  = self.alpha/(self.alpha + n)
        
        self.mu_muT       = self.mu @ self.mu.T
        self.XT_X         = x.T  @ x
        
        if self.rmt_denoise is not None:
            
            rmt  = RMT(self.X, ismat = False)
            rmt.optimise(q = self.m/self.n)
            cov, corr = rmt(self.rmt_denoise)
            
            self.XT_X = cov 
            
        self.feat_featT   = self.feat @ self.feat.T
        self.mu_featT     = self.mu @ self.feat.T
        
        self.PSI          = self.cov + self.XT_X \
                            + n * self.alpha_factor * self.mu_muT \
                            - (self.alpha_factor/self.alpha) * self.feat_featT \
                            - 2 * self.alpha_factor * self.mu_featT
        
        log_like  = 0
        log_like += -(n*m/2) * np.log(2*np.pi) 
        log_like += (m/2) * np.log(self.alpha_factor)
        log_like += (m/2) * np.linalg.slogdet(self.cov)[1]
        log_like += -(self.n_m/2) * np.linalg.slogdet(self.PSI)[1]
        log_like += np.sum(gammaln(0.5*(self.n_m - np.arange(m)))) + 0.5*self.n_m*m*np.log(2)
        log_like += -np.sum(gammaln(0.5*(m - np.arange(m)))) - 0.5*m*m*np.log(2)
            
        return log_like
    
    def reinitialise(self):
        
        most_prob      = self.odds.index(max([self.odds[i] for i in self.PP]))
        most_prob_val  = self.odds[most_prob]
        most_prob_prob = 1/most_prob_val
        most_prob_1    = self.clust_pairs[most_prob][0] 
        most_prob_2    = self.clust_pairs[most_prob][1] 
        n_obs          = len(self.x[most_prob])

        self.linkage.append([most_prob_1, most_prob_2, most_prob_prob, n_obs])

        # Reinitialise with new clusters

        self.x_0.append(self.x[most_prob])
        self.d_0.append(self.d[most_prob])
        self.n_0.append(1)

        self.ml_0.append(self.probs[most_prob][0] + np.log(1+np.exp(self.probs[most_prob][1] - self.probs[most_prob][0])))

        return
    
    def compute_odds(self, i, j):
        
        x_t = self.x_0[i] + self.x_0[j]
    
        n_t = len(x_t)
        
        self.clust_pairs.append((i, j))
                
        self.x.append(self.x_0[i] + self.x_0[j])

        d_t      = self.log_kappa + gammaln(n_t) + np.log(1 + np.exp(self.d_0[i]+self.d_0[j] - self.log_kappa - gammaln(n_t)))
        
        self.d.append(d_t)
        
        data_t   = self.X[self.x[-1],:]

        log_p_1  = self.log_likelihood(data_t) + self.log_kappa + gammaln(n_t) - d_t

        log_p_2  = self.ml_0[i] + self.ml_0[j] + self.d_0[i] + self.d_0[j] - d_t

        log_odds = log_p_1 - log_p_2
                
        self.probs.append([log_p_1, log_p_2])
        
        self.odds.append(log_odds)
        
        self.PP.append(self.step)
        
        self.step     += 1
        
        return
            
    def fit(self, reweight = True):
        
        '''
        Parameters
        ----------
        reweight: Bool
            Boolean indicator to rescale the linkage matrix
        '''
            
        initial_odds = [self.compute_odds(i, j) for i in range(self.n - 1) for j in range(i+1, self.n)]
        
        tree_level   = 0
        
        while True:
            
            self.reinitialise()
            
            to_drop = self.linkage[tree_level][:2]
            
            self.leaf_ind = [_ for _ in self.leaf_ind if _ not in to_drop]
            
            if len(self.leaf_ind) == 0:
                
                break
                
            for node in self.leaf_ind:
                
                self.compute_odds(self.n+tree_level, node)
            
            self.PP = [y for y in self.PP if self.clust_pairs[y][0] not in to_drop and self.clust_pairs[y][1] not in to_drop]
            self.leaf_ind.append(self.n + tree_level)
            tree_level+=1
            
        if reweight:
            
            for i in range(len(self.linkage)):
                if self.linkage[i][2] < 0:
                    self.linkage[i][2] = 2 * max([y[2] for y in self.linkage])
                if self.linkage[i][0] > (self.n - 1):
                    self.linkage[i][2] += self.linkage[self.linkage[i][0] - self.n][2]
                if self.linkage[i][1] > (self.n - 1):
                    self.linkage[i][2] += self.linkage[self.linkage[i][1] - self.n][2]
        
        return      
        
    def plot_dendrogram(self):

        self.dend = dendrogram(self.linkage)
        
        plt.show()

        return 