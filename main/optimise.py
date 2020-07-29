import sys
import numpy as np
import matplotlib.pyplot as plt

from mlportopt.util.helperfuncs            import gen_real_data, train_test, merge_clusters, get_full_weights
from mlportopt.preprocessing.preprocessing import preprocess
from mlportopt.flatcluster.flatcluster     import DPGMM, TFSOM, GPcluster
from mlportopt.hiercluster.hiercluster     import scipy_linkage, BHC
from mlportopt.portfolio.portfolio         import Allocate, Evaluation, quasidiagonalise
from mlportopt.dependence.dependence       import Dependence

class Optimise:
    
    def __init__(self, data, 
                 train_test_v = 0.5,
                 frequency    = 'W',
                 residuals    = None, 
                 whiten       = True, 
                 reduce_dims  = 2,
                 dep_measure  = 'MI',
                 dist_measure = None,
                 dep_copula   = 'deheuvels',
                 dep_denoise  = None,
                 flat_cluster = {'DPGMM':{'clusters':5, 
                                          'iters': 500,
                                          'step': 1,
                                          'alpha': 1, 
                                          'ftol': 1e-6, 
                                          'gtol': 1e-6,  
                                          'kap': 1e-6, 
                                          'var_p': 1e-3, 
                                          'conc': 10}},
                 hier_cluster = {'Bayesian':{'alpha': 0.001,
                                             'beta': 0.001,
                                             'gamma': 1,
                                             'rmt_denoise':  None}},
                 intra_method = 'var',
                 inter_method = 'var',
                 plots        = True):
        
        '''
        Parameters
        ----------
        train_test_v: float
            Ratio of split between training and test data (Default is 0.5)
        frequency: str
            Frequency of data [Options: 'D', 'W', 'M'] (Default is 'W')
        residuals: ndarray
            Beta-adjusted returns (Regressed on the market). If not None, clustering is performed on these residuals. (Default is None)
        whiten: Bool
            Boolean indicator for demeaning and standardising (whitening) (Default is True)
        reduce_dims: int or None
            If not None, target data will be reduced via PCA to a lower dimension of size reduce_dims (Default is 2)
        dep_measure: str
            Chosen dependence measure [Options: 'MI', 'VI','CE','CD','corr','Waserstein'] (Default is 'MI')
        dist_measure: str or None
            If not None, the method for transforming a similarity matrix into a distance matrix [Options: 'angular', 'abs_angular', 'sq_angular'] (Default is None)
        dep_copula: str
            Chosen dependence copula [Options: 'deheuvels', 'gaussian','student','clayton','gumbel'] (Default is None)    
        dep_denoise: str or None
            If not None, the Random Matrix Theoretical approach to denoising Hermitian matrices [Options 'fixed','shrinkage','targeted_shrinkage'] (Default is None)
        flat_cluster: None or Nested Dictionary (see below for parameter descriptions)
            Parameter Dictionary for flat clustering of form {'Method':{Parameters}}
            [Options: {'DPGMM': {clusters, iters, step, alpha, ftol, gtol, kap, var_p, conc, verb}}
                      {'GP'   : {iters, step, s2, l, alpha, gamma, cov_lim, p_Num, latent, verbose}}]
        hier_cluster: Nested Dictionary
            Parameter Dictionary for hierarchical clustering of form {'Method':{Parameters}}
            [Options: {'Bayesian': {alpha, beta, gamma, rmt_denoise}}
                      {'single':   {dep_measure, hier_cluster, dist_measure, dep_copula, dep_denoise}}
                      {'average':  {Same as single}},
                      {'complete': {Same as single}},
                      {'ward':     {Same as single}}]
        intra_method: str
            Method for (inversely) weighting at the cluster level - see Risk Options below (Default is 'var')
        inter_method: str
            Method for (inversely) weighting at the tree level - see Risk Options below (Default is 'var')
        
        Risk Options
        -------
        - uniform
        - prob_sharpe
        - ann_sharpe
        - sharpe
        - var
        - vol (std)
        - ann_vol
        - VaR - normal  (VaR under normality assumption)
        - VaR - student (VaR under student t assumption)
        - VaR - gmm     (VaR from fitted GMM samples)
        - CVaR - normal  (CVaR under normality assumption)
        - CVaR - student (CVaR under student t assumption)
        - CVaR - gmm     (CVaR from fitted GMM samples)
        
        flatcluster - GP
        ----------
        iters: int
            Number of iterations in the graident descent method (Default is 10)
        step: float
            Step size in gradient descent method (Default is 0.1)
        s2: float
            Initial value for the variance (Default is 1)
        l: float
            Initial value for the lengthscale (Default is 1)
        alpha: float
            Initial value for alpha (Default is 1)
        gamma: float [0,1]
            Controls the proportion of the maximum variance we want to determine cluster boundaries. (Default is 1)
        cov_lim: float
            Limits the maximum covariance (Default is 0.99)
        p_Num:
            Determines the number of samples to be generated when fitting (Default is 20)
        latent: int
            Determines the size of the latent space if dimensionality reduction is required (Default is 3)
        verbose: Bool
            Boolean indicator for descriptive printing (Default is False)
        
        flatcluster - DPGMM
        ----------
        X: ndarray
        clusters: int
            Initial number of clusters - updated by the DP prior
        iters: int
            Number of iterations in gradient descent (Default is 500)
        step: float
            Step size (Default is 1)
        alpha: float
            Initial value for the Dirichlet hyper-parameter (Default is 1)
        ftol: float
            Tolerance for function value convergence (Default is 1e-6)
        gtol: float
            Tolerance for gradient value convergence (Default is 1e-6)
        kap: float
            Hyperparameter for prior mean (Default is 1e-6)
        var_p: float
            Prior value for the variance (Default is 1e-3)
        trunc: float
            Intial value for the truncation parameter (Default is 10)
        verb: Bool
            Boolean indicator for explanatory prints (Default is False)
        
        '''
        
        self.data             = data
        
        self.train, self.test = train_test(data, train_test_v)

        self.frequency        = frequency
        
        self.whiten           = whiten
        
        ######## Miscellaneous ###########
        
        self.residuals     = False
        self.merge_weights = None
        self.plots         = plots
        self.reduce_dims   = reduce_dims
        
        self.hier_cluster_dict = hier_cluster
        
        ######## Preprocessing ###########
        
        if residuals is not None: 
            
            self.residuals = True # Set the target data to the residuals if beta-adjustment is desired  
            self.X, _      = train_test(residuals, train_test_v)
        
        else:  self.X = self.train.copy()
            
        # Whiten and reduce
        
        tform        = self.X.copy() + 1e-7
        
        self.X       = preprocess(tform, axis = 1, white = self.whiten , reduce = False, n = 0)

        self.reduced = preprocess(tform, axis = 1, white = self.whiten , reduce = (reduce_dims > 0), n = reduce_dims)

        ########  Dependence  ############
        
        self.dep_measure  = dep_measure
        self.dist_measure = dist_measure
        self.dep_copula   = dep_copula
        self.dep_denoise  = dep_denoise
        
        ###### Cluster Risk Metrics ######
        
        self.intra_method = intra_method
        self.inter_method = inter_method

        ######## Flat Cluster ############
        
        if flat_cluster is not None:
        
            self.flat_cluster = list(flat_cluster.keys())[0]
            
        else:
            
            self.flat_cluster = flat_cluster
        
        if self.flat_cluster == 'DPGMM':
            
            param_dict = flat_cluster['DPGMM']
            
            clusters   = param_dict['clusters']
            iters      = param_dict['iters']
            step       = param_dict['step'] 
            alpha      = param_dict['alpha']
            ftol       = param_dict['ftol'] 
            gtol       = param_dict['gtol'] 
            kap        = param_dict['kap']
            var_p      = param_dict['var_p'] 
            trunc      = param_dict['trunc']
            verb       = False
            
            self.fclust = DPGMM(self.reduced, clusters, iters, step, alpha, ftol, gtol, kap, var_p, trunc, verb)
            
        elif self.flat_cluster == 'GP':
            
            param_dict = flat_cluster['GP']
            
            iters   = param_dict['iters']
            step    = param_dict['step']
            s2      = param_dict['s2']
            l       = param_dict['l']
            alpha   = param_dict['alpha'] 
            gamma   = param_dict['gamma'] 
            cov_lim = param_dict['cov_lim']
            p_Num   = param_dict['p_Num']
            latent  = param_dict['latent'] 
            verbose = param_dict['verbose']
            
            self.fclust = GPcluster(self.reduced,iters, step, s2, l, alpha, gamma, cov_lim, p_Num, latent, verbose)

        return
    
    def param_hclust(self):

        ######### Hier Cluster ##########
        
        self.hier_cluster = list(self.hier_cluster_dict.keys())[0]
        
        param_dict  = self.hier_cluster_dict[self.hier_cluster]
        
        if self.hier_cluster == 'Bayesian':
            
            if self.reduce_dims < 2:
                
                print('Please reduce the dimensionality before attempting Bayesian Hierarchical Clustering')

            alpha       = param_dict['alpha']
            beta        = param_dict['beta']
            gamma       = param_dict['gamma'] 
            rmt_denoise = param_dict['rmt_denoise']
            
            self.hclust = BHC(self.reduced, alpha, beta, gamma, rmt_denoise)
            
        else:
        
            self.hclust = scipy_linkage(self.X, self.dep_measure, self.hier_cluster, self.dist_measure, self.dep_copula, self.dep_denoise)         
        
            return
        
    def f_cluster(self):
        
        ### Model Settings ###

        if self.flat_cluster == 'DPGMM': 
            
            self.fclust.fit()

            if self.plots: self.fclust.plot()
            self.fclust.split_all()
            if self.plots: self.fclust.plot()
            

#         elif self.flat_cluster == 'SOM':
            
#             self.fclust.train(self.X).fit(self.X)

        elif self.flat_cluster == 'GP':
            
            self.fclust.fit()
            
        ### Assign Clusters ###
            
        self.assigned_clusters  = self.fclust.assigned_clusters  
        
        self.unique_flat_clusts = np.unique(self.assigned_clusters).shape[0]

        ### Merge the clusters weighted by chosen metric to create new data for hierarchical clustering ###
        
        if self.residuals:

            _, self.X, self.merge_weights = merge_clusters(data      = self.train, 
                                                           clusters  = self.assigned_clusters,
                                                           resids    = self.X,
                                                           freq      = self.frequency,
                                                           method    = self.intra_method)
            
            self.X = preprocess(self.X, axis = 1, white = self.whiten , reduce = False, n = 0)
            
        else:

            self.X, _, self.merge_weights = merge_clusters(data      = self.train, 
                                                           clusters  = self.assigned_clusters, 
                                                           resids    = None,
                                                           freq      = self.frequency,
                                                           method    = self.intra_method)
            
            self.X  = preprocess(self.X, axis = 1, white = self.whiten , reduce = False, n = 0)
        
        return

    def h_cluster(self):
        
        self.param_hclust()

        if self.hier_cluster == 'Bayesian': self.hclust.fit()

        if self.plots: self.hclust.plot_dendrogram()

        self.linkage = np.array(self.hclust.linkage)

        return

    def allocate(self):

        self.allocation = Allocate(self.train, self.linkage, self.frequency)

        self.weights    = self.allocation.recursively_partition(inter_cluster_metric = self.inter_method, 
                                                                intra_cluster_metric = self.intra_method)

        
        if self.merge_weights is not None:

            self.weights = get_full_weights(self.weights, self.merge_weights)

        return

    def evaluate(self, plots = False):

        self.evaluation = Evaluation(self.train, self.test, self.weights, self.frequency)

        self.evaluation(plots)

        return

    def __call__(self):

        if self.flat_cluster is not None:

            self.f_cluster()

        self.h_cluster()
        self.allocate()
        self.evaluate(self.plots)
        self.summary_df = self.evaluation.summary_df

        return