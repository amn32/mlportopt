import random
import sys
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt

from mlportopt.util        import *
from mlportopt.riskmetrics import *
from mlportopt.dependence  import *
from mlportopt.hiercluster import *

def quasidiagonalise(link): 
    
    '''Quasi diagonalisation of linkage matrix as per Lopez de Prado'''

    try: link = link.astype(int)
    except: pass
    
    sorted_index   = pd.Series([link[-1,0],link[-1,1]])
    num_items      = link[-1,3]                         

    while sorted_index.max() >= num_items:

        sorted_index.index = range(0, sorted_index.shape[0]*2, 2)
        split              = sorted_index[sorted_index >= num_items]
        i                  = split.index; 
        j                  = split.values - num_items

        sorted_index[i]    = link[j,0]
        split              = pd.Series(link[j,1], index = i+1)

        sorted_index       = sorted_index.append(split)
        sorted_index       = sorted_index.sort_index()
        sorted_index.index = range(sorted_index.shape[0])

    return sorted_index.tolist()

class Allocate:
    
    '''
    Methods
    -------
    
    cluster_risk_metric(inter_cluster_metric, intra_cluster_metric, index)
        For a given cluster, calculate the intra and inter risk weightings
    recursviely_partition(inter_cluster_metric = 'var', intra_cluster_metric = 'var')
        Recursively partition to allocate weights
    '''
    
    def __init__(self, data, linkage_matrix, frequency = 'D'):
        
        '''
        Parameters
        ----------
        data: ndarray
        linkage: mat
            Linkage matrix
        frequency: str
            Frequency of data [Options: 'D', 'W', 'M'] (Default is 'W')
        '''
        
        self.data         = data
        self.n, self.m    = self.data.shape
        self.annual       = {'D':262, 'W':52, 'M': 12}
        self.freq         = self.annual[frequency]
        self.frequency    = frequency
        self.cov          = np.cov(data)
        self.ann_cov      = self.cov * self.freq
        
        self.link         = linkage_matrix
        
        self.ann_returns  = self.data.mean(1) * self.freq
        
        self.sorted_index = quasidiagonalise(self.link)
        
    def cluster_risk_metric(self, inter_cluster_metric, intra_cluster_metric, index):
        
        '''
        Parameters
        ----------
        inter_cluster_metric: str
            Metric for inverse weighing at tree level - see Options (Default is 'var')
        intra_cluster_metric: str
            Metric for inverse weighing at cluster level - see Options (Default is 'var')
            
        Options
        -------
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
        '''

        
        intra_gmm, inter_gmm = None, None
        if 'gmm' in intra_cluster_metric: intra_gmm = 'Gauss'
        if 'gmm' in inter_cluster_metric: inter_gmm = 'Gauss'
        
        ### Calculate the intra cluster weightings

        sub_data  = self.data[index,:]
        
        intra     = np.empty(sub_data.shape[0])
        
        for i in range(intra.shape[0]):
            
            intra_rm = RiskMetrics()
            intra_rm.fit(sub_data[i,:], freq = self.frequency, mm = intra_gmm)
            
            intra[i] = intra_rm(intra_cluster_metric)

        ### Calculate the inter cluster weight

        intra_weights = (1/intra)/((1/intra).sum(0)).reshape(-1,1)

        weighted_data = intra_weights @ sub_data

        inter_rm = RiskMetrics()
        inter_rm.fit(weighted_data, freq = self.frequency, mm = inter_gmm)
        
        cluster_metric = inter_rm(inter_cluster_metric)

        return cluster_metric
    
    def recursively_partition(self, inter_cluster_metric = 'var', intra_cluster_metric = 'var'): 
        
        w       = np.ones(len(self.sorted_index))
        
        sub_ind = [self.sorted_index]                   

        while len(sub_ind) > 0:

            sub_ind = [item[j:k] for item in sub_ind for j,k in ((0, len(item)//2), (len(item)//2, len(item))) if len(item) > 1]

            for i in range(0, len(sub_ind), 2): 

                cluster1     = sub_ind[i]
                cluster2     = sub_ind[i+1]
                c_met1       = self.cluster_risk_metric(inter_cluster_metric, intra_cluster_metric, index = cluster1)
                c_met2       = self.cluster_risk_metric(inter_cluster_metric, intra_cluster_metric, index = cluster2)
                alpha        = 1 - (c_met1/(c_met1+c_met2))
                w[cluster1] *= alpha 
                w[cluster2] *= 1-alpha 

        return w[np.sort(self.sorted_index)]

class IVP:
    
    '''Risk Parity optimiser'''
    
    def __init__(self, data, frequency = 'D'):
        
        self.data        = data
        self.n, self.m   = self.data.shape
        self.annual      = {'D':262, 'W':52, 'M': 12}
        self.freq        = self.annual[frequency]

        self.cov         = np.cov(data)
        
        self.ann_cov     = self.cov * self.freq
        
        self.ann_returns = self.data.mean(1) * self.freq
        
    def __call__(self):
        
        
        if self.n == 1:
            
            diag = self.ann_cov
            
        else:
        
            diag = np.diag(self.ann_cov)
        
        ivp  = 1/diag
        ivp /= ivp.sum()
        
        return ivp

class Markowitz:
    
    '''
    Markowtiz Optimiser based on Monte Carlo sampling
    
    Methods
    -------
    random_portfolios(n_portfolios)
        generate n_portfolios of random weightings (batched for efficiency)
    optimise(n_portfolios = 10000, plot = True)
        Return the maximum sharpe ratio portfolio
    plot()
        Plot the portfolios in the mean-variance space
    '''
    
    def __init__(self, data, frequency = 'D'):
        
        self.data        = data
        self.n, self.m   = self.data.shape
        self.annual      = {'D':262, 'W':52, 'M': 12}
        self.freq        = self.annual[frequency]
        
        self.cov         = np.cov(data)
        self.ann_cov     = self.cov * self.freq
        
        self.ann_returns = self.data.mean(1) * self.freq
        
    def random_portfolios(self, n_portfolios):
        
        iterations = n_portfolios//5000
        
        returns    = np.empty(n_portfolios)
        volatility = np.empty(n_portfolios)
        weights    = np.empty((n_portfolios, self.n))
        
        for i in range(iterations):
        
            weights_    = np.random.random((5000, self.n))
            weights_   /= weights_.sum(axis = 1)[:,None]

            
            weights[i*5000:5000*(i+1)]    = weights_
            returns[i*5000:5000*(i+1)]    = weights_ @ self.ann_returns 
            volatility[i*5000:5000*(i+1)] = np.diag(np.sqrt(weights_ @ self.ann_cov @ weights_.T))

        sharpes = returns/volatility
            
        return returns, volatility, sharpes, weights
    
    def optimise(self, n_portfolios = 10000, plot = True):
        
        self.returns, self.volatility, self.sharpes, self.weights = self.random_portfolios(n_portfolios)
        
        self.ind             = np.argmax(self.sharpes)
        
        self.optimal_weights = self.weights[self.ind, :]
        
        if plot: self.plot()
        
        return self.optimal_weights.T
    
    def plot(self):
        
        plt.scatter(self.volatility, self.returns, c = self.sharpes, cmap = 'RdYlGn')
        plt.scatter(self.volatility[self.ind], self.returns[self.ind], marker = 'D', c = 'k', s = 100, label = 'Optimal Portfolio')
        plt.colorbar()
        plt.legend()
        plt.xlabel(r'$\sigma$')
        plt.ylabel(r'$\mu$')
        plt.title(f'Markowitz Bullet for {self.returns.shape[0]} random portfolios')
        
class Evaluation:
    
    '''Evaluation class for comparing portoflio optimisation methods
    
    Methods
    -------
    custom()
        Weightings based on mlportopt clustering methods
    markowitz()
        Markowitz based Monte Carlo optimised weightings
    hrp()
        Hierarchical Risk Parity weightings (Lopez de Prado)
    rp()
        Naive risk parity weightings
    all_data()
        Collect weightings
    plot_perf()
        Plot the realised performance on a nominal £1000
    summary()
        Descriptive summary including plot
    '''
    
    def __init__(self, train, test, weights, frequency = 'D'):
        
        self.train     = train
        self.test      = test
        self.weights   = weights
        self.frequency = frequency
        self.annual    = {'D':262, 'W':52, 'M': 12}
        self.freq      = self.annual[frequency]
        
        return
        
    def custom(self):
        
        self.my_data = self.weights @ self.test
        
        return
        
    def markowitz(self):
        
        mk              = Markowitz(self.train)
        self.mk_weights = mk.optimise(50000, plot = False)
        
        self.mk_data    = self.mk_weights @ self.test
        
        return
        
    def hrp(self):
        
        hclust           = scipy_linkage(self.train, metric = 'corr', method = 'single')
        allocation       = Allocate(self.train, hclust.linkage)
        self.hrp_weights = allocation.recursively_partition(intra_cluster_metric = 'var', 
                                                            inter_cluster_metric = 'var')
        
        self.hrp_data    = self.hrp_weights @ self.test
        
        return
        
    def rp(self):
        
        ivp              = IVP(self.train)
        self.ivp_weights = ivp()
        
        self.ivp_data   = self.ivp_weights @ self.test
        
        return
        
    def all_data(self):
        
        self.all_data = {'Custom': self.my_data, 
                         'MK': self.mk_data, 
                         'HRP':self.hrp_data, 
                         'IVP':self.ivp_data}
        
        return
        
    def plot_perf(self):
        
        for k, v in self.all_data.items():
            
            plot_data = 1000*np.cumprod((v + 1))
            
            plot_data = np.insert(plot_data, 0, 1000)
 
            plt.plot(plot_data, label = k)
            plt.legend()
        
        plt.show()
        
        return

    def summary(self, verbose = True):

        columns = ['Ann. Ret', 'Ann. Vol', 'Sharpe', 'Prob. Sharpe', 'VaR', 'CVaR']
        
        rows    = list(self.all_data.keys())
        
        data_   = np.empty((len(rows), len(columns)))
        
        for i, v in enumerate(self.all_data.values()):
        
            rm = RiskMetrics()
            rm.fit(v, self.frequency)
            
            mean       = rm('ann_ret')
            vol        = rm('ann_vol')
            sharpe     = rm('ann_sharpe')
            p_sharpe   = rm('prob_sharpe')
            norm_var   = rm('VaR - normal')
            norm_cvar  = rm('CVaR - normal')

            data_[i,:] = [mean, vol, sharpe, p_sharpe, norm_var, norm_cvar]
            
        self.summary_df = pd.DataFrame(data_, index = rows, columns = columns)
        
        if verbose: print(self.summary_df)
        
        return

    def __call__(self, plot = True):
        
        self.custom()
        self.markowitz()
        self.hrp()
        self.rp()
        self.all_data()
        if plot: self.plot_perf()
        self.summary(plot)
        
        return
        