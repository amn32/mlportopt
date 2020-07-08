import numpy  as np
import pandas as pd
from   scipy  import stats

from mlportopt.mixturemodels import *

class RiskMetrics:
    
    def __init__(self):
        
        self.annual = {'D':261, 'W':52, 'M': 12}
        
    def fit(self, data, freq = 'D', alpha = 0.05, bmark = None, mm = None):
        
        self.data  = data.T
        self.freq  = self.annual[freq]
        self.alpha = alpha
        self.mm    = mm

        self.gmm_samples = None
        
        if bmark is not None:
            
            rm_bmark   = RiskMetrics().fit(bmark, freq = freq, alpha = alpha)
            self.bmark = rm_bmark('sharpe')
            
        else:
            
            self.bmark = 0
            
        if self.mm == 'Gauss':
            
            self.gmm_samples = GMM(self.data.reshape(-1,1), epochs = 50, 
                                   initialisation='Fixed', 
                                   bias= 0.1, 
                                   constrained=True, 
                                   constraints = [0.2, 0.6,0.3]).fit().sample(100000, False)
            
        elif self.mm == 'Gamma':
            
            self.gmm_samples = GMM(self.data).fit().sample(100000, False) #Replace this with Gamma Mixture model
             
        self.sharpe()
        self.prob_sharpe()
        self.ann_vol()
        self.VaR()
        self.CVaR()
        
        if self.n < 30: print('Need more observations for CLT assumptions to be valid')
        
        return self
    
    def VaR(self, percentile = 95):
        
        gmm     = None
        
        norm    = stats.norm.ppf(1 - percentile/100, loc = self.data.mean(), scale = self.data.std())
        
        student = stats.t.ppf(1 - percentile/100, 3, loc = self.data.mean(), scale = self.data.std())
        
        if self.mm is not None:
            
            gmm     = np.percentile(self.gmm_samples, (100 - percentile))
        
        self.VaR_dict = {'Normal': norm, 'Student': student, 'GMM':gmm}

        return self.VaR_dict
    
    def CVaR(self, percentile = 95):
        
        cgmm     = None
        
        norm     = stats.norm.ppf(1 - percentile/100, loc = self.data.mean(), scale = self.data.std())
        
        student  = stats.t.ppf(1 - percentile/100, 3, loc = self.data.mean(), scale = self.data.std())
        
        if self.mm is not None:
        
            gmm      = np.percentile(self.gmm_samples, (100 - percentile))
            cgmm     = self.gmm_samples[self.gmm_samples < gmm].mean()
        
        cnorm    = self.data[self.data < norm].mean()
        
        if np.any(self.data < student):
        
            cstudent = self.data[self.data < student].mean()
            
        else:
            
            cstudent = None

        self.CVaR_dict = {'Normal': cnorm, 'Student': cstudent, 'GMM':cgmm}
        
        return self.CVaR_dict
    
    def ann_vol(self):
        
        self.annualised_vol = np.std(self.data) * np.sqrt(self.freq)
        
        return 
    
    def sharpe(self):
        
        self.n              = self.data.shape[0]
        self.skew           = stats.skew(self.data)
        self.kurtosis       = stats.kurtosis(self.data, fisher = False)
        
        self.sharpe_hat     = (np.mean(self.data)/np.std(self.data))
        
        self.ann_sharpe_hat = self.sharpe_hat * np.sqrt(self.freq)
        
        self.s2_sharpe_hat  = ((1/(self.n-1))*(1 + 0.5*self.sharpe_hat**2 - self.skew*self.sharpe_hat + ((self.kurtosis - 3)/4)*self.sharpe_hat**2))**0.5
        
        return
    
    def prob_sharpe(self):
        
        self.sharpe()
        
        self.prob_sharpe = stats.norm.cdf((self.sharpe_hat - self.bmark) / self.s2_sharpe_hat)
        
        self.req_samples = 1 + (1 - self.skew * self.sharpe_hat + ((self.kurtosis - 1)/4)*self.sharpe_hat**2) * (stats.norm.ppf(1-self.alpha) / (self.sharpe_hat - self.bmark)) ** 2
        
        return
    
    def __call__(self, metric = 'prob_sharpe'):
        
        if metric == 'prob_sharpe':
        
            return self.prob_sharpe, self.req_samples
        
        elif metric == 'ann_sharpe':
            
            return self.ann_sharpe_hat
        
        elif metric == 'sharpe':
            
            return self.sharpe_hat
        
        elif metric == 'var':
            
            return self.data.var()
        
        elif metric == 'std':
            
            return self.data.std()
        
        elif metric == 'ann_vol':
            
            return self.annualised_vol
        
        elif metric == 'VaR - normal':
            
            return self.VaR_dict['Normal']
        
        elif metric == 'VaR - student':

            return self.VaR_dict['Student']
        
        elif metric == 'VaR - gmm':

            return self.VaR_dict['GMM']
            
        elif metric == 'CVaR - normal':
            
            return self.CVaR_dict['Normal']
        
        elif metric == 'CVaR - student':

            return self.CVaR_dict['Student']
        
        elif metric == 'CVaR - gmm':

            return self.CVaR_dict['GMM']

