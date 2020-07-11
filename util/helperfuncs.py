import numpy as np
import sklearn.decomposition as decomp
import matplotlib.pyplot as plt
from   mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import yfinance as yf 
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import bs4 as bs
import pickle
import requests
import lxml
import os
import shutil
import datetime as dt
from sklearn.linear_model import LinearRegression

from mlportopt.riskmetrics.riskmetrics import RiskMetrics

def train_test(data, split = 0.5):
    
    obs = data.shape[1]
    
    split_val = int(obs * split)
    
    train = data[:, :split_val]
    test  = data[:, split_val:]
    
    return train, test

def plot_3d(data, labels = 'r'):

    fig = plt.figure()
    ax  = plt.axes(projection='3d')
    ax.scatter3D(data[:,0], data[:,1], data[:,2], c=labels)
    plt.show()

    return

def plot_2d(data, labels = 'r'):

    plt.scatter(data[:,0], data[:,1], c = labels, s = 0.5)
    plt.show()

    return

def plot_clusters(data, labels = 'r'):

    if data.shape[1] == 2:

        plot_2d(data, labels)

    elif data.shape[1] == 3:

        plot_3d(data, labels)

    return

def plot_corr(corr, labels):
    
    fig1     = plt.figure()
    ax1      = fig1.add_subplot(111)
    heatmap  = ax1.pcolor(corr, cmap=plt.cm.RdYlGn)
    fig1.colorbar(heatmap)

    ax1.set_xticks(np.arange(corr.shape[1]) + 0.5, minor=False)
    ax1.set_yticks(np.arange(corr.shape[0]) + 0.5, minor=False)
    ax1.invert_yaxis()
    ax1.xaxis.tick_top()
    ax1.set_xticklabels(labels)
    ax1.set_yticklabels(labels)
    plt.xticks(rotation=90)
    heatmap.set_clim(-1, 1)
    plt.tight_layout()
    plt.show()

def gen_clusters(clusters  = 4, 
                 n         = 1000,
                 m         = 100,
                 centres   = [0,0.3,0.5,-0.1], 
                 variances = [2,1,1,1], 
                 noise_variances = [0.5,1,0.75,0.3],
                 seed = False):

    if seed: np.random.seed(0)
    
    data   = np.empty((n*clusters, m))
    labels = np.empty(n*clusters)
    
    for j in range(clusters):
        
        vals = np.random.normal(centres[j], variances[j], m)
        
        for i in range(n):
            
            noise = np.random.normal(0, noise_variances[j], m)
            
            data[i + j*n,:] = vals + noise
            
            labels[i + j*n] = j
            
    shuffle = np.random.choice(range(clusters*n), size = clusters*n, replace = False)
    
    data    = data[shuffle,:]
    labels  = labels[shuffle]
    
    return data, labels

def gen_covmatrix(n = 1000, m = 100, alpha = 0.995):
    
    rand  = np.random.normal(size = (n, m))
    
    cov   = rand @ rand.T
    
    cov  += np.diag(np.random.uniform(size = n))

    noise = np.cov(np.random.normal(size = (n*(n//m), n)), rowvar = False)
    
    cov   = alpha * noise + (1 - alpha) * cov
        
    corr = cov/np.outer(np.sqrt(np.diag(cov)), np.sqrt(np.diag(cov)))

    corr[corr<-1] = -1
    corr[corr> 1] =  1

    return cov, corr

def gen_real_data(frequency = 'W', spy_adj = False, n = 100, m = 50, seed = 1, ret_data = False):
    
    np.random.seed(seed)
    
    data = pd.read_csv('sp500_joined_closes.csv')
    data.set_index('Date', inplace=True)
    data = data.loc['2015-01-04':,:]
    null_index = list(~data.loc['2017-01-04'].isnull())
    data = data.loc[:,null_index].fillna(method = 'backfill')
    data = data.astype(float)
    data.index = pd.to_datetime(data.index)
    
    sub_data = data.groupby(pd.Grouper(freq=frequency)).last()
    sub_data = sub_data.pct_change()
    sub_data = sub_data.iloc[2:,:]
    
    sub_ind1 = np.random.choice(range(sub_data.shape[1]),size = n, replace = False)
    sub_ind2 = list(sub_data.index)[-m:]
    sub_data = sub_data.iloc[:,sub_ind1]
    sub_data = sub_data.loc[sub_ind2,:]
    
    if ret_data:
        
        return sub_data.values.T, list(sub_data.columns), sub_data
    
    if spy_adj:
        
        spind = get_spy(process = True, frequency = frequency, m = m)
        
        assert(np.all(list(sub_data.index) == list(spind.index)))
        
        adj_data = beta_adjust(sub_data.values.T,  spind.values.T)
        
        return sub_data.values.T, list(sub_data.columns), adj_data
    
    else:
    
        return sub_data.values.T, list(sub_data.columns)

def get_spy(process = True, frequency = 'W', m = 50):

    ticker = '^GSPC'
    
    if os.path.isdir('spy'): 
        shutil.rmtree('spy')
        os.makedirs('spy')
    else:
        os.makedirs('spy')

    start = dt.datetime(2010, 1, 1)
    end   = dt.datetime(2020, 3, 10) #dt.datetime.now()

    df = yf.download(ticker, start, end, threads = True, progress = False)
    df.reset_index(inplace=True)
    df.set_index("Date", inplace=True)
    df.to_csv('{}.csv'.format(ticker))
    
    if process:
        
        df.rename(columns={'Adj Close': 'SPY'}, inplace=True)
        df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)
        df = df.loc['2015-01-04':,:]
        null_index = list(~df.loc['2017-01-04'].isnull())
        df = df.loc[:,null_index].fillna(method = 'backfill')
        df = df.astype(float)
        df.index = pd.to_datetime(df.index)
        sub_data = df.groupby(pd.Grouper(freq=frequency)).last()
        sub_data = sub_data.pct_change()
        sub_data = sub_data.iloc[2:,:]
        
        sub_ind2 = list(sub_data.index)[-m:]
        sub_data = sub_data.loc[sub_ind2,:]
        
        df = sub_data.copy()
        
    return df

def beta_adjust(data, ind):
    
    n, m      = data.shape
    
    residuals = np.empty((n, m))
    
    x         = ind.reshape(-1,1)
    
    for i in range(n):
        
        y     = data[i,:].reshape(-1,1)
    
        model = LinearRegression().fit(x, y)
        
        y_hat = model.predict(x)
        
        resids = y - y_hat
        
        residuals[i,:] = resids.reshape(-1,1).T
        
    return residuals

def merge_clusters(data, clusters, resids = None, freq = 'W', method = 'ann_vol'):
    
    n, m              = data.shape
    
    unique            = np.unique(clusters).shape[0]
    
    data_risk_measure = np.empty(n)
    
    new_data          = np.empty((unique,m))
    
    new_resids        = np.empty((unique,m))
    
    clust_weights     = {}
    
    for i in range(n):
    
        data_risk_measure[i] = RiskMetrics().fit(data[i, :].T, freq = freq)('ann_vol')
        
    weights = (1/data_risk_measure) #/(1/data_risk_measure).sum(0)

    for i, j in enumerate(np.unique(clusters)):
        
        clust_ind        = clusters == j
        
        sub_weights      = weights[clust_ind]/weights[clust_ind].sum()
        
        clust_weights[j] = sub_weights
        
        new_data[i,:]    = sub_weights @ data[clust_ind,:]
        
        if resids is not None:
        
            new_resids[i,:] = sub_weights @ resids[clust_ind,:]
   
    return new_data, new_resids, clust_weights

def get_full_weights(flat_weights, cluster_weights):
    
    full_weights = []

    ind = 0

    for v, value in enumerate(cluster_weights.values()):

        for item in value:

            full_weights.append(item * flat_weights[v])

            ind +=1    
            
    return np.array(full_weights)