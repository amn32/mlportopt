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

#     import pandas_datareader.data as web
from tqdm.notebook import tqdm
from sklearn.linear_model import LinearRegression

from pandas.tseries.offsets import BDay

isBusinessDay = BDay().is_on_offset

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
    
    rand  = np.random.normal(0, 5, size = (n, m))
    
    cov   = rand @ rand.T
    
    cov  += np.diag(np.random.uniform(0,3,size = n))

    noise = np.cov(np.random.normal(0, 1, size = (n*(n//m), n)), rowvar = False)
    
    cov   = alpha * noise + (1 - alpha) * cov
        
    corr = cov/np.outer(np.sqrt(np.diag(cov)), np.sqrt(np.diag(cov)))

    corr[corr<-1] = -1
    corr[corr> 1] =  1

    return cov, corr

def import_data(path = 'sp500_joined_closes.csv', frequency = 'D'):
    
    data       = pd.read_csv(path)
    data.set_index('Date', inplace=True)
    data       = data.loc['2015-01-01':,:]
    null_index = list(~data.loc['2017-01-04'].isnull())
    data       = data.loc[:,null_index].fillna(method = 'backfill')
    data       = data.astype(float)
    data.index = pd.to_datetime(data.index) 
    bdays      = pd.to_datetime(data.index).map(isBusinessDay)
    data       = data.loc[bdays,:]
    
#     checker    = data.pct_change().iloc[1:,:]

#     checker    = checker.drop(columns=checker.columns[((checker==0).sum()>5)],axis=1)
    
#     filt_ind   = checker.columns
    
    sub_data   = data.groupby(pd.Grouper(freq=frequency)).last()
    sub_data   = sub_data.pct_change()
    sub_data   = sub_data.iloc[1:,:]

    return sub_data

def gen_real_data(frequency = 'W', spy_adj = False, n_assets = 100, window_length = None, start = None, end = None, seed = 1):
    
    np.random.seed(seed)
    
    sub_data = import_data(frequency = frequency)
    
    if window_length is None:
        
        sub_data = sub_data.loc[start:end, :]
        
    else: 
        
        sub_ind1 = list(sub_data.index)[-window_length:]
        sub_data = sub_data.loc[sub_ind1, :]
        
    spind    = sub_data.loc[:,'^GSPC'].copy()
    
    drop = ['^GSPC', 'TT', 'LW', 'HPE', 'FTV', 'AMCR']
    
    sub_data.drop(drop, axis = 1, inplace = True)
    
    sub_ind2 = np.random.choice(range(sub_data.shape[1]),size = n_assets, replace = False)
    sub_data = sub_data.iloc[:,sub_ind2]

    if spy_adj:
        
        assert(np.all(list(sub_data.index) == list(spind.index)))
        
        adj_data = beta_adjust(sub_data.values.T,  spind.values.T)
        
        return sub_data.values.T, list(sub_data.columns), adj_data
    
    else:
    
        return sub_data.values.T, list(sub_data.columns), sub_data

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
    
    if unique < 5:
        
        return data, resids, None
    
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

class SPY:
    
    def __init__(self, start, end):
        
        '''
        Pass start and end as ISO 'YYYY-MM-DD'
        '''
        self.start = dt.datetime.fromisoformat(start)
        self.end   = dt.datetime.fromisoformat(end)
        
    def save_sp500_tickers(self):
        if os.path.isdir('temp_stock'): 
            shutil.rmtree('temp_stock')
            os.makedirs('temp_stock')
        else:
            os.makedirs('temp_stock')
        resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        soup = bs.BeautifulSoup(resp.text, 'html.parser')
        table = soup.find('table', {'class': 'wikitable sortable'})
        tickers = []
        for row in table.findAll('tr')[1:]:
            ticker = row.findAll('td')[0].text
            tickers.append(ticker[:-1])
        tickers.append('^GSPC')

        with open("temp_stock/sp500tickers.pickle","wb") as f:
            pickle.dump(tickers,f)

        return tickers
    
    def get_data_from_yahoo(self):
        with open("temp_stock/sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)
            os.makedirs('temp_stock/stock_dfs')

        start = self.start
        end   = self.end
        for ticker in tqdm(tickers, leave = False):
            try:
                df = yf.download(ticker, start, end, threads = True, progress = False)
                df.reset_index(inplace=True)
                df.set_index("Date", inplace=True)
                df.to_csv('temp_stock/stock_dfs/{}.csv'.format(ticker))
            except:
                continue

        return df
    
    def compile_data(self):
        with open("temp_stock/sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)

        main_df = pd.DataFrame()

        for count, ticker in enumerate(tickers):
            try:
                df = pd.read_csv('temp_stock/stock_dfs/{}.csv'.format(ticker))
                df.set_index('Date', inplace=True)

                df.rename(columns={'Adj Close': ticker}, inplace=True)
                df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)

                if main_df.empty:
                    main_df = df
                else:
                    main_df = main_df.join(df, how='outer')

                if count % 100 == 0:
                    print(count)
            except:
                continue
        main_df.to_csv('sp500_joined_closes.csv')
        shutil.rmtree('temp_stock')

        
    def __call__(self):
        self.save_sp500_tickers()
        self.get_data_from_yahoo()
        self.compile_data()
        return
    
    