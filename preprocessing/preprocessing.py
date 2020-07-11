import sys
import numpy             as np
import matplotlib.pyplot as plt

from sklearn.neighbors     import KernelDensity
from scipy.optimize        import minimize
from tqdm.notebook         import tqdm
from sklearn.covariance    import LedoitWolf

from mlportopt.util        import *
from mlportopt.riskmetrics import *

import warnings
import time
import tensorflow.compat.v1 as tf
import numpy                as np
import matplotlib.pyplot    as plt
import os
from IPython.display        import Image
from scipy                  import stats
from sklearn.decomposition  import PCA as sklearn_PCA
from tensorflow             import keras
from ipypb                  import ipb as tqdm

class PCA:
    
    '''
    Principal Component Analysis
    
    Methods
    -------
    
    fit(n)
        Fits the PCA
    reverse(evecs)
        Reverses the decomposition back to the original space
    calc_eigens
        Calculate the eigenvectors and eigenvalues
    get_nevecs
        Get the first n eigenvectors sorted by eigenvalue
    get_decomp
        Return the eigenvectors and eigenvalues
    recon_error
        Calculate the MSE of the reconstructed data
    '''
    
    def __init__(self, data, ismat = False):
        
        self.X   = data
        self.mat = np.corrcoef(self.X.T)
        
        if ismat: self.mat = data

    def fit(self, n = None):
        
        self.calc_eigen()
        
        if n is not None:
        
            self.n_evecs = self.evecs[:, :n]
            
        else:
            
            self.n_evecs = self.evecs.copy()
        
        self.reduced = (self.n_evecs.T @ self.X.T).T
        
        return self
    
    def reverse(self, evecs):
        
        self.reversed = (self.reduced @ evecs.T)
        
        return self.reversed

    def calc_eigen(self):
    
        eigenvalues, eigenvectors = np.linalg.eigh(self.mat)

        self.sort = np.argsort(eigenvalues)[::-1]
        
        self.evals = eigenvalues[self.sort]
        self.evecs = eigenvectors[:, self.sort]

        return
    
    def get_nevecs(self):
        
        return self.n_evecs 
    
    def get_decomp(self):
        
        return self.evals, self.evecs
    
    def recon_error(self):
        
        return np.mean(np.square(self.reversed - self.X))
    
##########################################################################################################################################################

class DimAE:
    
    '''Dimensionality Reducing Auto Encoder
    
    Methods
    -------
    
    create_model()
        Builds the model in tensorflow
    compute_loss()
        Computes the loss (MSE)
    optimise()
        Optimises the model with ADAM
    train()
        Trains the model
    reduce(x)
        Encodes the input data x to the latent layer through the trained Encoder
    
    '''
    
    def __init__(self, output_dir = './dimAE_logdir/', 
                 lr          = 0.001, 
                 nb_epochs   = 1, 
                 batch_size  = 50, 
                 n           = 60000,
                 num_chans   = 100,
                 activation  = tf.nn.relu,
                 encoder     = [80,40],
                 latent_dim  = 10,
                 regularizer = 'l2',
                 reg_coeff   = 0, 
                 drop_coeff  = 0,
                 verbose     = False):
        
        '''
        Parameters
        ----------
        output_dir: str
            Location of tf output (Default is './dimAE_logdir/')
        lr: float
            Learning rate of the ADAM optimiser (Default is 0.001)
        nb_epochs:
            Number of epochs in the training method (Default is 1)
        batch_size: int
            Batch size (Default is 50)
        n: int
            Number of training exemplars (Default is 60000)
        num_chans: int
            Number of input features (Default is 100)
        activation: tf object
            Activation function of dense layers
        encoder: list
            List of dense layer depths for the encoder. Additional entries will create new dense layers. The decoder will be built as a reverse of the encoder.
        latent_dim: int
            Number of nodes in the latent layer, equivalent to the desired size of the reduced dimension (Default is 10)
        regularizer: str
            Regulariser to be used in the encoder and decoder (Default is 'l2') [Options: 'l1','l2','l1l2', None]
        reg_coeff: float
            Size of the regularisation coefficient if regularizer is not None (Default is 0)
        drop_coeff: float
            Coefficient of the model dropout (Default is 0, corresponding to no dropout)
        verbose: Bool
            Boolean indicator of descriptive print statements
        '''
     
        self.n             = n
        self.num_chans     = num_chans
        self.nb_epochs     = nb_epochs
        self.lr            = lr
        self.batch_size    = batch_size
        self.nb_epochs     = nb_epochs
        self.nb_iterations = self.n // batch_size
        self.output_dir    = output_dir
        self.input         = tf.placeholder(tf.float32, [None, self.num_chans])
        self.dtype         = 'float32'

        self.activation    = activation
        self.encoder       = encoder
        self.latent_dim    = latent_dim
        self.verbose       = verbose
        
        self.regularizer   = regularizer
        self.reg_coeff     = reg_coeff
        self.drop_coeff    = drop_coeff
        
        
    def create_model(self):
        
        tf.keras.backend.set_floatx('float64')
        
        if self.regularizer == 'l1': regularizer = tf.keras.regularizers.l1(self.reg_coeff)
        else:                        regularizer = tf.keras.regularizers.l2(self.reg_coeff)
        
        with tf.variable_scope('Encoder', reuse=tf.AUTO_REUSE):
            
            self.encoder_layers = []
            
            for i, layer1 in enumerate(self.encoder):
                
                h = tf.keras.layers.Dense(layer1, dtype = self.dtype,
                                     activation = self.activation,
                                     kernel_regularizer = regularizer,
                                     name       = f'encoder_{i+1}')
                
                self.encoder_layers.append(h)
                
                d = tf.keras.layers.Dropout(self.drop_coeff, dtype = self.dtype) 
                
                self.encoder_layers.append(d)
            
            self.latent_layer =  tf.keras.layers.Dense(self.latent_dim,
                                                      dtype = self.dtype,
                                                      activation = self.activation,
                                                      kernel_regularizer = regularizer,
                                                      name       = f'latent')
            
            self.encoder_layers.append(self.latent_layer)
            
        with tf.variable_scope('Decoder', reuse=tf.AUTO_REUSE): 

            self.decoder_layers = []
            
            for j, layer2 in enumerate(self.encoder[::-1]):
                      
                k = tf.keras.layers.Dense(layer2,  dtype = self.dtype,
                                     activation = self.activation,
                                     kernel_regularizer = regularizer,
                                     name       = f'decoder{j+1}')

                self.decoder_layers.append(k)
                
                d = tf.keras.layers.Dropout(self.drop_coeff, dtype = self.dtype) 
                
                self.decoder_layers.append(d)
                
            k =  tf.keras.layers.Dense(self.num_chans,
                                       dtype = self.dtype,
                                       activation = self.activation,
                                       kernel_regularizer = regularizer,
                                       name       = f'last') 
            
            self.decoder_layers.append(k)
            
        self.encoder = tf.keras.models.Sequential(self.encoder_layers)
        
        self.decoder = tf.keras.models.Sequential(self.decoder_layers)
                
        self.model   = tf.keras.models.Sequential([self.encoder,self.decoder])

        self.recon = self.model(self.input)  
        
        if self.verbose:
            
            print(self.model.summary())
            print(self.encoder.summary()) 
            print(self.decoder.summary())
    
    def compute_loss(self):
        
        with tf.variable_scope('loss'):
            
            self.loss      = tf.losses.mean_squared_error(self.input, self.recon) 
            
            self.loss_summ = tf.summary.scalar("reconstruction_loss", self.loss)
                             
    def optimizer(self):
        
        with tf.variable_scope('optimizer'):
            
            optimizer       = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5)
            
            self.model_vars = tf.trainable_variables()
            
            self.trainer    = optimizer.minimize(self.loss, var_list=self.model_vars)
            
    def train(self, x): ### Train 

        self.create_model()
        self.compute_loss()
        self.optimizer()
        
        init = (tf.global_variables_initializer(),
                tf.local_variables_initializer())
        
        saver        = tf.train.Saver()
        summary      = tf.Summary()
        self.sess    = tf.InteractiveSession()
        self.sess.run(init)

        writer  = tf.summary.FileWriter(self.output_dir)
        writer.add_graph(self.sess.graph)
  
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.Loss     = []

        for epoch in tqdm(range(self.nb_epochs), leave=False):

            randomize = np.arange(x.shape[0])
            np.random.shuffle(randomize)

            x    = x.reshape(-1,self.num_chans)
            
            x_in = x[randomize,:]

            for i in tqdm(range(self.nb_iterations), leave=False):

                input_x_train = x_in[i*self.batch_size: (i+1)*self.batch_size]

                _ , x , recon, loss, loss_summ = self.sess.run([self.trainer,
                                                           self.input,  
                                                           self.recon, 
                                                           self.loss,
                                                           self.loss_summ],
                                       feed_dict = {self.input : input_x_train})
                
                if self.verbose:
                    
                    if i % (self.nb_iterations/5) == 0:
                    
                        print(loss)
                
                writer.add_summary(loss_summ, epoch * self.nb_iterations + i)
                self.Loss.append(loss)

            saver.save(self.sess, self.output_dir, global_step=epoch)  

        return
    
    def reduce(self, x):
        
        latent = self.sess.run(self.encoder(x))

        return latent
    
################################################################################################################################################################################
    
def PCA_(data, n, labels = 'r', visualise = False):
    
    '''
    sklearn PCA, actually performs Singular Value Decomposition
    '''
    
    pca = sklearn_PCA(n_components = n).fit(data)

    components = pca.transform(data)
    
    if n == 2 and visualise:
        
        plt.scatter(components[:,0], components[:,1], c = labels, s = 0.5)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.show()

    elif n == 3 and visualise:

        fig = plt.figure()
        ax  = plt.axes(projection='3d')
        ax.scatter3D(components[:,0], components[:,1], components[:,2], c=labels)
        plt.show()
    
    return components

def whiten(data):
    
    '''
    Simple function to demean and standardise the data
    '''
    
    return (data - data.mean(axis = 0))/data.std(axis = 0)

def preprocess(data, white = True, reduce = True, n = 2, visualise = False):
    
    '''
    Wrapper for whitening and dimensionality reduction
    '''
    
    if reduce:
        
        data = PCA_(data, n, visualise = visualise)
        
    if white:
        
        data = whiten(data)
        
    return data

################################################################################################################################################################################


class RMT:
    
    '''Toolbox for cleaning Hermitian matrices based on Random Matrix Theory
    
    Methods
    -------
    
    fitKDE(KDE_bwidth = 0.01, x = None)
        Estimate the Kernel Density and return the PDF
    MarcenkoPastur(s2, q, points = 100)
        Evaluate the Marcenko - Pastur distribution
    PDFdiff
        Return the difference between the MP distribution and the Kernel Density estimated distribution
    optimise
        Optimise the MP parameters by minimising the PDFdiff
    denoise_fixed
        Denoise the matrix by fixing the random eigenvectors to constants
    denoise_shrinkage
        Denoise the matrix through shrinkage
    denoise_targeted_shrinkage
        Denoise the matrix through targeted shrinkage of the random eigenvectors
    detone
        Remove the first n eigenvectors (sorted by eigenvalue). Removal of the first is equivalent to beta-adjusting the returns for financial correlation matrices.
    condition_number
        Return the cndition number of the matrix
    plot
        Plot the MP and KDE distributions
        
        '''
    
    def __init__(self, data, ismat = False):
        
        '''
        Parameters
        ----------
        ismat: Bool
            Boolean indicator if data being passed is cov/corr matrix or raw data
        '''
        
        self.ismat         = ismat
        
        self.data          = data
        
        self.corr_denoised = None
        
        self.evals, self.evecs = PCA(self.data, ismat = ismat).fit().get_decomp()
        
    def fitKDE(self,  KDE_bwidth = 0.01, x = None):
        
        '''
        Parameters
        ----------
        KDE_bwidth: float
            Bandwith for the Kernel Density Estimator (Default is 0.01)
        x: vector
            Data over which the pdf is evaluated. If None passed evaluates over unique values in the data (Default is None)
        '''
    
        if x is None:
    
            x  = np.unique(self.evals).reshape(-1,1)
        
        kde      = KernelDensity(kernel = 'gaussian', bandwidth = KDE_bwidth).fit(self.evals.reshape(-1,1))

        log_prob = kde.score_samples(x)
        
        self.kde_pdf = np.exp(log_prob).reshape(-1,1)

        return 
                                 
    def MarcenkoPastur(self, s2, q, points = 1000):
        
        '''
        Parameters
        ----------
        s2: float
            MP variance parameter
        q: float
            Ratio, T/N of the dimensions of the matrix
        points: int
            Linspace over which the PDF is calculated
        '''
        

        max_expected_eval = s2 * (1 + (1/q)**0.5)**2
        min_expected_eval = s2 * (1 - (1/q)**0.5)**2

        self.linspace = np.linspace(min_expected_eval, max_expected_eval, points)

        self.MP_pdf   = q * np.sqrt((max_expected_eval - self.linspace) * (self.linspace - min_expected_eval)) / (2 * np.pi * self.linspace * s2)

        return
    
    def PDFdiff(self, s2, q, KDE_bwidth):
        
        '''
        Parameters
        ----------
        s2: float
            MP variance parameter
        q: float
            Ratio, T/N of the dimensions of the matrix
        KDE_bwidth: float
            Bandwith for the Kernel Density Estimator (Default is 0.01)
        '''
        
        self.MarcenkoPastur(s2, q)
        self.fitKDE(KDE_bwidth, x = self.linspace)
        
        return np.sum(np.square(self.kde_pdf - self.MP_pdf))
    
    def optimise(self, q = 10, KDE_bwidth = 0.01, plot = False, verbose = False):
        
        '''
        Parameters
        ----------
        q: float
            Ratio, T/N of the dimensions of the matrix
        KDE_bwidth: float
            Bandwith for the Kernel Density Estimator (Default is 0.01)
        plot: Bool
            Boolean indicator to plot the distributions (Default is False)
        verbose: Bool
            Boolean indicator for descriptive printing (Default is False)    
        '''
        
        optimised = minimize(lambda *x: self.PDFdiff(*x), 
                             0.5, 
                             args = (q, KDE_bwidth), 
                             bounds = ((1e-5, 1-1e-5), ))
        
        if optimised['success'] == True: 
            
            self.var = optimised['x'][0]
            
        else: 
            
            self.var = 1
            
        self.emax = self.var * (1 + (1/q)**0.5)**2
        
        self.retained_index = self.evals >= self.emax        
        self.retained_evals = self.evals[self.retained_index]
        self.retained_evecs = self.evecs[self.retained_index]

        self.n_retained     = len(self.retained_evals)
        
        if verbose: print(f'{self.n_retained} features identified')
        
        if plot: self.plot()
        
        return
    
    def denoise_fixed(self):
        
        '''Denoise by fixing random eigenvalues'''
        
        evals = self.evals.copy()
        evecs = self.evecs.copy()
        
        evals[~self.retained_index] = np.mean(evals[~self.retained_index])
        
        cov   = evecs @ np.diag(evals) @ evecs.T
        
        corr  = cov/np.outer(np.sqrt(np.diag(cov)), np.sqrt(np.diag(cov)))
        
        corr[corr<-1] = -1
        corr[corr> 1] =  1
        
        self.cov_denoised  = cov
        self.corr_denoised = corr
        
        return cov, corr
    
    def denoise_shrinkage(self):
        
        if self.ismat: 
            
            print('Pass original data not correlation/covariance matrix')
            pass
        
        cov   = LedoitWolf().fit(self.data).covariance_
        
        corr  = cov/np.outer(np.sqrt(np.diag(cov)), np.sqrt(np.diag(cov)))
        
        corr[corr<-1] = -1
        corr[corr> 1] =  1
        
        self.cov_denoised  = cov
        self.corr_denoised = corr
        
        return cov, corr

    def denoise_targeted_shrinkage(self, alpha = 0):
        
        '''Target the random eigenvalues for shrinkage and not the signal'''
        
        corr1  = self.evecs[:,self.retained_index]  @ np.diag(self.evals[self.retained_index])  @ self.evecs[:,self.retained_index].T
                
        corr2  = self.evecs[:,~self.retained_index] @ np.diag(self.evals[~self.retained_index]) @ self.evecs[:,~self.retained_index].T

        corr   = corr1 + alpha * corr2  + (1-alpha) * np.diag(np.diag(corr2))
        
        self.corr_denoised = corr - 1e-9

        return corr, corr
    
    def detone(self, n = 1):
        
        '''Remove the first n eigenvalues and associated eigenvectors'''
        
        evals = self.evals.copy()
        evecs = self.evecs.copy()
        
        if self.corr_denoised is not None:
            
            evals, evecs = PCA(self.corr_denoised, ismat = True).fit().get_decomp()
        
        evals = evals[n:]
        evecs = evecs[:, n:]
        
        cov   = evecs @ np.diag(evals) @ evecs.T
        
        corr  = cov/np.outer(np.sqrt(np.diag(cov)), np.sqrt(np.diag(cov)))
        
        corr[corr<-1] = -1
        corr[corr> 1] =  1
        
        self.cov_detoned  = cov
        self.corr_detoned = corr
        
        return cov, corr
    
    def __call__(self, method = 'fixed'):
        
        if method == 'fixed':
            
            return self.denoise_fixed()
        
        elif method == 'shrinkage':
        
            return self.denoise_shrinkage()
        
        elif method == 'targeted_shrinkage':
            
            return self.denoise_targeted_shrinkage()
    
    def condition_number(self, corr):
        
        return np.linalg.cond(corr)
         
    def plot(self):
        
        plt.hist(self.evals, bins = 100, density = True, label = 'Empirical KDE distribution')
        plt.plot(self.linspace, self.MP_pdf, label = 'Marcenko Pastur PDF')
        plt.ylabel(r'Prob($\lambda$)')
        plt.xlabel(r'$\lambda$')
        plt.legend()
        plt.show()
        