import warnings
import time
import tensorflow.compat.v1 as tf
import numpy                as np
import matplotlib.pyplot    as plt
import os
import itertools
from matplotlib             import cm
from IPython.display        import Image
from scipy                  import stats
from sklearn.decomposition  import PCA
from tensorflow             import keras
from ipypb                  import ipb as tqdm
from scipy.spatial.distance import cdist
from scipy.special          import gammaln, digamma

# from mlportopt.util import *
from mlportopt.preprocessing import *

class TFSOM:
    
    '''
    Tensorflow Self Organising Map
    
    Methods
    -------
    
    optimiser()
        Optimises the SOM
    train(x)
        Train the SOM
    fit(x)
        Project the data onto the SOM and return the region
    '''
    
    def __init__(self, xdim, ydim, n_nodes, lr = 0.5, r = 0.5, iterations = 50):
        
        '''
        Parameters
        ----------
        xdim: int
            Number of horizontal nodes
        ydim: int
            Number of vertical nodes
        n_nodes: int
            Number of input features
        lr: float, optional
            Learning rate (default is 0.5)
        r: float, optional
            Initial map radius (default is 0.5)
        iterations: int, optional
            Number of iterations in training (default is 50)
        '''
        
        if xdim is None: 
            raise NotImplementedError('xdim not specified')
        if ydim is None:
            raise NotImplementedError('ydim not specified')
        if n_nodes is None:
            raise NotImplementedError('n_nodes not specified')
        
        tf.reset_default_graph() # Make sure we have no active sessions
        self.xdim      = xdim
        self.ydim      = ydim
        self.lr        = lr
        self.r         = r
        self.n_nodes   = n_nodes
        self.iters     = iterations
        self.dtype     = 'float32'
        self.size      = xdim*ydim
        self.t_size    = tf.constant(self.size, 'int32')
        self.progress  = tf.placeholder(self.dtype)
        self.nodes     = np.array([[i,j] for i in range(xdim) for j in range(ydim)])
        self.div       = tf.divide(self.progress, self.iters)
        self.map       = tf.Graph()
        self.euclid    = lambda p,q: tf.reduce_sum(tf.square(tf.subtract(p,q)), axis = 1) # Calculate Euclidean distance
        self.weights   = tf.Variable(tf.random_normal([self.t_size, self.n_nodes]), dtype = self.dtype, name = 'weights')
        self.x         = tf.placeholder(dtype = self.dtype, shape = (self.n_nodes, 1), name = 'data')
        
        self.big_x     = tf.transpose(tf.broadcast_to(self.x, (self.n_nodes, self.t_size)))
        self.dist      = tf.sqrt(self.euclid(self.weights,self.big_x))
        self.best      = tf.argmin(self.dist, axis = 0)
        self.best_loc  = tf.reshape(tf.slice(self.nodes, tf.pad(tf.reshape(self.best, [1]), np.array([[0, 1]])), tf.cast(tf.constant(np.array([1, 2])), dtype=tf.int64)), [2])
        self.dlr       = tf.multiply(self.lr, 1 - self.div)
        self.radius    = tf.multiply(self.r,  1 - self.div)
        
    def optimiser(self):
        
        self.b_matrix  = tf.broadcast_to(self.best_loc, (self.size,2))
        self.b_dist    = self.euclid(self.nodes, self.b_matrix)
        self.surround  = tf.exp(tf.negative(tf.divide(tf.cast(self.b_dist, self.dtype), tf.cast(tf.square(self.radius), self.dtype))))
        self.lr_matrix = tf.multiply(self.dlr, self.surround)    
        self.factor    = tf.stack([tf.broadcast_to(tf.slice(self.lr_matrix, np.array([node]), np.array([1])), (self.n_nodes,)) for node in range(self.size)])        
        self.factor   *= tf.subtract(tf.squeeze(tf.stack([self.x for i in range(self.size)]),2), self.weights)                              
        fitted_weights = tf.add(self.weights, self.factor)
        
        self.optimise  = tf.assign(self.weights, fitted_weights)                                       
        
    def train(self, x_train):
        
        self.sess = tf.Session()
        init      = tf.global_variables_initializer()
        
        self.sess.run(init)
        
        self.optimiser()
        
        for iteration in tqdm(range(self.iters), leave = False):
            
            for train in x_train:
                
                self.sess.run(self.optimise, feed_dict={self.x: train.reshape(-1,1),self.progress: iteration})

        self.fitted_weights  = np.array(self.sess.run(self.weights))

    def fit(self, x):
        
        fitted         = np.empty((x.shape[0],2))

        node_distances = np.empty((x.shape[0], self.nodes.shape[0]))
        
        for j in range(x.shape[0]):
            
            distances_to_nodes = [np.linalg.norm(x[j,:] - self.fitted_weights[i]) for i in range(self.fitted_weights.shape[0])]
            
            node_distances[j]  = distances_to_nodes
            
            best_node          = self.nodes[np.argmin(distances_to_nodes)]
            
            fitted[j]          = best_node
        
        return fitted, node_distances
    
###################################################################################################################################################################################### 

class GPcluster:
    
    '''
    Gaussian Process Clustering
    
    Methods
    -------
    
    RBF(x1, x2, overwrite_l = False, new_m = 2)
        Compute the Radial Basis Function between two data points
    var_func(x, use_red = False, X = None)
        Compute the variance function
    get_r_star()
        Find the optimum radius
    gradient_func(x)
        Compute the gradient at x
    gradient_descent(x)
        Perform graident descent until convergence or specified number of iterations
    get_equilibirum()
        Find the stable equilibrium points of clusters
    adjacency_matrix()
        Compute the adjaceny matrix
    point_clusters()
        Return cluster labels of the stable equilibirum points
    data_clusters()
        Return clusters associated with the stable equilibirum points
    plot_2d(red_dims = False, n = 50)
        Simple 2d plot function with dimensionality reduction if dims > 2
    plot_3d(red_dims = False)
        Simple 3d plot function with dimensionality reduction if dims > 3
    fit(plot = True)
        Method that fits the Gaussian Process Clustering algorithm
    '''
    
    def __init__(self, X, 
                 iters   = 10, 
                 step    = 0.1, 
                 s2      = 1, 
                 l       = 1, 
                 alpha   = 1,
                 gamma   = 1,
                 cov_lim = 0.99,
                 p_Num   = 20,
                 latent  = 3, 
                 verbose = False):
        
        '''
        Parameters
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
        gamma: float
            Initial value for gamma (Default is 1)
        cov_lim: float
            Limits the maxmum covariance (Default is 0.99)
        p_Num:
            Determines the number of samples to be generated when fitting (Default is 20)
        latent: int
            Determines the size of the latent space if dimensionality reduction is required (Default is 3)
        verbose: Bool
            Boolean indicator for descriptive printing (Default is False)
        
        '''
        
        self.X       = X
        self.m       = self.X.shape[1]
        self.s2      = s2
        self.l_val   = [l]
        self.l       = np.array([l]*self.m)
        self.alpha   = alpha
        self.gamma   = gamma
        self.iters   = iters
        self.step    = step
        self.p_Num   = p_Num
        self.cov_lim = cov_lim
        self.verbose = verbose
        self.latent  = latent
    
    def RBF(self, x1, x2, overwrite_l = False, new_m = 2):
        
        '''
        Parameters
        ----------
        x1: np.array
            First observation
        x2: np.array
            Second observation
        overwrite_l: Bool
            Boolean indicator determining if lengthscale needs reinitialising to match the latent dimension size (default is True)
        new_m: int
            Size of latent space if dimensionality reduction is required (default is 2)
        '''
        
        l = self.l
        
        if overwrite_l:
            
            l = np.array(self.l_val*new_m)
        
        x1 = x1/(l**2)
        x2 = x2/(l**2)

        self.norm  = cdist(x1, x2, metric = 'sqeuclidean')
        
        return  self.s2*np.exp(-self.norm/(2))

    def var_func(self, x, use_red = False, X = None):
        
        '''
        Parameters
        ----------
        x: np.array
            Observation
        use_red: Bool
            Boolean indicator for dimensionality reduction (Default is False)
        X: np.array
            Dimensionality reduced input data
        '''
        
        if not use_red:
            
            X = self.X
        
        if len(x.shape) < 2:
            
            x = x.reshape(1,-1)
        
        PHI_X_X      = self.RBF(X, X, overwrite_l = use_red)
        
        PHI_X_x_star = self.RBF(X, x, overwrite_l = use_red)
        
        PHI_x_x      = self.RBF(x, x, overwrite_l = use_red)
        
        L   = np.linalg.cholesky(PHI_X_X + self.alpha * np.var(X, axis = 1) * np.eye(X.shape[0]))
        
        inv = scipy.linalg.cho_solve((L, True), np.eye(X.shape[0]))
        
        var = PHI_x_x - PHI_X_x_star.T @ inv @ PHI_X_x_star
        
        return var
    
    def get_r_star(self):
        
        variances   = np.diag(self.var_func(self.X))
        
        self.vmax   = np.max(variances)
        
        self.r_star = self.X[np.argmax(variances),:]
    
        return self.r_star

    def gradient_func(self, x):
        
        '''
        Parameters
        ----------
        x: np.array
            Observation
        '''
        
        PHI_X_X      = self.RBF(self.X, self.X)
        
        PHI_X_x_star = self.RBF(self.X, x)

        inv          = np.linalg.inv(PHI_X_X + self.alpha * self.s2 * np.eye(self.X.shape[0]))
        
        K_grad       = -PHI_X_x_star * (x - self.X)
        
        X_grad       = K_grad.T @ inv @ PHI_X_x_star + PHI_X_x_star.T @ inv @ K_grad

        return np.diag(X_grad)
    
    def gradient_descent(self, x):
        
        '''
        Parameters
        ----------
        x: np.array
            Observation
        '''
        
        x_ = x.copy()
        
        for i in range(self.iters):
            
            x = x + self.step * self.gradient_func(x)
            
            if np.all(np.isclose(x, x_, rtol = 1e-6)):
                
                break
            
            x_ = x.copy()
        
            if self.verbose:
        
                print ("iteration: " , i, "x0: ", x_, "xt: ", x)
        
        return x

    def get_equilibrium(self):
        
        points = np.empty(self.X.shape)
        
        for i in tqdm(self.X.shape[0], leave = False):
          
            points[i,:] = self.gradient_descent(self.X[i,:].reshape(1,-1))
        
        self.pca = PCA(n_components = self.latent)
        
        points = self.pca.fit(points).transform(points)
        
        self.f_points = points
        
        bool_check = (self.RBF(points, points, overwrite_l=True, new_m=self.latent) > self.cov_lim) - np.eye(points.shape[0]) == 1

        index = {}

        for i in range(self.X.shape[0]):

            index[i] = list(np.where(bool_check[i,:])[0])

        keep = []
        
        all_ = []
        
        filtered_dict = {}

        for k, v in index.items():

            if len(v) == 0:

                keep.append(k)
                
                filtered_dict[k] = v

            else: 

                if not any(p in keep for p in v):

                    keep.append(k)
                    
                    filtered_dict[k] = v

        for i in range(self.X.shape[0]):
            
            k_ret = [k for k,v in filtered_dict.items() if i in v]
            
            if len(k_ret) == 0:
            
                all_.append(i)
            
            else:
                
                k_ret = [i for i in k_ret if i in keep][0]
                
                all_.append(k_ret)

        self.data_assignments = np.array(all_)
                
        self.points = np.array(points[keep,:]) 
            
        return self.points
    
    def adjacency_matrix(self):
        
        p = self.points.shape[0]
        
        Adj = np.eye(p)
        
        for i in tqdm(p, leave = False):
            for j in range(p):
                
                if i < j:
                    
                    connected = True
                    delta = self.points[i] - self.points[j]
                    dist  = np.sqrt(np.sum(delta**2))
                    p_num = dist * self.p_Num
                    
                    for k in range(int(p_num)):
                    
                        sample     = np.array(self.points[j] + (k+1) * delta/p_num)
                        sample     = self.pca.inverse_transform(sample)
                        sample_var = self.var_func(sample.T)[0]
                        
                        if sample_var > self.gamma * self.vmax:   
                            connected = False
                            break

                    if connected:
                        Adj[i][j] = 1
                
                elif i > j:
                    Adj[i][j] =Adj[j][i]
      
            
        self.Adj = Adj
        
        return 
        
    def point_clusters(self):
        
        p               = self.points.shape[0]
        current_cluster = 1
        clusters        = np.zeros(p)
        
        for i in range(p):
            
            for j in range(p):
                
                if self.Adj[i][j] == 1:
                    
                    clusters[j] = current_cluster
            current_cluster +=1
                    
        return (clusters - clusters.min()).astype(int)
    
    def data_clusters(self):
        
        self.final_clusters = np.zeros(self.X.shape[0])
        
        self.cluster_map = {i:j for i,j in zip(np.unique(self.data_assignments), self.assigned_cluster_equilibria)}
    
        for i, item in enumerate(self.data_assignments):
            
             self.final_clusters[i] = self.cluster_map[item]
                
        return 
    
    def plot_2d(self, red_dims = False, n = 50):
        
        '''
        Parameters
        ----------
        red_dims: Bool
            Boolean indicator for dimensionality reduction (Default is False)
        n: int
            Number of values in the linspace
        '''
        
        X        = self.X
        points   = self.points
        f_points = self.f_points
        use_red  = False
        
        if red_dims:
            
            X        = PCA_(X, n=2, visualise = False)
            use_red  = True
            
        xrange    = (np.min(X[:,0]), np.max(X[:,0]))
        yrange    = (np.min(X[:,1]), np.max(X[:,1]))
        xsum      = np.sum(xrange)/4
        ysum      = np.sum(yrange)/4
        
        grid_x    = np.linspace(xrange[0]-xsum,xrange[1]+xsum,n)
        grid_y    = np.linspace(yrange[0]-ysum,yrange[1]+ysum,n)
        grid      = list(itertools.product(grid_x, grid_y))
        variances = np.diag(self.var_func(np.array(grid), use_red = use_red, X = X)).reshape(n,n)
        plt.figure(figsize=(12, 8))
        plt.subplot(2,2,1)
        plt.title('Data with Equilibrium Points')
        plt.scatter(X[:,0], X[:,1], c = 'b', alpha = 0.2)
        plt.scatter(f_points[:,0], f_points[:,1], c = 'r', s = 10, alpha = 0.2)
        plt.scatter(points[:,0], points[:,1], c = 'k', s = 5)
        plt.legend(['Data', 'Equilibrium Points', 'Reduced Eq Points'])
        plt.subplot(2,2,2)
        plt.title('Assigned Clusters for Equilibria')
        plt.scatter(points[:,0], points[:,1], c = self.assigned_cluster_equilibria+1)
        plt.subplot(2,2,3)
        plt.title('Clustered Data')
        plt.scatter(X[:,0], X[:,1], c = self.final_clusters, alpha = 1)   
        plt.subplot(2,2,4)
        plt.title('Heatmap of variance function')
        plt.scatter(points[:,0], points[:,1], c = 'b')
        plt.imshow(variances.T,  extent=[xrange[0],xrange[1],yrange[0],yrange[1]], cmap = 'hot')
        plt.colorbar()
        plt.show()
        
    def plot_3d(self, red_dims = False):
        
        '''
        Parameters
        ----------
        red_dims: Bool
            Boolean indicator for dimensionality reduction (Default is False)
        '''
        
        X        = self.X
        points   = self.points
        f_points = self.f_points
        
        if red_dims:
            
            X = PCA_(X, n=3, visualise = False)
            
        fig = plt.figure()
        ax  = plt.axes(projection='3d')
        plt.title('Data with Equilibrium Points')
        ax.scatter3D(X[:,0], X[:,1],X[:,2], c = 'b', alpha = 0.2)
        ax.scatter3D(f_points[:,0], f_points[:,1], f_points[:,2], c = 'r', s = 10, alpha = 0.1)
        ax.scatter3D(points[:,0], points[:,1], points[:,2], c = 'k', s = 5)
        plt.legend(['Data', 'Equilibrium Points', 'Reduced Eq Points'])
        plt.show()

        fig = plt.figure()
        ax  = plt.axes(projection='3d')
        plt.title('Clustered Data')
        ax.scatter(X[:,0], X[:,1], X[:,2], c = self.final_clusters, alpha = 1)            
        plt.show
    
    def fit(self, plot = True):
        
        '''
        Parameters
        ----------
        plot: Bool
            Boolean indicator for plots (Default is True)
        '''

        self.get_r_star()
        self.get_equilibrium()
        self.adjacency_matrix()
        self.assigned_cluster_equilibria = self.point_clusters()
        self.data_clusters()
        
        if plot:
            
            dims = self.X.shape[1]
            
            if dims == 2:
                
                self.plot_2d()
            
            elif dims == 3:
                
                self.plot_2d(red_dims = True)
                self.plot_3d()

        return self
    
#############################################################################################################################################
class support:
    
    '''
    Helper function for the DPGMM Class
    
    Methods
    -------
    
    calc_entropy(X)
        Vectorised log_entropy
    cholesky_decomp(A)
        Compute the cholesky decompositon of matrix A
    tensor_inv()
        Compute the inverse of a 3d tensor with 2d Cholesky decompositon
    log_det()
        Compute the log determinant of a 3d tensor
    lgamma()
        Compute the sum of the log of the gamma functions per Wishart dist.
    '''
    
    def calc_entropy(self, X):
        
        '''
        Parameters
        ----------
        X: ndarray
        '''
        
        log_params  = X.copy() - X.max(axis = 1)[:,None]
        params      = np.exp(log_params)
        log_params -= log_params.sum(axis = 1)[:, None] # Normalise in log space
        params     /= params.sum(axis = 1)[:, None] # Normalise
        entropy     = -np.einsum('ab, ab -> ', params, log_params)

        return params, log_params, entropy
    
    def cholesky_decomp(self, A):
        
        '''
        Parameters
        ----------
        A: ndarray
        '''
        
        chols  = [np.linalg.cholesky(A[:,:,i]) for i in range(A.shape[2])]
        
        chinvs = [np.linalg.inv(C) for C in chols]
        
        return chols, chinvs
    
    def tensor_inv(self, A):
        
        '''
        Parameters
        ----------
        A: ndarray
        '''
        
        chinvs = self.cholesky_decomp(A)[1]
        
        return np.dstack([C_inv.T @ C_inv for C_inv in chinvs])
    
    def log_det(self, A):
        
        '''
        Parameters
        ----------
        A: ndarray
        '''

        chols = self.cholesky_decomp(A)[0]
        
        return np.array([np.log(C.diagonal().prod()) for C in chols])
    
    def lgamma(self, v, D):
        
        return np.sum([gammaln((v+1.-d)/2.) for d in range(1,D+1)],0)
    

class DPGMM(support):
    
    '''
    Fit a Gaussian Mixture Model with a Dirichlet Process prior on the number of mixture components
    
    Methods
    -------
    
    calc_entropy(X)
        Vectorised log_entropy
    cholesky_decomp(A)
        Compute the cholesky decompositon of matrix A
    tensor_inv()
        Compute the inverse of a 3d tensor with 2d Cholesky decompositon
    log_det()
        Compute the log determinant of a 3d tensor
    lgamma()
        Compute the sum of the log of the gamma functions per Wishart dist.
    '''
    
    def __init__(self, X, clusters, 
                 iters = 500, 
                 step  = 1, 
                 alpha = 1, 
                 ftol  = 1e-6, 
                 gtol  = 1e-6,  
                 kap   = 1e-6, 
                 var_p = 1e-3, 
                 conc  = 10,
                 verb  = False):
        
        '''
        Parameters
        ----------
        X: ndarray
        clusters: int
            Initial number of clusters - updated by the DP prior
        iters: int
            Number of iterations in gradient descent (Default is 500)
        step: float
            Step size (Default is 1)
        alpha: float
            Initial value for alpha (Default is 1)
        ftol: float
            Tolerance for function value convergence (Default is 1e-6)
        gtol: float
            Tolerance for gradient value convergence (Default is 1e-6)
        kap: float
            Initial value for kappa (Default is 1e-6)
        var_p: float
            Initial value for the variance (Default is 1e-3)
        conc: float
            Intiial value for the concentration parameter (Default is 10)
        verb: Bool
            Boolean indicator for explantory prints (Default is False)
        '''
        
        ### Initialisation for the VBEM optimiser
        
        self.ftol  = ftol
        self.gtol  = gtol
        self.iters = iters
        self.step  = step
        
        ### Initialisation for the model params
        
        self.X             = X
        self.N             = self.X.shape[0]
        self.features      = self.X.shape[1]
        self.mu            = self.X.mean(axis = 0)
        self.kap_scalar    = kap
        self.var           = var_p * np.eye(self.features)
        self.ldet          = 0.5*(np.linalg.slogdet(self.var**0.5)[1])
        self.n_clusts      = clusters
        self.alpha         = alpha
        
        self.verbose       = verb
        
        if conc > 0: self.concentration = conc 
            
        else: self.concentration = self.n_clusts + 1.
            
        if self.concentration < self.n_clusts: print("Warning: Unstable behaviour when concentration parameter is less than the number of clusters")
        
        ### Initialisation for the mixture model components
        
        self.mu_kernel = self.mu[:,None]*self.mu[None,:]
        
        self.kernel    = self.X[:,:,None]*self.X[:,None,:]

        self.kappa_scaled_mu_kernel = self.kap_scalar * self.mu_kernel
        
        self.reparameterise(np.random.randn(self.N, self.n_clusts))
    
    def reparameterise(self,params):
        
        '''
        Get the updated parameter values
        '''

        self.params0  = params.reshape(self.N, self.n_clusts)
        entropy_vars  = self.calc_entropy(self.params0)
        self.params   = entropy_vars[0] # Vector of the parameters
        self.lparams  = entropy_vars[1] # Log params
        self.entropy  = entropy_vars[2]
        self.gradient = -self.lparams

        self.params_cluster = self.params.sum(axis = 0)

        self.kaps     = self.params_cluster + self.kap_scalar
        self.concs    = self.params_cluster + self.concentration
        self.alphs    = self.params_cluster + self.alpha
        self.alph_nrm = self.alphs/self.alphs.sum()
        self.weighted = np.einsum('ac, ab -> cb', self.X, self.params)
        self.CLUSETRS = np.einsum('ad, abc -> bcd', self.params, self.kernel)
        
        self.clustmean        = (self.kap_scalar*self.mu[:,None] + self.weighted)/self.kaps[None,:]
        self.clustmean_kernel = self.clustmean[:,None,:]*self.clustmean[None,:,:]
        self.clustvar         = self.var[:,:,None] + self.CLUSETRS + self.kappa_scaled_mu_kernel[:,:,None] - self.kaps[None,None,:]*self.clustmean_kernel
        self.clustvar_inv     = self.tensor_inv(self.clustvar)
        self.clust_ldet       = self.log_det(self.clustvar)
        self.demeaned_x       = self.X[:,:,None]-self.clustmean[None,:,:]
        self.demeaned_kernel  = self.demeaned_x[:,:,None,:]*self.demeaned_x[:,None,:,:]
        
    def ELBO(self):
        
        '''
        Compute the evidence lower bound
        '''
        
        cluster      = -0.5*self.features*np.sum(np.log(self.kaps/self.kap_scalar))
        
        dirichlet1   = self.n_clusts*self.concentration*self.ldet - np.sum(self.concs*self.clust_ldet)
        
        dirichlet2   = np.sum(self.lgamma(self.concs, self.n_clusts))- self.n_clusts*self.lgamma(self.concentration, self.n_clusts)
        
        mixing_ELBO1 = gammaln((np.ones(self.n_clusts)*self.alpha).sum())-np.sum(gammaln((np.ones(self.n_clusts)*self.alpha)))
        
        mixing_ELBO2 = gammaln((self.alpha + self.params.sum(axis = 0)).sum()) - np.sum(gammaln(self.alpha + self.params.sum(axis = 0)))
        
        entropy      = self.entropy
        
        likelihood1  = -0.5*self.N*self.features*np.log(np.pi)
        
        return cluster + dirichlet1 + dirichlet2 + mixing_ELBO1 - mixing_ELBO2 + entropy + likelihood1

    def fit(self):
        
        current_ELBO = self.ELBO()
        
        for i in range(self.iters):

            var_grad  = self.clustvar_inv[None,:,:,:]*self.demeaned_kernel
            ldet_grad = np.dot(np.ones(self.features), np.dot(np.ones(self.features), var_grad))

            grad_phi =  (-0.5*self.features/self.kaps + 0.5*digamma((self.concs-np.arange(self.features)[:,None])/2.).sum(0) + digamma(self.alpha + self.params.sum(axis = 0)) - self.clust_ldet -1.) + (self.gradient-0.5*ldet_grad*self.concs)

            direction = grad_phi - np.sum(self.params*grad_phi, 1)[:,None] # corrects for softmax (over) parameterisation
            gradient  = (direction*self.params).flatten()
            
            current_phi = self.params0.flatten().copy()
   
            self.reparameterise(current_phi + self.step*direction.flatten())
            
            ELBO = self.ELBO()
                
            function_change = np.fabs(ELBO - current_ELBO)
            gradient_change = direction.flatten() @ gradient
            
            function_converge = function_change < self.ftol
            gradient_converge = gradient_change < self.gtol
            
            if self.verbose: print('\riteration '+str(i)+' ELBO='+str(-ELBO) + ' grad='+str(gradient_change))
            
            if function_converge or gradient_converge:
                
                if self.verbose: print('Convergence')
                
                break
                
            current_ELBO = ELBO 
            
        keep   = self.params_cluster > 1e-8
        new_K  = np.sum(keep)
        self.n_clusts = new_K
        self.reparameterise(self.params0[:,keep])
        self.assigned_clusters = np.argmax(self.params, axis = 1)
        
        return
            
    def split_cluster(self, clust_num, threshold=0.9):
        
        '''
        Compare splits and make the split if it leads to an improvment in the ELBO
        
        Parameters
        ----------
        clust_num: int
            Cluster index
        threshold: float
            Threshold on probability of assignment (Default is 0.9)
        '''

        check1 = clust_num > (self.n_clusts-1) # in range
        check2 = self.params_cluster[clust_num] < 1 # data to split
        check3 = np.sum(self.params[:,clust_num] > threshold) < 2 #ensure there's something to split
        
        if check1 or check2 or check3: return False
    
        previous_ELBO  = self.ELBO()
        self.n_clusts += 1
        
        min_param  = self.params0.min(axis = 1)[:,None]
        new_param  = np.hstack((self.params0,min_param))
        
        valid_data    = np.where((self.params[:,clust_num] > threshold) != 0)[0]
        random_datum  = np.random.choice(valid_data, size = 1)[0]

        new_param[valid_data,-1]     = new_param[valid_data,clust_num]
        new_param[random_datum,-1]   = np.max(new_param[random_datum])
        
        self.reparameterise(new_param.flatten())
        self.fit()
        
        if self.ELBO() - previous_ELBO < 1e-5: return False
        
        else:  return True
        
    def split_all(self, iters = 10):
        
        '''Compare all possible cluster splits'''
        
        for _ in range(iters):
        
            for i in range(self.n_clusts):

                self.split_cluster(i)

    def predict(self, X):
        
        '''
        Predict cluster assignment of new data
        '''
        
        diff  = X[:,:,None]-self.clustmean[None,:,:]
        ein1  = np.einsum('abd, bcd -> acd', diff, self.clustvar_inv)        
        mhlb  = np.einsum('abc, abc -> ac', ein1, diff)/(self.kaps+1.)*self.kaps*(self.concs-self.features+1.)
        ldet  = self.clust_ldet + 0.5*self.features*np.log((self.kaps+1.)/(self.kaps*(self.concs-self.features+1.)))
        
        term1 = gammaln(0.5*(self.concs[None,:]+1.))
        term2 = - gammaln(0.5*(self.concs[None,:]-self.features+1.))
        term3 = - (0.5*self.features)*(np.log(self.concs[None,:]-self.features+1.) + np.log(np.pi))
        term5 = - (0.5*(self.concs[None,:]+1.))*np.log(1.+mhlb/(self.concs[None,:]-self.features+1.))
                                                     
        lpred = term1 + term2 + term3 + term5 - ldet

        pred  = np.exp(lpred)
        
        model_pred  = self.params_cluster + self.alpha
        mpred = np.einsum('ab, b -> a', pred, model_pred)
        
        mpred /= model_pred.sum()
        
        return mpred, pred

    def plot(self, n = 100, thresh = 1e-8):
        
        '''
        Plot the clusters in a 2d space over the variance contours
        
        Parameters
        ----------
        
        n: int 
            Range of linspace (Default is 100)
        thresh:
            Cutoff below which we assume outside of the variance function (Default is 1e-8)
        '''
        
        check_dims = self.X.shape[1] == 2
        
        if not check_dims:
            
            data_ = preprocess(self.X, white = False, reduce = True, n = 2)
            
            plt.scatter(data_[:,0], data_[:,1], c = self.assigned_clusters, s  = 10)
            
        else:
            
            xrange        = (np.min(self.X[:,0]), np.max(self.X[:,0]))
            yrange        = (np.min(self.X[:,1]), np.max(self.X[:,1]))
            xsum          = (xrange[1]-xrange[0])/10
            ysum          = (yrange[1]-yrange[0])/10
            grid_x        = np.linspace(xrange[0]-xsum,xrange[1]+xsum,n)
            grid_y        = np.linspace(yrange[0]-ysum,yrange[1]+ysum,n)
            grid          = np.array(list(itertools.product(grid_x, grid_y)))
            grid_contours = self.predict(grid)[0].reshape(n, n)
            grid_points   = self.predict(grid)[1] * self.alph_nrm[None,:]
            data_contours = self.predict(self.X)[0]
            grid_contours[grid_contours < thresh] = np.nan
            plt.contourf(grid_x, grid_y,grid_contours.T, alpha = 0.3)
            plt.scatter(self.X[:,0], self.X[:,1], 5, self.assigned_clusters, cmap = cm.inferno, alpha = 1)




