"""
Custom kernel library for Gaussian Processes including sparse kernels and cross-covariance terms

The choice for an appropriate covariance function is important, 
as the GP's output directly depends on it. These parameters of the covariance function are 
referred to as the hyperparameters of the GP, which can be either given by a fixed covariance scale 
and noise, or learned from data by optimising the marginal likelihood. To handle the computational problem 
of inverting a large covariance matrix, sparse covariance function are included here as well.
One important requirement for constructing covariance kernels is that they must be defined 
to be both positive semi-definite and informative. 

For more information on sparse covariances in GPs, see the following paper: 
"A Sparse Covariance Function for Exact Gaussian Process Inference in Large Datasets" (2009, Melkumyan and Ramos)

This package is part of the machine learning project developed for the Agricultural Research Federation (AgReFed).

Copyright 2022 Sebastian Haan, Sydney Informatics Hub (SIH), The University of Sydney

This open-source software is released under the AGPL-3.0 License.
"""
from scipy import reshape, sqrt, identity
from scipy.linalg import pinv, solve, cholesky, solve_triangular
from scipy.optimize import minimize, shgo
from scipy.special import erf
import numpy as np
# import local functions
from utils import print2


def optimize_gp_3D(points3d_train, Y_train, Ynoise_train, xymin, zmin, Xdelta=None):
    """
    Optimize GP hyperparmenters  of amplitude, noise, and lengthscales
    Using Global optimisation shgo with sobol sampling.

    INPUT
        points3d_train: Array with training point coordinates for (z,y,x)
        Y_train: Training Data Vector
        Ynoise_train: Noise of Training data
        xymin: minimum GP lengthscale in x and y direction
        zmin: minimum GP lengthscale in vertical (z) direction
        Xdelta: 3D array (x,y,z) with uncertainty in data positions, same shape as points3d_train

    RETURN:
        Optimised Hyperparamters of best solution: (amplitude, noise, z_lengthscale, xy_lengthscale)
        Marginbal Log Likelihood of best solution
    """

    def calc_nlogl3D(gp_params, *args):
        # Calculate marginal loglikelihood
        gp_amp = gp_params[0]
        gp_noise = gp_params[1]
        gp_length = np.asarray([gp_params[2], gp_params[3], gp_params[3]])	
        D2, Y, Ynoise, Delta = args
        if Delta is not None:
            #kcov = gp_amp * (gpkernel_sparse_multidim2_noise(D2, gp_length, Delta)) + np.eye(len(D2[0,0])) * noise**2
            kcov = gp_amp * gpkernel_sparse_multidim_noise(D2, gp_length, Delta) + gp_noise * np.diag(Ynoise**2)
        else:
            #kcov = gp_amp * (gpkernel_sparse_multidim2(D2, gp_length)) + np.eye(len(D2[0,0])) * noise**2
            kcov = gp_amp * gpkernel_sparse_multidim(D2, gp_length) + gp_noise * np.diag(Ynoise**2)
        try:
            k_chol = cholesky(kcov, lower=True)
            Ky = solve_triangular(k_chol, Y, lower=True).flatten()
            log_det_k= 2 * np.log(np.diag(k_chol)).sum()
            n_log_2pi = Ntrain * np.log(2 * np.pi)
            logl = -0.5 * (np.dot(Ky, Ky) + log_det_k + n_log_2pi)
        except:
            logl = -np.nan
        return -logl

    Dtrain = calcDistanceMatrix_multidim(points3d_train)
    if Xdelta is not None:
        Delta_00 = calcDeltaMatrix_multidim(Xdelta)
    else:
        Delta_00 = None
    ystd = Y_train.std()
    ymean = Y_train.mean()
    Y = (Y_train - ymean) / ystd
    Ynoise = Ynoise_train / ystd
    print('Mean Input Noise: ', np.mean(Ynoise))
    Ntrain = len(points3d_train)

    # Optimize hyperparameters
    # Global optimisation
    # SHGO is disabled for now (too slow and latest scipy update made it unstable)
    # TBD: test dual_annealing with scipy.optimize.dual_annealing?
    # res = shgo(calc_nlogl3D, n=20, iters =20,
    #             #bounds=[(0.001, 1000), (ynoise_min, 1), (zmin, zmin*1000), (xymin, xymin*1000)],
    #             bounds=[(0.01, 10), (0.01, 2.0), (zmin, zmin*2000), (xymin, xymin*4000)],
    #             sampling_method='sobol',
    #             args = (Dtrain, Y, Ynoise, Delta_00))

    # Local optimisation
    """
    Some common local optimiser choices
    'BFGS': Broyden-Fletcher-Goldfarb-Shanno algorithm
    'SLSQP': Sequential Least Squares Programming
    'COBYLA': Constrained Optimization BY Linear Approximation
    'L-BFGS-B': Limited memory BFGS
    'Powell': Powell's conjugate direction method
    """
    optimiser = 'Powell'
    res = minimize(calc_nlogl3D, x0 = [1, 0.1, 10*zmin, 10*xymin],
                bounds=[(0.01, 10), (0.001, 2.0), (zmin, zmin*1000), (xymin, xymin*1000)],
                method=optimiser,
                args = (Dtrain, Y, Ynoise, Delta_00))         
    if not res.success:
        # Don't update parameters
        print('WARNING: ' + res.message) 
    else:
        print2(f'Optimized Hyperparameters (amplitude, y_noise_fac, lengthscale_z, lengths_xy): {res.x}')
        print('Marginal Log Likelihood: ', -res.fun)
    return res.x, -res.fun


	
def train_predict_3D(points3D_train, points3D_pred, Y_train, Ynoise_train, params_gp, Ynoise_pred = None, Xdelta = None,  calclogl = True, save_gptrain = True, out_covar = False):
	"""
	Train adn predict mean and covariance of GP model.

	INPUT
	    points3d_train: Array with trainig point coodinates for (z,y,x)
	    points3d_pred: Array with point coodinates to be predicted in form (z,y,x)
	    Y_train: vector of training data with same length as points3d_train
	    Ynoise_train: noise data with same length as points3d_train
	    params_gp: list of hyperparameters as (amplitude, noise_std, lengthscale_z, lengthscale_xy)
	    Ynoise_pred: noise data of the mean functio for predicted location, same length as points3d_pred
	    Xdelta: Uncertainity in X coordinates, same shape as points3d_train
	    calclogl: If True (default), calculates and returns the marginal log-likelihood of GP
	    save_gptrain: if True (default), returns list of arrays (gp_train) such as cholesky factors that 
				can be reused for any other prediction based on same training data (for instance to split-up prediction in blocks).

	RETURN:
	    predicted mean
	    predicted uncertainty stddev
	    marginal log likelihood
	    gp_train
	"""
	Ntrain = len(points3D_train)

	# Standardize data:
	ystd = Y_train.std()
	ymean = Y_train.mean()
	Y = (Y_train - ymean) / ystd
	Ynoise = Ynoise_train / ystd
	#Y = Y.reshape(-1,1)

	# Calculate Distance Matrixes: 
	D2_00 = calcDistanceMatrix_multidim(points3D_train)
	D2_01 = calcDistance2Matrix_multidim(points3D_pred, points3D_train)
	D2_11 = calcDistanceMatrix_multidim(points3D_pred)


	# Set GP hyperparameter
	params_gp = np.asarray(params_gp)
	gp_amp = params_gp[0]
	gp_noise = params_gp[1]
	gp_length = (params_gp[2], params_gp[3], params_gp[3])

	# noise of mean function for prediction points
	if Ynoise_pred is not None:
		Ynoise2 = Ynoise_pred / ystd
	else:
		Ynoise2 = np.ones(len(points3D_pred))


	# if with noise in position
	if Xdelta is not None:
		Delta_00 = calcDeltaMatrix_multidim(Xdelta)
		Delta_01 = calcDelta2Matrix_multidim(np.zeros_like(points3D_pred), Xdelta)
		kcov00 = gp_amp * gpkernel_sparse_multidim_noise(D2_00, gp_length, Delta_00) + gp_noise * np.diag(Ynoise**2)
		kcov01 = gp_amp * gpkernel_sparse_multidim_noise(D2_01, gp_length, Delta_01) 	
	else:
		kcov00 = gp_amp * gpkernel_sparse_multidim(D2_00, gp_length) + gp_noise * np.diag(Ynoise**2)
		kcov01 = gp_amp * gpkernel_sparse_multidim(D2_01, gp_length) 
	kcov11 = gp_amp * gpkernel_sparse_multidim(D2_11, gp_length) + gp_noise * np.diag(Ynoise2**2) 

	try:
		k_chol = cholesky(kcov00, lower=True)
	except:
		print("Cholesky decompostion failed, kov matrix is likely not positive semitive.")
		print("Change GP parameter settings")
		sys.exit(1)
	Ky = solve_triangular(k_chol, Y, lower=True).flatten() #shape(2*Nsensor)
	v = solve_triangular(k_chol, kcov01, lower=True)
	# predicted mean
	mu = np.dot(v.T, Ky)
	# predicted covariance
	covar = kcov11 - np.dot(v.T, v)
	if (Ynoise_pred is not None) & (gp_amp < 1):
		#Caclulate diaginal elements as amplitude weighted average of  noise and GP noise
		varcomb = gp_amp * np.diag(covar) + (1 - gp_amp) * Ynoise2**2 
		np.fill_diagonal(covar, varcomb)
	# Calculate marginal log likelihood
	if calclogl:
		log_det_k=  2 * np.sum(np.log(np.diag(k_chol)))
		n_log_2pi = Ntrain * np.log(2 * np.pi)
		logl = -0.5 * (np.dot(Ky, Ky) + log_det_k + n_log_2pi)
	else:
		logl = 0.
	# Transform predicted data back to original range:
	ypred =  mu * ystd + ymean  
	yvar = np.diag(covar) * ystd**2
	print('Logl: ', logl)
	# Save matrix decomposition of training data for subsequent prediction (so cholesky etc don't need to be computed again)
	if save_gptrain:
		gp_train = (k_chol, Ky, ymean, ystd)
	else:
		gp_train = 0
	if out_covar:
		return ypred, np.sqrt(yvar), logl, gp_train, covar * ystd**2
	else:
		return ypred, np.sqrt(yvar), logl, gp_train


def predict_3D(points3D_pred, gp_train, params_gp, Ynoise_pred = None, Xdelta = None, out_covar = False):
	"""
	Predict mean and covariance based on trained GP. 
	This caluclation saves time as cholesky decompsoition  is already pre-computed. 

	INPUT
	    points3d_pred: Array with point coodinates to be predicted in form (z,y,x)
	    gp_train: list with precomputed GP (k_chol, Ky, ymean, ystd)
	    params_gp: list of hyperparameters as (amplitude, noise_std, lengthscale_z, lengthscale_xy)
	    Ynoise_pred: noise data of the mean functio for predicted location, same length as points3d_pred
	    Xdelta: Uncertainty in X coordinates, same shape as points3d_train

	RETURN:
	    predicted mean
	    predicted uncertainty stddev
	"""
	k_chol, Ky, ymean, ystd = gp_train

	# noise of mean function for prediction points
	if Ynoise_pred is not None:
		Ynoise2 = Ynoise_pred / ystd
	else:
		Ynoise2 = np.ones(len(points3D_pred))

	# Calculate Distance Matrixes: 
	D2_01 = calcDistance2Matrix_multidim(points3D_pred, points3D_train)
	D2_11 = calcDistanceMatrix_multidim(points3D_pred)

	# Set GP hyperparameter
	params_gp = np.asarray(params_gp)
	gp_amp = params_gp[0]
	gp_noise = params_gp[1]
	gp_length = (params_gp[2], params_gp[3], params_gp[3])
	# Calculate Caovariance Functions
	# if with noise
	if Xdelta is not None:
		Delta_01 = calcDelta2Matrix_multidim(np.zeros_like(points3D_pred), Xdelta)
		kcov01 = gp_amp * gpkernel_sparse_multidim_noise(D2_01, gp_length, Delta_01) 	
	else:
		kcov01 = gp_amp * gpkernel_sparse_multidim(D2_01, gp_length) 
	kcov11 = gp_amp * gpkernel_sparse_multidim(D2_11, gp_length) + gp_noise * np.diag(Ynoise2**2) 
	v = solve_triangular(k_chol, kcov01, lower=True)
	# predicted mean
	mu = np.dot(v.T, Ky)
	# predicted covariance
	covar = kcov11 - np.dot(v.T, v)
	# if (Ynoise_pred is not None) & (gp_amp < 1):
	# 	#Caclulate diaginal elements as amplitude weighted average of noise and GP noise
	# 	varcomb = gp_amp * np.diag(covar) + (1 - gp_amp) * Ynoise2**2 
	# 	np.fill_diagonal(covar, varcomb)
	# Transform predicted data back to original range:
	ypred =  mu * ystd + ymean  
	#yvar = np.diag(covar) * ystd**2
	yvar = np.diag(covar) * ystd**2
	if out_covar:
		return ypred, np.sqrt(yvar), covar * ystd**2
	else:
		return ypred, np.sqrt(yvar)



def calcGridPoints3D(Lpix, pixscale):
    """
    returns grid points for distance matrix calculation.

    INPUT
        Lpix: number of pixels in each dimension as array (xLpix, yLpix, zLpix)
        pixscale: pixelscale in each dimension as array (xpixscale, ypixscale, zpixscale)

    RETURN
        gridpoints: array with grid points in form (z,y,x)
    """
    Lpix = np.asarray(Lpix)
    pixscale = np.asarray(pixscale)
    xLpix, yLpix, zLpix = Lpix[0], Lpix[1], Lpix[2]
    xpixscale, ypixscale, zpixscale = pixscale[0], pixscale[1], pixscale[2]
    xrange = np.arange(1, xLpix+1) * xpixscale
    yrange = np.arange(1, yLpix+1) * ypixscale
    zrange = np.arange(1, zLpix+1) * zpixscale
    _xg, _yg, _zg = np.meshgrid(xrange, yrange, zrange)
    xr, yr, zr = _xg.ravel(), _yg.ravel(), _zg.ravel()
    return np.asarray([xr, yr, zr]).T


def calcDistanceMatrix(nDimPoints, 
                       distFunc=lambda deltaPoint: sum(deltaPoint[d]**2 for d in range(len(deltaPoint)))):
    """ Returns the matrix of squared distances between coordinates in nDimPoints.
    
    INPUT
        nDimPoints: list of n-dim tuples
        distFunc: calculates the distance based on the differences

    RETURN
        distanceMatrix: n x n matrix of squared distances
    """
    nDimPoints = np.array(nDimPoints)
    dim = len(nDimPoints[0])
    delta = [None]*dim
    for d in range(dim):
        data = nDimPoints[:,d]
        delta[d] = data - np.reshape(data,(len(data),1)) # computes all possible combinations

    dist = distFunc(delta)
    #dist = dist + np.identity(len(data))*dist.max() # eliminate self matching
    # returns  squared distance:
    return dist 

def calcDistance2Matrix(nDimPoints0, nDimPoints1,
                       distFunc=lambda deltaPoint: sum(deltaPoint[d]**2 for d in range(len(deltaPoint)))):
    """ Returns the matrix of squared distances between two cooridnate sets.

    INPUT
        nDimPoints0: list of n-dim tuples
        nDimPoints1: list of n-dim tuples with same dimension as nDimPoints1
        distFunc: calculates the distance based on the differences

    RETURN
        distanceMatrix: n x n matrix of squared distances
    """
    nDimPoints0 = np.array(nDimPoints0)
    nDimPoints1 = np.array(nDimPoints1)
    dim = len(nDimPoints0[0])
    assert len(nDimPoints1[0]) == dim
    delta = [None]*dim
    for d in range(dim):
        data0 = nDimPoints0[:,d]
        data1 = nDimPoints1[:,d]
        delta[d] = data0 - np.reshape(data1,(len(data1),1)) # computes all possible combinations
    dist = distFunc(delta)
    #dist = dist + np.identity(len(data))*dist.max() # eliminate self matching
    # returns  squared distance:
    return dist 

def calcDistanceMatrix_multidim(nDimPoints):
    """ Returns the matrix of squared distances between points in multiple dimensions. 

    INPUT
        nDimPoints: list of n-dim tuples
        distFunc: calculates the distance based on the differences

    RETURN
        distanceMatrix: n x n matrix of squared distances
    """
    nDimPoints = np.array(nDimPoints)
    dim = len(nDimPoints[0])
    delta = [None]*dim
    for d in range(dim):
        data = nDimPoints[:,d]
        delta[d] = abs(data - np.reshape(data,(len(data),1))) # computes all possible combinations
    return np.asarray(delta) 

def calcDistance2Matrix_multidim(nDimPoints0, nDimPoints1):
    """ Returns the matrix of squared distances between to corrdinate sets in multiple dimensions.

    INPUT
        nDimPoints0: list of n-dim tuples
        nDimPoints1: list of n-dim tuples with same dimension as nDimPoints1
        distFunc: calculates the distance based on the differences

    RETURN
        distanceMatrix: n x n matrix of squared distances
    """
    nDimPoints0 = np.array(nDimPoints0)
    nDimPoints1 = np.array(nDimPoints1)
    dim = len(nDimPoints0[0])
    assert len(nDimPoints1[0]) == dim
    delta = [None]*dim
    for d in range(dim):
        data0 = nDimPoints0[:,d]
        data1 = nDimPoints1[:,d]
        delta[d] = abs(data0 - np.reshape(data1,(len(data1),1))) # computes all possible combinations
    return np.asarray(delta)


def calcDeltaMatrix_multidim(nDimPoints):
    """ Returns the matrix of the sum of data points from one coordinate to any other

    INPUT
        nDimPoints: list of n-dim tuples

    RETURN
        deltaMatrix: n x n matrix of squared distances
    """
    nDimPoints = np.array(nDimPoints)
    dim = len(nDimPoints[0])
    delta = [None]*dim
    for d in range(dim):
        data = nDimPoints[:,d]
        delta[d] = abs(data + np.reshape(data,(len(data),1))) # computes all possible combinations
    return np.asarray(delta) 

def calcDelta2Matrix_multidim(nDimPoints0, nDimPoints1):
    """ Returns the matrix of the sum of data points between two coordinate sets    

    INPUT
        nDimPoints0: list of n-dim tuples
        nDimPoints1: list of n-dim tuples with same dimension as nDimPoints1

    RETURN
        deltaMatrix: n x n matrix of squared distances
    """
    nDimPoints0 = np.array(nDimPoints0)
    nDimPoints1 = np.array(nDimPoints1)
    dim = len(nDimPoints0[0])
    assert len(nDimPoints1[0]) == dim
    delta = [None]*dim
    for d in range(dim):
        data0 = nDimPoints0[:,d]
        data1 = nDimPoints1[:,d]
        delta[d] = abs(data0 + np.reshape(data1,(len(data1),1))) # computes all possible combinations
    return np.asarray(delta)


def calc_square_distances2D(Lpix, pixscale):
    """
    Initialize (squared) distance matrix for stationary kernel.

    INPUT
        Lpix: number of pixels in each dimension
        pixscale: pixel scale in arcsec/pixel

    RETURN
        dist: squared distance matrix
    """
    Lpix = np.asarray(Lpix)
    pixscale = np.asarray(pixscale)
    xLpix, yLpix = Lpix[0], Lpix[1]
    xpixscale, ypixscale = pixscale[0], pixscale[1]
    xrange = (np.arange(0, xLpix) - xLpix/2.0) * xpixscale
    yrange = (np.arange(0, yLpix) - xLpix/2.0) * ypixscale
    _xg, _yg = np.meshgrid(xrange, yrange)
    xr, yr = _xg.ravel(), _yg.ravel()
    Dx = xr[:, np.newaxis] - xr[np.newaxis,:]
    Dy = yr[:, np.newaxis] - yr[np.newaxis,:]
    return Dx**2 + Dy**2



def gpkernel_sparse_multidim_noise(Delta, gamma, sigma_delta = None):
    """
    Multi-dimensional RBF kernel, defined in Melkumyan and Ramos, following Eq 9, 12
    lengthscale is roughly equivalent to 4 times the lengthcale of squared exponential

    INPUT
        Delta: pairwise square distances for each dimension
        gamma: kernel length scale for each dimension
        delta: uncertainty in pairwise distance, same shape as delta

    RETURN
        K: kernel matrix
    """
    Delta = np.asarray(Delta)
    gamma = np.asarray(gamma)
    if sigma_delta is not None:
        sigma_delta = np.asarray(sigma_delta)
    else:
        sigma_delta = np.zeros_like(Delta)
    ndata = Delta[0].shape
    dim = Delta.shape[0]
    assert len(gamma) == dim
    kres = np.ones(ndata)
    for d in range(dim):
        res = np.zeros(ndata)
        Di = Delta[d]
        l = gamma[d] + sigma_delta[d]
        rat = sigma_delta[d]/gamma[d]
        res[Di < l] = ((2 + np.cos(2*np.pi * Di[Di < l] /l[Di < l]))/3.*(1-Di[Di < l] /l[Di < l]) + 1/(2.*np.pi) * np.sin(2*np.pi*Di[Di < l] /l[Di < l]))/(1+rat[Di < l])
        # Remove floating errors
        res[res < 0.] = 0.
        kres *= res
    return kres


def gpkernel_sparse_multidim(Delta, gamma):
    """
    Multi-dimensional RBF kernel, defined in Melkumyan and Ramos, following Eq 9, 12
    lengthscale is roughly equivalent to 4 times the lengthcale of squared exponential

    INPUT
        Delta: pairwise square distances for each dimension
        gamma: kernel length scale for each dimension

    RETURN
        K: kernel matrix
    """
    Delta = np.asarray(Delta)
    gamma = np.asarray(gamma)
    ndata = Delta[0].shape
    dim = Delta.shape[0]
    assert len(gamma) == dim
    kres = np.ones(ndata)
    for d in range(dim):
        res = np.zeros(ndata)
        Di = Delta[d]
        l = gamma[d]
        res[Di < l] = (2 + np.cos(2*np.pi * Di[Di < l] /l))/3.*(1-Di[Di < l] /l) + 1/(2.*np.pi) * np.sin(2*np.pi*Di[Di < l] /l)
        # Remove floating errors
        res[res < 0.] = 0.
        kres *= res
    return kres


def gpkernel_sparse_multidim2(Delta, gamma):
    """
    Multi-dimensional RBF kernel, defined in Melkumyan and Ramos, following Eq 13, 20
    lengthscale is roughly equivalent to 4 times the lengthcale of squared exponential

    INPUT
        Delta: pairwise square distances for each dimension
        gamma: kernel length scale for each dimension

    RETURN
        K: kernel matrix
    """
    Delta = np.asarray(Delta)
    gamma = np.asarray(gamma)
    ndata = Delta[0].shape
    dim = Delta.shape[0]
    assert len(gamma) == dim
    r2 = np.zeros(ndata)
    res = np.zeros(ndata)
    for d in range(dim):
        r2 += (Delta[d] / gamma[d])**2  
    r = np.sqrt(r2)
    res[r < 1] = (2 + np.cos(2*np.pi * r[r < 1]))/3.*(1-r[r < 1]) + 1/(2.*np.pi) * np.sin(2*np.pi*r[r < 1])
    return res

def gpkernel_sparse_multidim2_noise(Delta, gamma, sigma_delta = None):
    """
    Modified Multi-dimensional RBF kernel with X unicertainity, 
    original defined in Melkumyan and Ramos 2009, following Eq 13, 20
    lengthscale is roughly equivalent to 4 times the lengthcale of squared exponential

    INPUT   
        Delta: pairwise square distances for each dimension
        gamma: kernel length scale for each dimension
        delta: uncertainty in pairwise distance, same shape as delta    

    RETURN
        K: kernel matrix
    """
    Delta = np.asarray(Delta)
    gamma = np.asarray(gamma)
    ndata = Delta[0].shape
    if sigma_delta is not None:
        sigma_delta = np.asarray(sigma_delta)
    else:
        sigma_delta = np.zeros_like(Delta)
    dim = Delta.shape[0]
    assert len(gamma) == dim
    r2 = np.zeros(ndata)
    res = np.zeros(ndata)
    for d in range(dim):
        r2 += (Delta[d] / (gamma[d] + sigma_delta[d]))**2  
    r = np.sqrt(r2)
    res[r < 1] = (2 + np.cos(2*np.pi * r[r < 1]))/3.*(1-r[r < 1]) + 1/(2.*np.pi) * np.sin(2*np.pi*r[r < 1])
    return res


def gpkernel_sparse(D2, gamma):
    """
    2-D rsparse RBF kernel with , defined in Melkumyan and Ramos, 2009, Eq 13 
    lengthscale is roughly equivalent to 4 times the lengthcale of squared exponential
    
    INPUT
        D2: pairwise square distances
        gamma: kernel length scale

    RETURN
        K: kernel matrix
    """
    D2 = np.sqrt(D2)
    #gamma = 4 * gamma
    res = np.zeros_like(D2)
    res[D2 < gamma] = (2 + np.cos(2*np.pi * D2[D2 < gamma] /gamma))/3.*(1-D2[D2 < gamma] /gamma) + 1/(2.*np.pi) * np.sin(2*np.pi*D2[D2 < gamma] /gamma)
    # Remove floating errors
    res[res<0.] = 0.
    return res
