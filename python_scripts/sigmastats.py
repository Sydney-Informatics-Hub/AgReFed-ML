# Function to calculate weighted mean and uncertainity for uncorrelated AND correlated values

import numpy as np
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt


def averagestats(x, var):
	"""
	Compute weighted mean and uncertainty for correlated and uncorrelated measurements with uncertainties.
	For uncorrelated errors the weighted mean is
	wmean = sum(x/var) / sum(1/var)

	and  the variance of the weighted mean is:
	wsigma2 = 1 / sum (1/var) 

	INPUT
	-----
	x: 1D array of values
	var: covariance or variance of x. 
		For correlated measurements, provide the covariance matrix as 2D array with shape (length(x), length(x)).
		For uncorrelated measurments, provide the variance of x either as 1D array with same lengths as x, 
		or as 2D array with shape (length(x), length(x)) with the variance elements on the diagonal axis
		and the off-diagonal elements beeing zero.
		If sigma is only a constant, all x values are assumed uncorrelated and will be averaged with same weight.	

	Return:
	-------
	Mean: the weighted mean
	sigma: the weighted stddev
	"""

	
	if not hasattr(x, "__len__") | (x.size == 1):
		return np.asarray([x]), np.asarray([np.sqrt(var)])
	if not hasattr(var, "__len__") | (var.size == 1):
		var = np.diag(var * np.ones(len(x)))
		#print("Calculating average with constant variance")
	elif var.size == len(x):
		#print("Calculating average with uncorrelated variance")
		var = np.diag(var)
	elif var.shape == (len(x),len(x)):
		calc_covar = True
		#Assume correlated meausrements given by covariance sigma
		#print("Calculating average with covariance matrix var")
	else:
		print("Unsupported input for var")

	if (len(x) == 1) & (len(var) == 1):
		return x,  np.sqrt(var)

	# To Do: filter for only finite elements and covaraince diagonal elements larger than zero
	var[np.isnan(var) | ~np.isfinite(var)] = 1e9

	#Calculate weights
	inv = np.linalg.inv(var)
	w = np.nansum(inv, axis=1) / np.nansum(inv)
	# Scale sum of w to 1
	w = w / np.sum(w)
	#assert np.sum(w) = 1
	#Calculate weighted mean
	wmean = np.nansum(w * x)
	#Calculate error of mean
	wmat1, wmat2 = np.meshgrid(w,w)
	w2 = wmat1 * wmat2
	wsigma2 = np.nansum(w2 * var)
	if hasattr(wsigma2, "__len__") & ((wsigma2 < 0) | ~np.isfinite(wsigma2)).any():
		wsigma2[(wsigma2 < 0) | ~np.isfinite(wsigma2)] = np.nan
	elif wsigma2 < 0:
		wsigma2 = np.nan

	return wmean, np.sqrt(wsigma2)


def calc_featurecorrelations(X, feature_names, thresh = 0.9, plot = False):
	"""
	Identifying correlated (multicollinear) features by performing hierarchical clustering 
	on the Spearman rank-order correlations. 
	A threshold is used to identify correlated fearures

	INPUT
	-----
	X: data array of feautres with shape (nsample, nfeatures)
	feature_names: list of fetaure names
	tresh: treshold 
	plot: if Tru, plots dendogram of correlation

	Return
	------
	correlated feature array
	index array of corralted features for input features
	correlation coeffcients

	"""
	names = np.asarray(feature_names)
	corr = spearmanr(X).correlation
	corr_linkage = hierarchy.ward(corr)
	# keep only upper triangle of corr
	corr2 = np.triu(corr, k=1)
	#sel correlation large than 
	sel = np.where(corr2 >= thresh)
	if plot:
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
		dendro = hierarchy.dendrogram(corr_linkage, labels=feature_names, ax=ax1, leaf_rotation=90)
		dendro_idx = np.arange(0, len(dendro['ivl']))
		ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']])
		ax2.set_xticks(dendro_idx)
		ax2.set_yticks(dendro_idx)
		ax2.set_xticklabels(dendro['ivl'], rotation='vertical')
		ax2.set_yticklabels(dendro['ivl'])
		fig.tight_layout()
		plt.show()
	return np.asarray([names[sel[0]], names[sel[1]]]).T, corr[sel], sel


def calc_change(X1, X2, var_X1, var_X2, cov_X1X2 = None):
	"""
	Calculate change in feature values from one time step to the next.

	delta = X2 - X1

	sigma = sqrt(sigma1**2 + sigma2**2 - 2*cov_X1X2)

	INPUT
	-----
	X1: data array of values at time t1
	X2: data array of values at time t2
	var_X1: variance of X1
	var_X2: variance of X2
	cov_X1X2: covariance array of X1 and X2 with shape (size(X1), (size(X2)))

	Return
	------
	delta: change in feature values X2 - X1
	sigma: uncertainty of change in feature values
	"""

	delta = X2 - X1
	if cov_X1X2 is None:
		sigma = np.sqrt(var_X1 + var_X2)
	else:
		sigma = np.sqrt(var_X2 + var_X1 - 2*cov_X1X2)
	return delta, sigma
		





