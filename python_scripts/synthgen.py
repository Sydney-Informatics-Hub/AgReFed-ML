"""
Toolset for generating geospatial synthetic data-sets with a range of noise and spatial correlations 

Requirements:
- python>=3.9
- matplotlib>=3.5.1
- numpy>=1.22.0
- pandas>=1.3.5
- PyYAML>=6.0
- scikit_learn>=1.0.2
- scipy>=1.7.3


This package is part of the machine learning project developed for the Agricultural Research Federation (AgReFed).

Copyright 2022 Sebastian Haan, Sydney Informatics Hub (SIH), The University of Sydney

This open-source software is released under the AGPL-3.0 License.
"""

import os
import itertools
import sys
import yaml
import shutil
import argparse
from types import SimpleNamespace  
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, RobustScaler
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from sklearn.metrics import pairwise_distances
import matplotlib as mpl
import matplotlib.pyplot as plt


def create_kernel_uniform(r):
    """
    Create a circular kernel buffer

    Input: 
        r: radius of the kernel in units

    Return:
        kernel: kernel buffer
    """
    #Make an list of indexes 
    Y,X = np.ogrid[0:2*r-1, 0:2*r-1]
    dist_from_center = np.sqrt((X-r+1)**2 + (Y-r+1)**2)+1
    mask = dist_from_center <= r
    return(mask*1.0)

def create_kernel_expsquared(D, gamma):
    """
    Create exponential squared kernel

    Input: 
        X: distance matrix
        gamma: length scale parameter

    Return:
        kernel: kernel buffer
    """
    return np.exp(-D**2 / (2*gamma**2))


def gen_synthetic(n_features, n_informative_features = 10, 
                n_samples = 200,  outpath = None, 
                model_order = 'quadratic', correlated = False, 
                noise= 0.1, corr_length = 10, corr_amp = 0.2, 
                spatialsize = 100, center = [0,0],  crs = 'EPSG:8059'):
	"""
	Generate synthetic datasets

	Input:
		n_features: number of features
        n_infromative_features: number of important features
		n_samples: number of samples
        outpath: path to save simulated data	
		model_order: order of the model, either 'linear', 'quadratic', or 'cubic'
		correlated: if True, the features are correlated
		noise: random noise level [range: 0-1] that is added to the synthetic data
        corr_length: spatial correlation length [Gaussian FWHM in meters]
        corr_amp: spatial correlation amplitude
        spatialsize: size in x and y direction [in meters]
        center: [x,y] coordinates of the center of the data in meter (Easting, Northings)
        crs: coordinate reference system [Default: 'EPSG:8059']


	Return:
		dfsim: dataframe with simulated features
		coefsim: simulated coefficients
		feature_names: list of feature names
	"""
    # Initiate Random generator
    random_state = 42
    np.random.seed(random_state)
	if correlated:
		n_rank = int(n_features/2)
	else:
		n_rank = None
	Xsim, ysim, coefsim = make_regression(n_samples=_n_samples, n_features = n_features, n_informative=n_informative, n_targets=1, 
		bias=0.5, noise=noise, shuffle=False, coef=True, random_state=random_state, effective_rank = n_rank)	
	feature_names = ["Feature_" + str(i+1) for i in range(n_features)]
	coefsim /= 100
	scaler = MinMaxScaler()
	scaler.fit(Xsim)
	Xsim = scaler.transform(Xsim)
	# Make model
	if model_order == 'linear':
        # make first-order model
		ysim_new = np.dot(Xsim, coefsim)
	elif model_order == 'quadratic':
		# make quadratic model
		Xcomb = []
		for i, j in itertools.combinations(Xsim.T, 2):
			Xcomb.append(i * j) 
		Xcomb = np.asarray(Xcomb).T
		Xcomb = np.hstack((Xsim, Xcomb, Xsim**2))
		coefcomb = []
		for i, j in itertools.combinations(coefsim, 2):
			coefcomb.append(i * j) 
		coefcomb = np.asarray(coefcomb)
		coefcomb = np.hstack((coefsim, coefcomb, coefsim**2))
		ysim_new = np.dot(Xcomb, coefcomb)
	elif model_order == 'quadratic_pairwise':
		# make quadratic model
		Xcomb = []
		for i, j in itertools.combinations(Xsim.T, 2):
			Xcomb.append(i * j) 
		Xcomb = np.asarray(Xcomb).T
		Xcomb = np.hstack((Xsim, Xcomb))
		coefcomb = []
		for i, j in itertools.combinations(coefsim, 2):
			coefcomb.append(i * j) 
		coefcomb = np.asarray(coefcomb)
		coefcomb = np.hstack((coefsim, coefcomb))
		ysim_new = np.dot(Xcomb, coefcomb)
    
    # add randomly distributed cartesian points:
    x, y = np.random.uniform(- 0.5 * spatialsize, + 0.5 * spatialsize, (2, n_sample))
    
    # Add spatial correlation function:
    if (corr_amp > 0) & (corr_length > 0):
        dist = pairwise_distances(np.asarray([x,y]).T, metric='euclidean')
        # Add spatial correlation with 2-dimensional distance kernel function
        kernel = create_kernel_expsquared(dist, corr_length)
        ycorr = np.dot(kernel, ysim_new)
        # Normalize to unit variance
        fscale = ycorr.mean()/ysim_new.mean()
        ycorr = corr_amp * ycorr /fscale
        ysim_new += ycorr - ycorr.mean()

    # Add random noise as normal distribution
    ysim_new += np.random.normal(scale=noise, size = n_samples)
	#Save data as dataframe and coefficients on file
	header = np.hstack((feature_names, 'Ytarget'))
	data = np.hstack((Xsim, ysim_new.reshape(-1,1)))
	df = pd.DataFrame(data, columns = header)
	if outpath is not None:
		os.makedirs(outpath, exist_ok=True)
		df.to_csv(os.path.join(outpath, f'SyntheticData_{model_order}_{n_features}nfeatures_{noise}noise.csv'), index = False)
		df_coef = pd.DataFrame(coefsim.reshape(-1,1).T, columns = feature_names)
		df_coef.to_csv(os.path.join(outpath, f'SyntheticData_coefficients_{model_order}_{n_features}nfeatures_{noise}noise.csv'), index = False)
	return df, coefsim, feature_names
