"""
Toolset for generating geospatial synthetic data-sets with multiple features, noise and spatial correlations. 

The genaretd models are regression models and can be either linear, quadratic or cubic.
Options for spatial correlations are defined by spatial correlation lengthscale and amplitude
and implemented by using a squared exponential kernel (currently only option).

User settings, such as output paths and synthetic data options, are set in the settings file 
(Default filename: settings_synthgen.yaml) 
Alternatively, the settings file can be specified as a command line argument with: 
'-s', or '--settings' followed by PATH-TO-FILE/FILENAME.yaml 
(e.g. python featureimportance.py -s settings_featureimportance.yaml).

Requirements:
- python>=3.9
- matplotlib>=3.5.1
- numpy>=1.22.0
- pandas>=1.3.5
- PyYAML>=6.0
- scikit_learn>=1.0.2
- scipy>=1.7.3


ToDo:
    - enable/disable return of dataframe with simulated features in function gen_synthetic()
    - generate geopackage including coordinates and features

Possible future add-ons:
    - add spatial correllation with different lengthscales for each dimension 
    (currently implementation had one lengthscale for spatial distance in x,y plane)
    - mix of regression and categorical features (current implementation has only regression features)
    - add options for third dimension (currently only 2D)
    - add temporal features
    - add multiple kernel functions for spatial correlations (currently only squared exponential implemented)


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
import datetime
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

# Settings yaml file
_fname_settings = 'settings_synthgen.yaml'


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
        n_informative_features: number of important features
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
    # Generate regression features:
	Xsim, ysim, coefsim = make_regression(n_samples=_n_samples, n_features = n_features, n_informative=n_informative, n_targets=1, 
		bias=0.5, noise=noise, shuffle=False, coef=True, random_state=random_state, effective_rank = n_rank)	
	# Name features:
	feature_names = ["Feature_" + str(i+1) for i in range(n_features)]

    # Scale features to the range [0,1]
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
		# Add datetime now to filename
		date = datetime.datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
		df.to_csv(os.path.join(outpath, f'SyntheticData_{model_order}_{n_features}nfeatures_{date}.csv'), index = False)
		# Now save coefficients and other parameters in extra file:
		df_coef = pd.DataFrame(coefsim.reshape(-1,1).T, columns = feature_names)
		# Add columns with spatial correlation function
		if (corr_amp > 0) & (corr_length > 0):
			df_coef['corr_amp'] = corr_amp
			df_coef['corr_length'] = corr_length
		# Add column with noise level
		df_coef['noise'] = noise
		df_coef.to_csv(os.path.join(outpath, f'SyntheticData_coefficients_{model_order}_{n_features}nfeatures_{date}.csv'), index = False)
	return df, coefsim, feature_names


def main(fname_settings):
	"""
	Main function for generating synthetic data.

	Input:
		fname_settings: path and filename to settings file
	"""
    # Load settings from yaml file
	with open(fname_settings, 'r') as f:
		settings = yaml.load(f, Loader=yaml.FullLoader)
	# Parse settings dictionary as namespace (settings are available as 
	# settings.variable_name rather than settings['variable_name'])
	settings = SimpleNamespace(**settings)

	# Verify output directory and make it if it does not exist
	os.makedirs(settings.outpath, exist_ok = True)

    # Generate synthetic data
    df, coefsim, feature_names = gen_synthetic(n_features = settings.n_features, 
                                            n_informative_features = settings.n_informative_features, 
                                            n_sample = settings.n_sample , outpath = settings.outpath, 
                                            model_order = settings.model_order, correlated = settings.correlated, 
                                            noise=settings, corr_length = settings.corr_length, corr_amp = settings.corr_amp, 
                                            spatialsize = settings.spatialsize, center = settings.center,  crs = settings.crs)


if __name__ == '__main__':
	# Parse command line arguments
	parser = argparse.ArgumentParser(description='Calculating feature importance.')
	parser.add_argument('-s', '--settings', type=str, required=False,
						help='Path and filename of settings file.',
						default = _fname_settings)
	args = parser.parse_args()

	# Run main function
	main(args.settings)