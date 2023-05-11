"""
Toolset for generating geospatial synthetic data-sets with multiple features, noise and spatial correlations. 

The genrated models are regression models and can be either linear, quadratic or cubic.
Options for spatial correlations are defined by spatial correlation lengthscale and amplitude
and implemented by using a squared exponential kernel (currently only option).

User settings, such as output paths and synthetic data options, are set in the settings file 
(Default filename: settings_synthgen.yaml) 
Alternatively, the settings file can be specified as a command line argument with: 
'-s', or '--settings' followed by PATH-TO-FILE/FILENAME.yaml 
(e.g. python featureimportance.py -s settings_featureimportance.yaml).

This package is part of the machine learning project developed for the Agricultural Research Federation (AgReFed).
"""

import os
import itertools
import sys
import yaml
import shutil
import argparse
from types import SimpleNamespace  
from pathlib import Path
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
                spatialsize = 100, center = [140,-35],  crs = 'EPSG:4326', grid = False):
	"""
	Generate synthetic datasets

	Input:
		n_features: number of features
        n_informative_features: number of important features
		n_samples: number of samples (if grid = True then n_samples corresponds to the number of points along each axis)
        outpath: path to save simulated data	
		model_order: order of the model, either 'linear', 'quadratic', or 'cubic'
		correlated: if True, the features are correlated
		noise: random noise level [range: 0-1] that is added to the synthetic data
        corr_length: spatial correlation length [Gaussian FWHM in meters]
        corr_amp: spatial correlation amplitude
        spatialsize: size in x and y direction [in meters]
        center: [x,y] coordinates of the center of the data in meter (Easting, Northings)
        crs: coordinate reference system [Default: 'EPSG:8059']
		grid: if True, the synthetic data is generated on a regular grid, if not, it is generated on a random distribution

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
	if grid:
		n_samples = n_samples**2
	if crs == 'EPSG:4326':
		# convert form arcsec to degrees
		spatialsize = spatialsize/3600
    # Generate regression features:
	Xsim, ysim, coefsim = make_regression(n_samples=n_samples, n_features = n_features, n_informative=n_informative_features, n_targets=1, 
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
	if grid:
		x, y = np.meshgrid(np.linspace(-spatialsize/2, spatialsize/2, int(np.sqrt(n_samples))), np.linspace(-spatialsize/2, spatialsize/2, int(np.sqrt(n_samples))))
		x = x.flatten()
		y = y.flatten()
	else:
		x, y = np.random.uniform(- 0.5 * spatialsize, + 0.5 * spatialsize, (2, n_samples))

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

	# Add coordinate center to x,y
	x += center[0]
	y += center[1]

	# Add random noise as normal distribution
	ysim_new += np.random.normal(scale=noise, size = n_samples)
	#Save data as dataframe and coefficients on file
	if crs == 'EPSG:4326':
		header = np.hstack((feature_names, 'Ytarget', 'Longitude', 'Latitude'))
	else:
		header = np.hstack((feature_names, 'Ytarget', 'Easting', 'Northing'))
	data = np.hstack((Xsim, ysim_new.reshape(-1,1), x.reshape(-1,1), y.reshape(-1,1)))
	df = pd.DataFrame(data, columns = header)
	if outpath is not None:
		os.makedirs(outpath, exist_ok=True)
		if grid:
			gridname = '_grid'
		else:
			gridname = ''
		# Add datetime now to filename
		date = datetime.datetime.now().strftime("%Y-%m-%d")
		outfname = os.path.join(outpath, f'SyntheticData_{model_order}_{n_features}nfeatures_{date}{gridname}.csv')
		df.to_csv(outfname, index = False)
		# Now save coefficients and other parameters in extra file:
		df_coef = pd.DataFrame(coefsim.reshape(-1,1).T, columns = feature_names)
		# Add columns with spatial correlation function
		if (corr_amp > 0) & (corr_length > 0):
			df_coef['corr_amp'] = corr_amp
			df_coef['corr_length'] = corr_length
		# Add column with noise level
		df_coef['noise'] = noise
		outfname_coef = os.path.join(outpath, f'SyntheticData_coefficients_{model_order}_{n_features}nfeatures_{date}{gridname}.csv')
		df_coef.to_csv(outfname_coef, index = False)
	return df, coefsim, feature_names, outfname


def sample_fromgrid(fname_grid, nsample):
	"""
	Sample random points from grid of synthetic data

	Parameters
	----------
	fname_grid : str
		Name of grid file
	nsample : int
		Number of samples to be drawn from grid

	Returns
	-------
	filename of output file
	"""
	df_grid = pd.read_csv(fname_grid)
	df_grid = df_grid.sample(n = nsample, random_state = 42)
	outfname = os.path.join(Path(fname_grid).parent,f'{Path(fname_grid).stem}_{nsample}sample.csv')
	df_grid.to_csv(outfname, index = False)
	return outfname


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
	df, coefsim, feature_names, outfname = gen_synthetic(n_features = settings.n_features, 
	n_informative_features = settings.n_informative_features, 
	n_samples = settings.n_samples , outpath = settings.outpath, 
	model_order = settings.model_order, correlated = settings.correlated, 
	noise=settings.noise, corr_length = settings.corr_length, corr_amp = settings.corr_amp, 
	spatialsize = settings.spatialsize, center = settings.center,  crs = settings.crs, grid = settings.grid)

	# Draw random samples from grid of synthetic data (if grid is used)
	if settings.grid & (settings.nsample_from_grid > 0):
		outfname_samples = sample_fromgrid(outfname, settings.nsample_from_grid)
		print(f'Samples from grid saved to {outfname_samples}')

if __name__ == '__main__':
	# Parse command line arguments
	parser = argparse.ArgumentParser(description='Calculating feature importance.')
	parser.add_argument('-s', '--settings', type=str, required=False,
						help='Path and filename of settings file.',
						default = _fname_settings)
	args = parser.parse_args()

	# Run main function
	main(args.settings)