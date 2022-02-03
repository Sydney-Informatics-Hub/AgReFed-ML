"""
Toolset for calculating feature importance based on multiple methods:

- Hierarchical clustered Spearman correlation diagram
- linear/log-scaled Bayesian Linear Regression
- Random Forest Permutation Importance
- Model-agnostic correlation coefficients 
(see "A new coefficient of correlation" Chatterjee, S. (2019, September 22)

This script can also generate synthetic data and includes tests for all methods,
which can be used to compare the results.

User settings, such as input/output paths and all other options, are set in the settings file 
(Default filename: settings_featureimportance.yaml) 
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
- xicor>=1.0.1

For more package details see conda environment file: environment.yaml

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
import matplotlib as mpl
import matplotlib.pyplot as plt
from xicor.xicor import Xi


# Settings yaml file
_fname_settings = 'settings_featureimportance.yaml'


def plot_feature_correlation_spearman(X, feature_names, outpath, show = False):
	"""
	Plot feature correlations using Spearman correlation coefficients.
	Feature correlations are automatically clustered using hierarchical clustering.

	Result figure is automatically saved in specified path.

	Input:
		X: data array
		feature names: list of feature names
		outpath: path to save plot
		show: if True, interactive matplotlib plot is shown
	"""
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
	corr = spearmanr(X).correlation
	corr_linkage = hierarchy.ward(corr)
	dendro = hierarchy.dendrogram(corr_linkage, labels=feature_names, ax=ax1, leaf_rotation=90)
	dendro_idx = np.arange(0, len(dendro['ivl']))

	# Plot results:
	pos = ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']])
	ax2.set_xticks(dendro_idx)
	ax2.set_yticks(dendro_idx)
	ax2.set_xticklabels(dendro['ivl'], rotation='vertical')
	ax2.set_yticklabels(dendro['ivl'])
	fig.colorbar(pos, ax = ax2)
	fig.tight_layout()
	plt.savefig(os.path.join(outpath, 'Feature_Correlations_Hierarchical_Spearman.png'), dpi = 300)
	if show:
		plt.show()



def calc_new_correlation(X, y):
	"""
	Calculation of general correlation coefficient 
	as described in Chatterjee, S. (2019, September 22):
	A new coefficient of correlation. arxiv.org/abs/1909.10140
	Returns correlation coefficient for testing independence

	In comparison to Spearman, Pearson, and Kendalls, this new correlation coefficient
	can measure associations that are not monotonic or non-linear, 
	and is 0 if X and Y are independent, and is 1 if there is a function Y= f(X).
	Note that this coefficient is intentionally not summetric, because we want to understand 
	if Y is a function X, and not just if one of the variables is a function of the other.
	
	This function also removes Nans and converts factor variables to integers automatically.

	Input:
		X: data feature array with shape (npoints, n_features)
		y: target variable as vector with size npoints

	Return:
		corr: correlation coefficients
	"""
	n_features = X.shape[1] 
	corr = np.empty(n_features)
	pvals = np.empty(n_features)
	for i in range(n_features):
		xi_obj = Xi(X[:,i], y)
		corr[i] = xi_obj.correlation
		pvals[i] = xi_obj.pval_asymptotic(ties=False, nperm=1000)
	# set correlation coefficient to zero for non-significant p_values (P > 0.01)
	corr[pvals>0.01] = 0
	return corr


def test_calc_new_correlation():
	"""
	Test function for calc_new_correlation
	"""
	dfsim, coefsim, feature_names = create_simulated_features(6, model_order = 'quadratic', noise = 0.01)
	X = dfsim[feature_names].values
	y = dfsim['Ytarget'].values
	corr = calc_new_correlation(X, y)
	assert np.argmax(coefsim) == np.argmax(corr)


def blr_factor_importance(X, y, logspace = False, signif_threshold = 2):
	"""
	Trains Bayesian Linear Regresssion model and returns the estimated significance of regression coefficients.
	The significance of the linear coefficient is defined by dividing the estimated coefficient 
	over the standard deviation of this estimate. The correlation significance is set to zero if below threshold.

	Input:
		X: input data matrix with shape (npoints,nfeatures)
		y: target varable with shape (npoints)
		logspace: if True, models regression in logspace
		signif_threshold: threshold for coefficient significance to be considered significant (Default = 2)
		

	Return:
		coef_signif: Significance of coefficients (Correlation coefficient / Stddev)
	"""
	# Scale data using RobustScaler:
	if X.shape[1] == 1:
		X = x.reshape(-1,1)
	scaler = RobustScaler(unit_variance = True)
	X = scaler.fit_transform(X)
	y = scaler.fit_transform(y.reshape(-1,1)).ravel()
	if logspace:
		X = np.log(X - X.min(axis = 0) + 1)
		y = np.log(y - y.min() + 1)	
	#sel = np.where(np.isfinite(x) & np.isfinite(y))

	y = y.reshape(-1,1).ravel()

	# Apply Bayesian Linear Regression:
	reg = BayesianRidge(tol=1e-6, fit_intercept=True, compute_score=True)
	reg.fit(X, y)

	#print('BLR regresssion coefficients:')
	# Set none significant coeffcients to zero
	coef = reg.coef_.copy()
	coef_sigma = np.diag(reg.sigma_).copy()
	coef_signif = coef / coef_sigma
	#for i in range(len(coef)):
	#	print('X' + str(i), ' wcorr=' + str(np.round(coef[i], 3)) + ' +/- ' + str(np.round(coef_sigma[i], 3)))
	# Set not significant coefficients to zero:
	coef_signif[coef_signif < signif_threshold] = 0
	return coef_signif


def test_blr_factor_importance():
	"""
	Test function for blr_factor_importance
	"""
	dfsim, coefsim, feature_names = create_simulated_features(6, model_order = 'linear', noise = 0.05)
	X = dfsim[feature_names].values
	y = dfsim['Ytarget'].values
	coef_signif = blr_factor_importance(X, y)
	assert np.argmax(coefsim) == np.argmax(coef_signif)


def rf_factor_importance(X_train, y_train, correlated = False):
	"""
	Factor importance using RF permutation test and optional corrections 
	for multi-collinarity (correlated) features. 
	Including training of Random Forest regression model with training data 
	and setting non-significant coefficients to zero.

	Input:
		X: input data matrix with shape (npoints,nfeatures)
		y: target varable with shape (npoints)
		correlated: if True, features are assumed to be correlated

	Return:
		imp_mean_corr: feature importances
	"""
	rf_reg = RandomForestRegressor(n_estimators=500, min_samples_leaf=4, random_state = 42)
	rf_reg.fit(X_train, y_train)
	result = permutation_importance(rf_reg, X_train, y_train, n_repeats=20, random_state=42, 
		n_jobs=1, scoring = "neg_mean_squared_error")
	imp_mean = result.importances_mean
	imp_std = result.importances_std
	# Make corrections for correlated features
	# This is necessary since permutation importance are lower for correlated features
	if correlated:
		corr = spearmanr(X_train).correlation
		imp_mean_corr = np.zeros(len(imp_mean))
		imp_std_corr = np.zeros(len(imp_mean))
		for i in range(len(imp_mean)):
			imp_mean_corr[i] = np.sum(abs(corr[i]) * imp_mean)
			imp_std_corr[i] = np.sqrt(np.sum(abs(corr[i]) * imp_std**2))
	else:
		imp_mean_corr = imp_mean
		imp_std_corr = imp_std
	#print("Random Forest factor importances: ", imp_mean_corr)
	#print("Random Forest factor importances std: ", imp_std_corr)
	# Set non significant features to zero:
	imp_mean_corr[imp_mean_corr / imp_std_corr < 3] = 0
	imp_mean_corr[imp_mean_corr < 0.001] = 0
	return imp_mean_corr


def test_rf_factor_importance():
	"""
	Test function for rf_factor_importance
	"""
	dfsim, coefsim, feature_names = create_simulated_features(6, model_order = 'quadratic', noise = 0.05)
	X = dfsim[feature_names].values
	y = dfsim['Ytarget'].values
	imp_mean_corr = rf_factor_importance(X, y)
	assert np.argmax(coefsim) == np.argmax(imp_mean_corr)



def create_simulated_features(n_features, outpath = None, n_samples = 200, model_order = 'quadratic', correlated = False, noise= 0.1):
	"""
	Generate synthetic datasets for testing

	Input:
		n_features: number of features
		outpath: path to save simulated data
		n_samples: number of samples	
		model_order: order of the model, either 'linear', 'quadratic', or 'cubic'
		correlated: if True, the features are correlated
		noise: noise level [range: 0-1]

	Return:
		dfsim: dataframe with simulated features
		coefsim: simulated coefficients
		feature_names: list of feature names
	"""
	if correlated:
		n_rank = int(n_features/2)
	else:
		n_rank = None
	Xsim, ysim, coefsim = make_regression(n_samples=200, n_features = n_features, n_informative=int(n_features/2), n_targets=1, 
		bias=0.5, noise=noise, shuffle=False, coef=True, random_state=42, effective_rank = n_rank)	
	feature_names = ["Feature_" + str(i+1) for i in range(n_features)]
	coefsim /= 100
	scaler = MinMaxScaler()
	scaler.fit(Xsim)
	Xsim = scaler.transform(Xsim)
	"""
	if outpath is not None:
		plot_feature_correlation_spearman(Xsim, feature_names, outpath)
		# Plot all correlations
		sorted_idx = coefsim.argsort()
		fig, ax = plt.subplots(figsize = (6,5))
		ypos = np.arange(len(coefsim))
		bar = ax.barh(ypos, coefsim[sorted_idx], tick_label = np.asarray(feature_names)[sorted_idx], align='center')
		gradientbars(bar, coefsim[sorted_idx])
		plt.xlabel("True Feature Coefficients")
		plt.tight_layout()
		plt.savefig(os.path.join(outpath, 'Feature_True_Coef.png'), dpi = 300)
		plt.close('all')
	"""
	#plot_feature_correlation(Xsim, feature_names)
	# make first-order model
	if model_order == 'linear':
		ysim_new = np.dot(Xsim, coefsim) + np.random.normal(scale=noise, size = n_samples)
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
		ysim_new = np.dot(Xcomb, coefcomb) + np.random.normal(scale=noise, size = n_samples)
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
		ysim_new = np.dot(Xcomb, coefcomb) + np.random.normal(scale=noise, size = n_samples)
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


def gradientbars(bars, data):
	"""
	Helper function for making colorfull bars

	Input:
		bars: list of bars
		data: data to be plotted
	"""
	ax = bars[0].axes
	lim = ax.get_xlim()+ax.get_ylim()
	ax.axis(lim)
	for bar in bars:
		bar.set_zorder(1)
		bar.set_facecolor("none")
		x,y = bar.get_xy()
		w, h = bar.get_width(), bar.get_height()
		grad = np.atleast_2d(np.linspace(0,1*w/max(data),256))
		ax.imshow(grad, extent=[x,x+w,y,y+h], aspect="auto", zorder=0, norm=mpl.colors.NoNorm(vmin=0,vmax=1))
	ax.axis(lim)


def plot_correlationbar(corrcoefs, feature_names, outpath, fname_out, name_method = None, show = False):
	"""
	Helper function for plotting feature correlation.
	Result plot is saved in specified directory.

	Input:
		corrcoefs: list of feature correlations
		feature_names: list of feature names
		outpath: path to save plot
		fname_out: name of output file (should end with .png)
		name_method: name of method used to compute correlations
		show: if True, show plot
	"""
	sorted_idx = corrcoefs.argsort()
	fig, ax = plt.subplots(figsize = (6,5))
	ypos = np.arange(len(corrcoefs))
	bar = ax.barh(ypos, corrcoefs[sorted_idx], tick_label = np.asarray(feature_names)[sorted_idx], align='center')
	gradientbars(bar, corrcoefs[sorted_idx])
	if name_method is not None:	
		plt.title(f'{name_method}')	
	plt.xlabel("Feature Importance")
	plt.tight_layout()
	plt.savefig(os.path.join(outpath, fname_out), dpi = 200)
	if show:
		plt.show()
	plt.close('all')


def test_plot_correlationbar(outpath):
	"""
	Test function for plot_correlationbar
	"""
	dfsim, coefsim, feature_names = create_simulated_features(6, model_order = 'quadratic', noise = 0.05)
	plot_correlationbar(coefsim, feature_names, outpath, 'test_plot_correlationbar.png', show = True)


def main(fname_settings):
	"""
	Main function for running the script.

	Input:
		fname_settings: path and filename to settings file
	"""
	# Load settings from yaml file
	with open(fname_settings, 'r') as f:
		settings = yaml.load(f, Loader=yaml.FullLoader)
	# Parse settings dictinary as namespace (settings are available as 
	# settings.variable_name rather than settings['variable_name'])
	settings = SimpleNamespace(**settings)

	# Verify output directory and make it if it does not exist
	os.makedirs(settings.outpath, exist_ok = True)

	# Read data
	data_fieldnames = settings.name_features + [settings.name_target]
	df = pd.read_csv(os.path.join(settings.inpath, settings.infname), usecols=data_fieldnames)

	# Verify that data is cleaned:
	assert df.select_dtypes(include=['number']).columns.tolist().sort() == data_fieldnames.sort(), 'Data contains non-numeric entries.'
	assert df.isnull().sum().sum() == 0, "Data is not cleaned, please run preprocess_data.py before"


	# 1) Generate Spearman correlation matrix
	print("Calculate Spearman correlation matrix...")
	plot_feature_correlation_spearman(df[data_fieldnames].values, data_fieldnames, settings.outpath, show = False)

	# 2) Generate feature importance based on model-agnostic correlation 
	print("Calculate feature importance for mode-agnostic correaltions...")
	X = df[settings.name_features].values
	y = df[settings.name_target].values
	corr = calc_new_correlation(X, y)
	plot_correlationbar(corr, settings.name_features, settings.outpath, 'Model-agnostic-correlation.png', name_method = 'Model-agnostic', show = False)

	# 3) Generate feature importance based on significance of Bayesian Linear Regression coeffcicients:
	print("Calculate feature importance for Bayesian Linear Regression...")
	corr = blr_factor_importance(X, y, logspace = False)
	plot_correlationbar(corr, settings.name_features, settings.outpath, 'BLR-linear-correlation.png', name_method = 'BLR linear correlation significance', show = False)
	# and in log-space
	corr = blr_factor_importance(X, y, logspace = True)
	plot_correlationbar(corr, settings.name_features, settings.outpath, 'BLR-log-correlation.png', name_method = 'BLR log-correlation significance', show = False)

	# 4) Generate feature importance based on Random Forest permutation importance
	print("Calculate feature importance for Random Forest permutation importance...")
	corr = rf_factor_importance(X, y)
	plot_correlationbar(corr, settings.name_features, settings.outpath, 'RF-permutation-importance.png', name_method = 'RF permutation importance', show = False)




def test_main():
	"""
	Test function for main function.

	This test automatically generates synthetic data and generates feature importance plots
	in the subfolder `test_featureimportance`.
	"""
	# Make temporary result folder
	outpath = 'test_featureimportance'
	os.makedirs(outpath, exist_ok = True)

	# Generate simulated data
	print("Generate simulated data...")
	dfsim, _, feature_names_sim = create_simulated_features(8, outpath = outpath)

	# Generate settings file for simulated data
	# (Note: you could also just simply set settings variables here, but this is also testing the settings file readout)
	fname_settings_sim = 'settings_featureimportance_simulation.yaml'
	shutil.copyfile(_fname_settings, os.path.join(outpath, fname_settings_sim))
	# Change yaml file to simulation specifications:
	with open(os.path.join(outpath, fname_settings_sim), 'r') as f:
		settings_sim = yaml.load(f, Loader=yaml.FullLoader)
	settings_sim['name_features'] = feature_names_sim
	settings_sim['name_target'] = 'Ytarget'
	settings_sim['infname'] = 'SyntheticData_quadratic_8nfeatures_0.1noise.csv'
	settings_sim['inpath'] = outpath
	settings_sim['outpath'] = outpath
	with open(os.path.join(outpath, fname_settings_sim), 'w') as f:
		yaml.dump(settings_sim, f)

	# Run main function
	main(os.path.join(outpath, fname_settings_sim))

	# Check that plots were generated
	assert os.path.isfile(os.path.join(outpath, 'Feature_Correlations_Hierarchical_Spearman.png')), 'Plot for Spearman correlation not generated'
	assert os.path.isfile(os.path.join(outpath, 'Model-agnostic-correlation.png')), 'Plot for model-agnostic correlation not generated'
	assert os.path.isfile(os.path.join(outpath, 'BLR-linear-correlation.png')), 'Plot for BLR linear correlation not generated'
	assert os.path.isfile(os.path.join(outpath, 'BLR-log-correlation.png')), 'Plot for BLR log-correlation not generated'
	assert os.path.isfile(os.path.join(outpath, 'RF-permutation-importance.png')), 'Plot for RF permutation importance not generated'

	# Remove temporary result folder
	# shutil.rmtree(outpath)

if __name__ == '__main__':
	# Parse command line arguments
	parser = argparse.ArgumentParser(description='Calculating feature importance.')
	parser.add_argument('-s', '--settings', type=str, required=False,
						help='Path and filename of settings file.',
						default = _fname_settings)
	args = parser.parse_args()

	# Run main function
	main(args.settings)