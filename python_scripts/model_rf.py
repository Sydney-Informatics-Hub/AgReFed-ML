"""
Random Forest Model with uncertainty estimates and feature importance.


This package is part of the machine learning project developed for the Agricultural Research Federation (AgReFed).

Copyright 2022 Sebastian Haan, Sydney Informatics Hub (SIH), The University of Sydney

This open-source software is released under the AGPL-3.0 License.

@author: Sebastian Haan
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr

print_info = False


def pred_ints(model, X, percentile=95):
	"""
	Predict standard deviation and CI using stats of all decision trees

	INPUT
	model: trained sklearn model
	X: input data matrix with shape (npoints,nfeatures)
	percentile: percentile of confidence interval

	RETURN
	stddev: standard deviation of prediction
	err_down: lower bound of confidence interval
	err_up: upper bound of confidence interval
	"""
	preds = []
	for pred in model.estimators_:
	    preds.append(pred.predict(X))
	preds = np.asarray(preds)
	stddev = np.std(preds, axis =0)
	err_down = np.percentile(preds, (100 - percentile) / 2., axis = 0)
	err_up = np.percentile(preds, 100 - (100 - percentile) / 2., axis = 0)
	return stddev, err_down, err_up


def rf_train(X_train, y_train):
	"""
	Trains Random Fortest regression model with trainig data

	INPUT
	X: input data matrix with shape (npoints,nfeatures)
	y: target varable with shape (npoints)

	RETURN
	rf_model: trained sklearn RF model
	"""
	rf_reg = RandomForestRegressor(n_estimators=1000, min_samples_leaf=2, max_features = 0.3, random_state = 42) # 'reg:linear' depreceasted 
	rf_reg.fit(X_train, y_train)
	return rf_reg
	


def rf_predict(X_test, rf_model, y_test = None, outpath = None):
	"""
	Returns Prediction for Random Forest regression model

	INPUT
	X_text: input datpoints in shape (ndata,n_feature). THe number of features has to be the same as for the training data
	xg_model: pre-trained XGboost regression model
	y_test: if True, uses true y data for normalized RMSE calculation

	Return
	ypred: predicted y values
	ypred_std: standard deviation of prediction
	rmse_test: RMSE of test data (if y_test is not None)
	"""
	ypred = rf_model.predict(X_test)
	ypred_std, _ , _ = pred_ints(rf_model, X_test, percentile=95)
	if y_test is not None:
		rmse_test = np.sqrt(np.mean((ypred - y_test)**2)) / y_test.std()
		if print_info: print("Random Forest normalized RMSE Test: ", np.round(rmse_test, 4))
		if outpath is not None:
			plt.figure()  # inches
			plt.title('Random Forest Test Data')
			plt.errorbar(y_test, ypred, ypred_std, linestyle='None', marker = 'o', c = 'b')
			#plt.scatter(y_test, ypred, c = 'b')
			plt.xlabel('y True')
			plt.ylabel('y Predict')
			plt.savefig(os.path.join(outpath,'RF_test_pred_vs_true.png'), dpi = 300)
			plt.close('all')
	else:
		rmse_test = None
	return ypred, ypred_std, rmse_test


def rf_train_predict(X_train, y_train, X_test, y_test = None, outpath = None):
	"""
	Trains Random Forest regression model with trainig data and returns prediction for test data

	INPUT
	X_train: input data matrix with shape (npoints,nfeatures)
	y_train: target varable with shape (npoints)
	X_test: input data matrix with shape (npoints_test,nfeatures)
	y_test: target varable with shape (npoints_test)
	outpath: path to save plots

	RETURN
	ypred: predicted y values
	residuals: residuals of prediction
	"""

	# Train BNN
	rf_model = rf_train(X_train, y_train)

	# Predict for X_test
	ypred, ypred_std, nrmse_test = rf_predict(X_test, rf_model, y_test = y_test, outpath = outpath)

	# calculate square errors
	if y_test is not None:
		residual = ypred - y_test
	else:
		residual = np.zeros_like(y_test)
	return ypred, residual


def test_rf(logspace = False, nsamples = 600, nfeatures = 14, ninformative = 12, noise = 0.2, outpath = None):
	"""
	Test RF model on synthetic data

	INPUT
	logspace: if True, uses logarithmic scale for features
	nsamples: number of samples
	nfeatures: number of features
	ninformative: number of informative features
	noise: noise level
	outpath: path to save plots

	"""
	# Create simulated data
	from sklearn.datasets import make_regression
	Xorig, yorig, coeffs = make_regression(n_samples=nsamples, 
		n_features=nfeatures, n_informative=ninformative, 
		n_targets=1, bias=2.0, tail_strength=0.2, noise=noise, shuffle=True, coef=True, random_state=42)
	if logspace:
		Xorig = np.exp(Xorig)
		yorig = np.exp(yorig/100)

	if outpath is not None:	
		os.makedirs(outpath, exist_ok = True)

	X_train, X_test, y_train, y_test = train_test_split(Xorig, yorig, test_size=0.5, random_state=42)

	# Run BNN
	y_pred, residual = rf_train_predict(X_train, y_train, X_test, y_test = y_test, outpath = outpath)

	# Calculate normalized RMSE:
	nrmse = np.sqrt(np.nanmean(residual**2)) / y_test.std()
	nrmedse = np.sqrt(np.median(residual**2)) / y_test.std()
	if print_info:
		print('Normalized RMSE for test data: ', np.round(nrmse,3))
		print('Normalized ROOT MEDIAM SE for test data: ', np.round(nrmedse,3))
	#Feature Importance
	feature_importance = rf_factor_importance(X_train, y_train)
	if print_info:
		print('Model correlation coefficients:', coeffs )
		print('Feature Importance:', feature_importance)
