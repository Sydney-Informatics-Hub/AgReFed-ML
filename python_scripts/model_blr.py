"""
Bayesian Linear Regression with uncertainty estimates and feature importance.

This package is part of the machine learning project developed for the Agricultural Research Federation (AgReFed).
"""
import warnings
warnings.filterwarnings('ignore') 
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

import matplotlib.pyplot as plt
import numpy as np
import os
import random
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import BayesianRidge

print_info = False


def scale_data(X, y, scaler = 'power'):
	"""
	Scaling data with power scaler and multiplication of y
	see also as introduction to different scalers:
	https://www.analyticsvidhya.com/blog/2020/07/types-of-feature-transformation-and-scaling/
	https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html

	INPUT
	-----
	X: input data matrix with shape (npoints,nfeatures)
	y: target variable with shape (npoints)
	scaler: 'power' , 'standard', or 'robust'. Default is Powertransform

	Return:
	-------
	Xs: scaled X
	ys: scaled y
	fitparams: fit parameters of scaler
	"""
	if scaler == 'power':
		scaler_x = PowerTransformer(copy=True, method='yeo-johnson', standardize=True)
	elif scaler == 'standard':
		scaler_x = StandardScaler()
	elif scaler == 'robust':
		scaler_x = RobustScaler()
	#scaler_x = MinMaxScaler(feature_range=(0.01,1.01)) 
	scaler_x = scaler_x.fit(X)
	Xs = scaler_x.transform(X)

	# Need all data larger than 0 for logspace conversion
	if scaler == 'power':
		scaler_y = PowerTransformer(copy=True, method='yeo-johnson', standardize=True)
	elif scaler == 'standard':
		scaler_y = StandardScaler()
	elif scaler == 'robust':
		scaler_y = RobustScaler()
	scaler_y = scaler_y.fit(y.reshape(-1, 1)) # use .ravel() instead?
	ys = scaler_y.transform(y.reshape(-1, 1)).flatten() # use .ravel() instead?

	return Xs, ys, (scaler_x, scaler_y)


def invscale_data(X, y, scale_param):
	"""
	Inverse Scaling data with standard scaler and multiplication of y

	INPUT
	-----
	X: input data matrix with shape (npoints,nfeatures)
	y: target variable with shape (npoints)
	scale_param: (scaler_x, scaler_y)

	Return:
	-------
	Xinv: inverse scaled X
	yinv: inverse scaled y
	"""
	scaler_x, scaler_y = scale_param

	Xinv = scaler_x.inverse_transform(X)
	yinv =  scaler_y.inverse_transform(y.reshape(-1, 1))
	return Xinv, yinv.flatten()


def blr_train(X_train, y_train, logspace = False):
	"""
	Trains Bayesian Linear Regresssion model 

	INPUT
	-----
	X: input data matrix with shape (npoints,nfeatures)
	y: target varable with shape (npoints)
	logspace: if True, models regression in logspace

	Return
	------
	regression model
	"""
	if logspace:
		x = np.log(X_train)
		y = np.log(y_train)
	else:
		x = X_train
		y = y_train
	#sel = np.where(np.isfinite(x) & np.isfinite(y))
	if x.shape[1] == 1:
		x = x.reshape(-1,1)
	y = y.reshape(-1,1)
	reg = BayesianRidge(tol=1e-6, fit_intercept=True, compute_score=True)
	reg.fit(x, y)

	if print_info:
		print('BLR regresssion coefficients:')
		print(reg.coef_)
		print('BLR regresssion coefficients uncertainity:')
		print(np.diag(reg.sigma_))
		print('BLR regresssion intercept:')
		print(reg.intercept_)
	# Set not significant coeffcients to zero
	coef = reg.coef_.copy()
	coef_sigma = np.diag(reg.sigma_).copy()
	sel = np.where(abs(reg.coef_) > 3 * coef_sigma)
	if print_info:
		for i in sel[0]:
			print('X' + str(i), ' wcorr=' + str(np.round(coef[i], 3)) + ' +/- ' + str(np.round(coef_sigma[i], 3)))
		print("Number of insignificant features: ", x.shape[1] - len(sel[0]))
	xnew = x[:, sel[0]]
	regnew = BayesianRidge(tol=1e-6, fit_intercept=True, compute_score=True)
	regnew.fit(xnew, y)
	coefnew = regnew.coef_.copy() 
	coef *= 0
	coef[sel] = coefnew 
	reg.coef_ = coef

	return reg
	


def blr_predict(X_test, blr_reg, y_test = None, outpath = None, logspace = False):
	"""
	Returns Prediction for BL regression model

	INPUT
	-----
	X_text: input datpoints in shape (ndata,n_feature). The number of features has to be the same as for the training data
	xg_model: pre-trained XGboost regression model
	y_test: if not None, uses true y data for normalized RMSE calculation
	outpath: if not None, saves prediction to file
	logspace: if True, models regression in logspace

	Return
	------
	ypred: predicted y values
	ypred_std: predicted y values standard deviation
	rmse: RMSE
	"""
	if logspace:
		x = np.log(X_test)
	else:
		x = X_test
	if x.shape[1] == 1:
		x = x.reshape(-1,1)
	y, ystd = blr_reg.predict(x, return_std=True)
	if logspace:
		ypred  = np.exp(y).flatten()
	else:
		ypred = y.flatten()

	if y_test is not None:
		rmse_test = np.sqrt(np.mean((ypred - y_test)**2)) / y_test.std()
		if print_info: print("BLR normalized RMSE Test: ", np.round(rmse_test, 4))

		if outpath is not None:
			plt.figure()  # inches
			plt.title('BLR Test Data')
			plt.scatter(y_test, ypred, c = 'b')
			plt.xlabel('y True')
			plt.ylabel('y Predict')
			plt.savefig(os.path.join(outpath,'BLR_test_pred_vs_true.png'), dpi = 300)
			plt.close('all')
	else:
		rmse_test = None

	return ypred, ystd, rmse_test


def blr_train_predict(X_train, y_train, X_test, y_test = None, outpath = None, logspace=False):
	"""
	Trains and predicts BLR model

	INPUT:
	-----
	X_train: input data matrix with shape (npoints,nfeatures)
	y_train: target varable with shape (npoints)
	X_test: input data matrix with shape (npoints,nfeatures)
	y_test: target varable with shape (npoints)
	outpath: path to save plots
	logspace: if True, models regression in logspace

	Return:
	------
	ypred: predicted y values
	ypred_std: predicted y values standard deviation
	rmse: RMSE
	"""
	Xs_train, ys_train, scale_param = scale_data(X_train, y_train)
	scaler_x, scaler_y = scale_param
	Xs_test = scaler_x.transform(X_test)
	if y_test is not None:
		ys_test = scaler_y.transform(y_test.reshape(-1, 1)) # use .ravel() instead?
		ys_test = ys_test.flatten()
	else:
		ys_test = None
	# Train BLR
	Xs_test = X_test
	Xs_train = X_train
	ys_train= y_train
	ys_test = y_test
	blr_model = blr_train(Xs_train, ys_train, logspace = logspace)

	# Predict for X_test
	ypred, nrmse_test = blr_predict(Xs_test, blr_model, y_test = ys_test, outpath = outpath, logspace = logspace)

	# Rescale data to original scale
	y_pred =  scaler_y.inverse_transform(ypred.reshape(-1, 1)) # use .ravel() instead?
	y_pred = y_pred.flatten()
	y_pred = ypred

	# calculate square errors
	if y_test is not None:
		residual = y_pred - y_test
	else:
		residual = np.zeros_like(y_test)

	return y_pred, residual


def test_blr(logspace = True, nsamples = 600, nfeatures = 14, ninformative = 12, noise = 0.1, outpath = None):
	"""
	Test BLR model on synthetic data

	INPUT:
	-----
	logspace: if True, models regression in logspace
	nsamples: number of samples for training
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
	
	Xs, ys, params = scale_data(Xorig, yorig) 

	if outpath is not None:	
		os.makedirs(outpath, exist_ok = True)

	X_train, X_test, y_train, y_test = train_test_split(Xs, ys, test_size=0.5, random_state=42)

	# Run BNN
	y_pred, residual = blr_train_predict(X_train, y_train, X_test, y_test = y_test, outpath = outpath, logspace = logspace)

	# Calculate normalized RMSE:
	nrmse = np.sqrt(np.nanmean(residual**2)) / y_test.std()
	nrmedse = np.sqrt(np.median(residual**2)) / y_test.std()
	if print_info:
		print('Normalized RMSE for test data: ', np.round(nrmse,3))
		print('Normalized ROOT MEDIAM SE for test data: ', np.round(nrmedse,3))
