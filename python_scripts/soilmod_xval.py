"""
Machine Learning models and 3D Cube Soil Generator using Gaussian Process Priors. 
Please see for model details and theoretical background the documentation in docs/description_paper/


See Documentation

Copyright 2022 The University of Sydney

Version: 0.1

@author: Sebastian Haan

To Do: 
- split function in two, one for the xval and one for the prediction
- add to GP input noise not just predicted noise, but sum of predicted variance plus train rariance
- test gradient descent hyperparamter optimization insated of SHGO

"""

import numpy as np
import pandas as pd
import geopandas as gpd
import os
import sys
#from scipy.linalg import pinv, solve, cholesky, solve_triangular
#from scipy.optimize import minimize, shgo
from scipy.special import erf
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
#import pyvista as pv # helper module for the Visualization Toolkit (VTK)
import subprocess
from sklearn.model_selection import train_test_split 
# Save and load trained models and scalers:
import pickle
import json
import yaml
from types import SimpleNamespace  
from tqdm import tqdm

# Custom local libraries:
from utils import find_zone, array2geotiff, align_nearest_neighbor, print2, truncate_data
from sigmastats import averagestats
from preprocessing import gen_kfold
import GPmodel as gp # GP model plus kernel functions and distance matrix calculation

# Settings yaml file
_fname_settings = 'settings_soilmod_xval.yaml'

# Load settings from yaml file
with open(_fname_settings, 'r') as f:
    settings = yaml.load(f, Loader=yaml.FullLoader)
# Parse settings dictinary as namespace (settings are available as 
# settings.variable_name rather than settings['variable_name'])
settings = SimpleNamespace(**settings)

show = False

if type(settings.mean_functions) != list:
    settings.mean_functions = [settings.mean_functions]

for mean_function in settings.mean_functions:
    if mean_function == 'blr':
        import model_blr as blr
    if mean_function == 'rf':
        import model_rf as rf


######################### Main Script ############################

# Intialise output info file:
os.makedirs(settings.outpath, exist_ok = True)
print2('init')
print2(f'--- Parameter Settings ---')
print2(f'Mean Functions: {settings.mean_functions}')
print2(f'Target Name: {settings.name_target}')
print2(f'--------------------------')


if __name__ == '__main__':
    """
    Main script for running 3D cubing with Gaussian Process.
    See Documentation and comments below for more details.
    """

    # Start with reading in data
    # check if outpath exists, if not create direcory
    #os.makedirs(outpath, exist_ok = True)
    outpath_root = settings.outpath
    # Pre-process data
    print('Reading and pre-processing data...')
    # Read in data
    dfsel = pd.read_csv(os.path.join(settings.inpath, settings.infname))
    dfsel = gen_kfold(dfsel, nfold = 10, label_nfold = 'nfold', id_unique = ['x','y'], precision_unique = 0.01)

    # Select data between zmin and zmax
    dfsel = dfsel[(dfsel['z'] >= settings.zmin) & (dfsel['z'] <= settings.zmax)]


    ## Get coordinates for training data and set coord origin to (0,0)
    bound_xmin = dfsel.x.min()
    bound_xmax = dfsel.x.max()
    bound_ymin = dfsel.y.min()
    bound_ymax = dfsel.y.max()

    # get train and test data, here we can include loop over ix for cross validation
    range_nfold = np.sort(dfsel.nfold.unique())

    nrmse_meanfunction = []
    nrmse_meanfunction_std = []
    theta_meanfunction = []
    theta_meanfunction_std = []

    for mean_function in settings.mean_functions:
        # Loop over all mean function models
        print(f'Computing {len(range_nfold)}-fold xrossvalidation for mean function model: {mean_function}')
        subdir = 'Xval_' + str(len(range_nfold)) + '-fold_' + mean_function + '_' + settings.name_target
        outpath = os.path.join(outpath_root, subdir)

        ### X-fold Crossvalidation ###
        # Intialise lists to hold summary results for n-fold validation
        rmse_nfold = []
        nrmse_nfold = []
        rmedse_nfold = []
        nrmedse_nfold = []
        meansspe_nfold = []
        rmse_fmean_nfold = []
        nrmse_fmean_nfold = []
        rmedse_fmean_nfold = []
        nrmedse_fmean_nfold = []
        histresidual = []
        histsspe = []

        for ix in range_nfold:
            # Loop over all train/test sets (test sets are designated by ix; training set is defined by the remaining set)
            print('Processing for nfold ', ix)
            # update outpath with iteration of cross-validation
            outpath_nfold = os.path.join(outpath, 'nfold_' + str(ix) + '/')
            os.makedirs(outpath_nfold, exist_ok = True)
            # Normalize all data

            # split into train and test data
            dftrain = dfsel[dfsel[settings.name_ixval] != ix].copy()
            dftest = dfsel[dfsel[settings.name_ixval]  == ix].copy()

            # Copy dataframe for saving results later
            dfpred = dftest.copy() 

            points3D_train = np.asarray([dftrain.z.values, dftrain.y.values - bound_ymin, dftrain.x.values - bound_xmin ]).T
            points3D_test = np.asarray([dftest.z.values, dftest.y.values - bound_ymin, dftest.x.values - bound_xmin ]).T
            # Check for nan values:

            y_train = dftrain[settings.name_target].values
            y_test = dftest[settings.name_target].values
            # Uncertainty in coordinates
            Xdelta_train = np.asarray([0.5 * dftrain.z_diff.values, dftrain.y.values * 0, dftrain.x.values * 0.]).T
            Xdelta_test = np.asarray([0.5 * dftest.z_diff.values, dftest.y.values * 0, dftest.x.values * 0.]).T


            if mean_function == 'rf':
                # Estimate GP mean function with Random Forest Regressor
                X_train = dftrain[settings.name_features].values
                y_train = dftrain[settings.name_target].values
                X_test = dftest[settings.name_features].values
                y_test = dftest[settings.name_target].values
                rf_model = rf.rf_train(X_train, y_train)
                ypred_rf_train, ynoise_train, nrmse_rf_train = rf.rf_predict(X_train, rf_model, y_test = y_train)
                ypred_rf, ynoise_pred, nrmse_rf_test = rf.rf_predict(X_test, rf_model, y_test = y_test)
                y_train_fmean = ypred_rf_train
                plt.figure()  # inches
                plt.title('Random Forest Model')
                plt.errorbar(y_train, ypred_rf_train, ynoise_train, linestyle='None', marker = 'o', c = 'r', label = 'Train Data')
                plt.errorbar(y_test, ypred_rf, ynoise_pred, linestyle='None', marker = 'o', c = 'b', label = 'Test Data')
                plt.legend(loc = 'upper left')
                plt.xlabel('y True')
                plt.ylabel('y Predict')
                plt.savefig(os.path.join(outpath_nfold, settings.name_target + '_RF_pred_vs_true.png'), dpi = 300)
                if show:
                    plt.show()
                plt.close('all')
            elif mean_function == 'blr':
                X_train = dftrain[settings.name_features].values
                y_train = dftrain[settings.name_target].values
                X_test = dftest[settings.name_features].values
                y_test = dftest[settings.name_target].values
                # Scale data
                #Xs_train, ys_train, scale_params = blr.scale_data(X_train, y_train)
                #scaler_x, scaler_y = scale_params
                #Xs_test = scaler_x.transform(X_test)
                # Train BLR
                blr_model = blr.blr_train(X_train, y_train)
                # Predict for X_test
                ypred_blr, ypred_std_blr, nrmse_blr_test = blr.blr_predict(X_test, blr_model, y_test = y_test)
                ypred_blr_train,  ypred_std_blr_train, nrmse_blr_train = blr.blr_predict(X_train, blr_model, y_test = y_train)
                # Rescale data to original scale
                #ypred_blr[ypred_blr < y_train.min()] = ys_train.min()
                #ypred_blr[ypred_blr > y_train.max()] = ys_train.max()
                #ypred_blr =  scaler_y.inverse_transform(ypred_blr.reshape(-1, 1))
                ypred_blr = ypred_blr.flatten()
                # First need check for rescalein that not lower than original
                #ypred_blr_train[ypred_blr_train < ys_train.min()] = ys_train.min()
                #ypred_blr_train[ypred_blr_train > ys_train.max()] = ys_train.max()
                #ypred_blr_train =  scaler_y.inverse_transform(ypred_blr_train.reshape(-1, 1))
                ypred_blr_train = ypred_blr_train.flatten()
                y_train_fmean = ypred_blr_train
                # to invrese scale noise we need to multiply with the factor stddev(data_original) /stddev(data_transformed)
                #fac_noise = np.nanstd(y_train) / np.nanstd(ys_train)
                #fac_noise_train = abs((y_train - y_train.mean())) / abs((ys_train - ys_train.mean()))
                #fac_noise_pred = abs((y_test - y_test.mean())) / abs((ys_test - ys_test.mean()))
                ynoise_train = ypred_std_blr_train #* fac_noise_train 
                ynoise_pred = ypred_std_blr #* fac_noise_pred
                plt.figure()  # inches
                plt.title('BLR Model')
                plt.scatter(y_train, ypred_blr_train, c = 'r', label='Train Data')
                plt.scatter(y_test, ypred_blr, c = 'b', label = 'Test Data')
                plt.legend(loc = 'upper left')
                plt.xlabel('y True')
                plt.ylabel('y Predict')
                plt.savefig(os.path.join(outpath_nfold, settings.name_target + '_BLR_pred_vs_true.png'), dpi = 300)
                if show:
                    plt.show()
                plt.close('all')


            # Subtract mean function of depth from training data 
            y_train -= y_train_fmean

            # plot training and testing distribution
            plt.figure(figsize=(8,6))
            #plt.imshow(ystd.reshape(len(yspace),len(xspace)) * np.nan,origin='lower', aspect = 'equal', extent = extent)
            #plt.scatter(dfsel.Easting.values - bound_xmin, dfsel.Northing.values - bound_ymin, c = dfsel.ESP.values, edgecolors = 'k')
            plt.scatter(dftrain.x.values - bound_xmin, dftrain.y.values - bound_ymin, alpha=0.3, c = 'b', label = 'Train') 
            plt.scatter(dftest.x.values - bound_xmin, dftest.y.values - bound_ymin, alpha=0.3, c = 'r', label = 'Test')  
            plt.axis('equal')
            plt.xlabel('Easting')                                                                                                                                                                  
            plt.ylabel('Northing')                                                                                                                                                                 
            #plt.colorbar()                                                                                                                                                                         
            plt.title('Mean subtracted ' + settings.name_target) 
            #plt.colorbar()
            plt.legend()
            plt.tight_layout()                                                                                                                                                        
            plt.savefig(os.path.join(outpath_nfold, settings.name_target + '_train.png'), dpi = 300)
            if show:
                plt.show()


            ### Plot histogram of target values after mean subtraction 
            plt.clf()
            plt.hist(y_train, bins=30)
            plt.xlabel('Mean subtracted y_train')
            plt.ylabel('N')
            plt.savefig(os.path.join(outpath_nfold,'Hist_' + settings.name_target + '_train.png'), dpi = 300)
            if show:
                plt.show() 
            plt.close('all')  

            # optimise GP hyperparameters 
            # Use mean of X uncertainity for optimizing since otherwise too many local minima
            print('Mean of Y:  ' +str(np.round(np.mean(y_train),4)) + ' +/- ' + str(np.round(np.std(y_train),4))) 
            print('Mean of Mean function:  ' +str(np.round(np.mean(y_train_fmean),4)) + ' +/- ' + str(np.round(np.std(y_train_fmean),4))) 
            print('Mean of Mean function noise: ' +str(np.round(np.mean(ynoise_train),4)) + ' +/- ' + str(np.round(np.std(ynoise_train),4))) 
            print('Optimizing GP hyperparameters...')
            Xdelta_mean = Xdelta_train * 0 + np.nanmean(Xdelta_train,axis=0)
            opt_params, opt_logl = gp.optimize_gp_3D(points3D_train, y_train, ynoise_train, xymin = 30, zmin = 0.05, Xdelta = Xdelta_mean)
            #opt_params, opt_logl = optimize_gp_3D(points3D_train, y_train, ynoise_train, xymin = 30, zmin = 0.05, Xdelta = Xdelta_train)
            params_gp = opt_params

            # Calculate predicted mean values
            points3D_pred = points3D_test.copy()	

            print('Computing GP predictions for test set nfold ', ix)
            ypred, ypred_std, logl, gp_train = gp.train_predict_3D(points3D_train, points3D_pred, y_train, ynoise_train, params_gp, Ynoise_pred = ynoise_pred, Xdelta = Xdelta_train)
            ypred_train, _ , _ , _ = gp.train_predict_3D(points3D_train, points3D_train, y_train, ynoise_train, params_gp, Ynoise_pred = ynoise_train, Xdelta = Xdelta_train)

            # Add mean function to prediction
            if mean_function == 'rf':
                y_pred_zmean = ypred_rf
                y_pred_train_zmean = ypred_rf_train
            elif mean_function == 'blr':
                y_pred_zmean = ypred_blr
                y_pred_train_zmean = ypred_blr_train


            y_pred = ypred + y_pred_zmean
            y_pred_train = ypred_train + y_pred_train_zmean


            # Calculate Residual, RMSE, SSPE
            residual_test = y_pred - y_test
            rmse = np.sqrt(np.nanmean(residual_test**2))
            rmse_norm = rmse / y_test.std()
            rmedse = np.sqrt(np.median(residual_test**2))
            rmedse_norm = rmedse / y_test.std()
            #sspe = residual_test**2 / ystd_test**2
            sspe = residual_test**2 / (ypred_std**2)
            print("GP Marginal Log-Likelihood: ", np.round(logl,2))
            print("GP Normalized RMSE: ",np.round(rmse_norm,4))
            print("GP Normalized ROOT MEDIAN SE: ",np.round(rmedse_norm,4))
            print("GP Mean Theta: ", np.round(np.mean(sspe),4))
            print("GP Median Theta: ", np.round(np.median(sspe)))

            # Calculate also residual for mean function
            residual_fmean = y_pred_zmean - y_test
            rmse_fmean = np.sqrt(np.nanmean(residual_fmean**2))
            rmse_norm_fmean = rmse_fmean / y_test.std()
            rmedse_fmean = np.sqrt(np.median(residual_fmean**2))
            rmedse_norm_fmean = rmedse_fmean / y_test.std()
            print("Normalized RMSE of Mean function: ", np.round(rmse_norm_fmean,4))
            print("Normalized ROOT MEDIAN SE of Mean function: ", np.round(rmedse_norm_fmean,4))

            # Save results in dataframe
            dfpred[settings.name_target + '_GPmean'] = y_pred_zmean
            dfpred[settings.name_target + '_GPmean_residual'] = residual_fmean
            #dftest[name_target + '_GPmean_stddev'] = y_pred_zmean_std
            dfpred[settings.name_target + '_GPpredict'] = y_pred
            dfpred[settings.name_target + '_GPstddev'] = ypred_std
            dfpred['GPresidual'] = residual_test
            dfpred['GPresidual_squared'] = residual_test**2
            dfpred['Theta'] = sspe
            dfpred.to_csv(os.path.join(outpath_nfold, settings.name_target + '_results_nfold' + str(ix) + '.csv'), index = False)

            #Residual Map
            plt.figure(figsize=(8,6))
            #plt.imshow(ystd.reshape(len(yspace),len(xspace)) * np.nan,origin='lower', aspect = 'equal', extent = extent)
            #plt.scatter(dfsel.Easting.values - bound_xmin, dfsel.Northing.values - bound_ymin, c = dfsel.ESP.values, edgecolors = 'k')
            plt.scatter(dftest.x.values - bound_xmin, dftest.y.values - bound_ymin, c=residual_test, alpha=0.3)  
            plt.axis('equal')
            plt.xlabel('Easting')                                                                                                                                                                  
            plt.ylabel('Northing')                                                                                                                                                                 
            #plt.colorbar()                                                                                                                                                                         
            plt.title('Residual Test Data ' + settings.name_target) 
            plt.colorbar()
            plt.legend()
            plt.tight_layout()                                                                                                                                                        
            plt.savefig(os.path.join(outpath_nfold, settings.name_target + '_residualmap.png'), dpi = 300) 
            if show:
                plt.show() 

            # Residual Plot
            import seaborn as sns
            plt.subplot(2, 1, 1)
            sns.distplot(residual_test, norm_hist = True)
            plt.title(settings.name_target + ' Residual Analysis of Test Data')
            plt.ylabel('Residual')
            plt.subplot(2, 1, 2)
            sns.distplot(sspe, norm_hist = True)
            plt.ylabel(r'$\Theta$')
            plt.savefig(os.path.join(outpath_nfold, 'Residual_hist_' + settings.name_target + '_nfold' + str(ix) + '.png'), dpi=300)
            if show:
                plt.show() 
            plt.close('all')

            # Plot Y true vs predict for train and test set:
            plt.figure() # inches
            plt.scatter(y_train + y_train_fmean, y_pred_train, c = 'r', label='Train Set')
            #plt.title('BNN')
            plt.scatter(y_test, y_pred, c = 'b', label='Test Set')
            plt.xlabel(settings.name_target + ' True')
            plt.ylabel(settings.name_target + ' Predict')
            plt.legend()
            plt.savefig(os.path.join(outpath_nfold,'pred_vs_true' + settings.name_target + '_nfold' + str(ix) + '.png'), dpi = 300)
            if show:
                plt.show() 
            plt.close('all')


            rmse_nfold.append(rmse)
            nrmse_nfold.append(rmse_norm)
            rmedse_nfold.append(rmedse)
            nrmedse_nfold.append(rmedse_norm)
            meansspe_nfold.append(np.mean(sspe))
            rmse_fmean_nfold.append(rmse_fmean)
            nrmse_fmean_nfold.append(rmse_norm_fmean)
            rmedse_fmean_nfold.append(rmedse_fmean)
            nrmedse_fmean_nfold.append(rmedse_norm_fmean)
            histsspe.append(sspe)
            histresidual.append(residual_test)

        # Save and plot summary results of residual analysis for test data:
        rmse_nfold = np.asarray(rmse_nfold)
        nrmse_nfold = np.asarray(nrmse_nfold)
        rmedse_nfold = np.asarray(rmedse_nfold)
        nrmedse_nfold = np.asarray(nrmedse_nfold)
        meansspe_nfold = np.asarray(meansspe_nfold)
        rmse_fmean_nfold = np.asarray(rmse_fmean_nfold)
        nrmse_fmean_nfold = np.asarray(nrmse_fmean_nfold)
        rmedse_fmean_nfold = np.asarray(rmedse_fmean_nfold)
        nrmedse_fmean_nfold = np.asarray(nrmedse_fmean_nfold)
        dfsum = pd.DataFrame({'nfold': range_nfold, 'RMSE': rmse_nfold, 'nRMSE': nrmse_nfold, 'RMEDIANSE': rmedse_nfold, 'nRMEDIANSE': nrmedse_nfold, 'Theta': meansspe_nfold, 'RMSE_fmean': rmse_fmean_nfold, 'nRMSE_fmean': nrmse_fmean_nfold, 'RMEDIANSE_fmean': rmedse_fmean_nfold, 'nRMEDIANSE_fmean': nrmedse_fmean_nfold})
        dfsum.to_csv(os.path.join(outpath, settings.name_target + 'nfold_summary_stats.csv'), index = False)

        print("---- X-validation Summary -----")
        print("Mean normalized RMSE: " + str(np.round(np.mean(nrmse_nfold),3)) + " +/- " + str(np.round(np.std(nrmse_nfold),3)))
        print("Mean normalized RMSE of Meanfunction: " + str(np.round(np.mean(nrmse_fmean_nfold),3)) + " +/- " + str(np.round(np.std(nrmse_fmean_nfold),3)))
        print("Median normalized RMSE: " + str(np.round(np.median(nrmedse_nfold),3)))
        print("Median normalized RMSE of Meanfunction: " + str(np.round(np.median(nrmedse_fmean_nfold),3)))
        print("Mean Theta: " + str(np.round(np.mean(meansspe_nfold),3)) + " +/- " + str(np.round(np.std(meansspe_nfold),3)))

        histresidual_cut = truncate_data(histresidual, 1)
        histsspe_cut = truncate_data(histsspe, 1)
        plt.subplot(2, 1, 1)
        sns.distplot(histresidual_cut, norm_hist = True)
        plt.title(settings.name_target + ' Residual Analysis of Test Data')
        plt.ylabel('Residual')
        plt.subplot(2, 1, 2)
        sns.distplot(histsspe_cut, norm_hist = True)
        plt.ylabel(r'$\Theta$')
        #plt.title(valuename + ' SSPE Test')
        plt.savefig(os.path.join(outpath, 'Xvalidation_Residual_hist_' + settings.name_target + '.png'), dpi=300)
        if show:
            plt.show()
        plt.close('all')

        nrmse_meanfunction.append(np.round(np.mean(nrmse_nfold),3))
        nrmse_meanfunction_std.append(np.round(np.std(nrmse_nfold),3))
        theta_meanfunction.append(np.round(np.mean(meansspe_nfold),3))
        theta_meanfunction_std.append(np.round(np.std(meansspe_nfold),3))

    #End of xval loop over all models
    #Print best models sorted with nRMSE
    ix_meanfunction_sorted = [nrmse_meanfunction.index(x) for x in sorted(nrmse_meanfunction)]
    print('')
    print('-------------------------------')
    print('Models ranked based on nRMSE:')
    print('')
    for ix in ix_meanfunction_sorted:
        print(f'{settings.mean_functions[ix]}: Mean nRMSE = {nrmse_meanfunction[ix]} +/- {nrmse_meanfunction_std[ix]}, Theta = {theta_meanfunction[ix]} +/- {theta_meanfunction_std[ix]}')



