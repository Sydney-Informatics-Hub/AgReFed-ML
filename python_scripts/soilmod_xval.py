"""
Probabilistic machine learning models and evaluation using Gaussian Process Priors with mean functions.

Current models implemented:
- Gaussian Process with bayesian linear regression (BLR) as mean function and sparse spatial covariance function
- Gaussian Process with random forest (RF) regression as mean function and sparse spatial covariance function

Core functions:
- train baseline models (mean functions): BLR and RF
- hyperparameter optimisation of GP model
- n-fold cross-validation of models
- model evaluations: RMSE, NRMSE, R2, uncertainty of predictions
- residual plots and analysis
- ranking of best models

User settings, such as input/output paths and all other options, are set in the settings file 
(Default filename: settings_soilmodel_xval.yaml) 
Alternatively, the settings file can be specified as a command line argument with: 
'-s', or '--settings' followed by PATH-TO-FILE/FILENAME.yaml 
(e.g. python featureimportance.py -s settings_featureimportance.yaml).

See README.md for more information.

This package is part of the machine learning project developed for the Agricultural Research Federation (AgReFed).
"""

import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
import yaml
import argparse
from types import SimpleNamespace  

# Custom local libraries:
from utils import print2, truncate_data
from preprocessing import gen_kfold
import GPmodel as gp # GP model plus kernel functions and distance matrix calculation
import model_blr as blr
import model_rf as rf

# Settings yaml file
_fname_settings = 'settings_soilmod_xval.yaml'

# flag to show plot figures interactively or not (True/False)
_show = False

def runmodel(dfsel, model_function, settings):
    """
    Train model function on dataframe dfsel and return nfold cross-validation results.
    This function creates multiple diagnostic charts and evaluation statistics saved in the output folder.

    Input:
    ------
        dfsel: dataframe with data for training and testing (nfold column required to split data)
        model_function: str, function to train model (supported: 'blr', 'rf', 'blr-gp', 'rf-gp', 'gp-only')
        settings: settings for model function

    Returns:
    --------
        dfsum: dataframe with summary results
        stats_summary: list of summary statistics
        outpath: path to output files
    """
    outpath_root = settings.outpath

    # set conditional mean function
    if (model_function == 'blr') | (model_function == 'rf'):
        # only mean function model
        calc_mean_only = True
    else:
        calc_mean_only = False
    if (model_function == 'blr-gp') | (model_function == 'blr'):
        mean_function = 'blr'
        # print('mean function:', mean_function)
    if (model_function == 'rf-gp') | (model_function == 'rf'):
        mean_function = 'rf'
    if model_function == 'gp-only':
        mean_function = 'const'
        # print('mean function:', mean_function)

    # get train and test data, here we can include loop over ix for cross validation
    range_nfold = np.sort(dfsel.nfold.unique())
    
    print(f'Computing {len(range_nfold)}-fold xrossvalidation for function model: {model_function}')
    subdir = 'Xval_' + str(len(range_nfold)) + '-fold_' + model_function + '_' + settings.name_target
    outpath = os.path.join(outpath_root, subdir)

    ### X-fold Crossvalidation ###
    # Intialise lists to hold summary results for n-fold validation
    rmse_nfold = []
    nrmse_nfold = []
    rmedse_nfold = []
    nrmedse_nfold = []
    meansspe_nfold = []
    r2_nfold = []
    histresidual = []
    histsspe = []
    # dataframe to hold all predictions:
    dfpred_all = pd.DataFrame()

    # Loop over all folds
    for ix in range_nfold:
        # Loop over all train/test sets (test sets are designated by ix; training set is defined by the remaining set)
        print('Processing for nfold ', ix)
        # update outpath with iteration of cross-validation
        outpath_nfold = os.path.join(outpath, 'nfold_' + str(ix) + '/')
        os.makedirs(outpath_nfold, exist_ok = True)

        # split into train and test data
        dftrain = dfsel[dfsel[settings.name_ixval] != ix].copy()
        dftest = dfsel[dfsel[settings.name_ixval]  == ix].copy()

        # Copy dataframe for saving results later
        dfpred = dftest.copy() 

        points3D_train = np.asarray([dftrain.z.values, dftrain.y.values, dftrain.x.values]).T
        points3D_test = np.asarray([dftest.z.values, dftest.y.values, dftest.x.values]).T
        # Check for nan values:

        y_train = dftrain[settings.name_target].values
        y_test = dftest[settings.name_target].values
        # Uncertainty in coordinates
        if 'z_diff' in list(dftrain):
            Xdelta_train = np.asarray([0.5 * dftrain.z_diff.values, dftrain.y.values * 0, dftrain.x.values * 0.]).T
            Xdelta_test = np.asarray([0.5 * dftest.z_diff.values, dftest.y.values * 0, dftest.x.values * 0.]).T
        else:
            Xdelta_train = np.asarray([0 * dftrain.z.values, dftrain.y.values * 0, dftrain.x.values * 0.]).T
            Xdelta_test = np.asarray([0 * dftest.z.values, dftest.y.values * 0, dftest.x.values * 0.]).T

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
            if not calc_mean_only:
                plt.figure()  # inches
                plt.title('Random Forest Mean Function')
                plt.errorbar(y_train, ypred_rf_train, ynoise_train, linestyle='None', marker = 'o', c = 'r', label = 'Train Data', alpha =0.5)
                plt.errorbar(y_test, ypred_rf, ynoise_pred, linestyle='None', marker = 'o', c = 'b', label = 'Test Data', alpha =0.5)
                plt.legend(loc = 'upper left')
                plt.xlabel('y True')
                plt.ylabel('y Predict')
                plt.savefig(os.path.join(outpath_nfold, settings.name_target + '_meanfunction_pred_vs_true.png'), dpi = 300)
                if _show:
                    plt.show()
                plt.close('all')
        elif mean_function == 'blr':
            X_train = dftrain[settings.name_features].values
            y_train = dftrain[settings.name_target].values
            X_test = dftest[settings.name_features].values
            y_test = dftest[settings.name_target].values
            # Scale data
            Xs_train, ys_train, scale_params = blr.scale_data(X_train, y_train)
            scaler_x, scaler_y = scale_params
            Xs_test = scaler_x.transform(X_test)
            # Train BLR
            blr_model = blr.blr_train(Xs_train, y_train)
            # Predict for X_test
            ypred_blr, ypred_std_blr, nrmse_blr_test = blr.blr_predict(Xs_test, blr_model, y_test = y_test)
            ypred_blr_train,  ypred_std_blr_train, nrmse_blr_train = blr.blr_predict(Xs_train, blr_model, y_test = y_train)
            ypred_blr = ypred_blr.flatten()
            ypred_blr_train = ypred_blr_train.flatten()
            y_train_fmean = ypred_blr_train
            ynoise_train = ypred_std_blr_train #* fac_noise_train 
            ynoise_pred = ypred_std_blr #* fac_noise_pred
            if not calc_mean_only:
                plt.figure()  # inches
                plt.title('BLR Mean function')
                plt.errorbar(y_train, ypred_blr_train, ynoise_train, linestyle='None', marker = 'o', c = 'r', label = 'Train Data', alpha =0.5)
                plt.errorbar(y_test, ypred_blr, ynoise_pred, linestyle='None', marker = 'o', c = 'b', label = 'Test Data', alpha =0.5)
                plt.legend(loc = 'upper left')
                plt.xlabel('y True')
                plt.ylabel('y Predict')
                plt.savefig(os.path.join(outpath_nfold, settings.name_target + '_meanfunction_pred_vs_true.png'), dpi = 300)
                if _show:
                    plt.show()
                plt.close('all')
        elif mean_function == 'const':
            y_train_fmean = np.mean(y_train) * np.ones(y_train.shape)
            ypred_const = np.mean(y_train) * np.ones(y_test.shape)
            ypred_const_train = np.mean(y_train) * y_train_fmean
            ynoise_train = 1e-6 * np.ones(y_train.shape)
            ynoise_pred = 1e-6 * np.ones(y_test.shape)

        # Subtract mean function from training data for GP with zero mean
        y_train -= y_train_fmean

        # plot training and testing distribution
        plt.figure(figsize=(8,6))
        plt.scatter(dftrain.x.values, dftrain.y.values, alpha=0.3, c = 'b', label = 'Train') 
        plt.scatter(dftest.x.values, dftest.y.values, alpha=0.3, c = 'r', label = 'Test')  
        plt.axis('equal')
        plt.xlabel('Easting')                                                                                                                                                                  
        plt.ylabel('Northing')                                                                                                                                                                 
        #plt.colorbar()                                                                                                                                                                         
        plt.title('Mean subtracted ' + settings.name_target) 
        #plt.colorbar()
        plt.legend()
        plt.tight_layout()                                                                                                                                                        
        plt.savefig(os.path.join(outpath_nfold, settings.name_target + '_train.png'), dpi = 300)
        if _show:
            plt.show()

        ### Plot histogram of target values after mean subtraction 
        plt.clf()
        plt.hist(y_train, bins=30)
        plt.xlabel('Mean subtracted y_train')
        plt.ylabel('N')
        plt.savefig(os.path.join(outpath_nfold,'Hist_' + settings.name_target + '_train.png'), dpi = 300)
        if _show:
            plt.show() 
        plt.close('all')  

        if not calc_mean_only:
            # optimise GP hyperparameters 
            # Use mean of X uncertainity for optimizing since otherwise too many local minima
            print('Mean of Y:  ' +str(np.round(np.mean(y_train),4)) + ' +/- ' + str(np.round(np.std(y_train),4))) 
            print('Mean of Mean function:  ' +str(np.round(np.mean(y_train_fmean),4)) + ' +/- ' + str(np.round(np.std(y_train_fmean),4))) 
            print('Mean of Mean function noise: ' +str(np.round(np.mean(ynoise_train),4)) + ' +/- ' + str(np.round(np.std(ynoise_train),4))) 
            print('Optimizing GP hyperparameters...')
            Xdelta_mean = Xdelta_train * 0 + np.nanmean(Xdelta_train,axis=0)
            # TBD: find automatic way to set hyperparameter boundaries based on data
            xymin = 0.5 * (points3D_train[:,1].max() - points3D_train[:,1].min()) / np.unique(points3D_train[:,1]).size
            zmin = 0.5 * (points3D_train[:,0].max() - points3D_train[:,0].min()) / np.unique(points3D_train[:,0]).size
            opt_params, opt_logl = gp.optimize_gp_3D(points3D_train, y_train, ynoise_train, xymin = xymin, zmin = zmin, Xdelta = Xdelta_mean)
            #opt_params, opt_logl = optimize_gp_3D(points3D_train, y_train, ynoise_train, xymin = 30, zmin = 0.05, Xdelta = Xdelta_train)
            params_gp = opt_params

            # Calculate predicted mean values
            points3D_pred = points3D_test.copy()	
            print('Computing GP predictions for test set nfold ', ix)
            ypred, ypred_std, logl, gp_train = gp.train_predict_3D(points3D_train, points3D_pred, y_train, ynoise_train, params_gp, Ynoise_pred = ynoise_pred, Xdelta = Xdelta_train)
            ypred_train, ypred_std_train, _ , _ = gp.train_predict_3D(points3D_train, points3D_train, y_train, ynoise_train, params_gp, Ynoise_pred = ynoise_train, Xdelta = Xdelta_train)
        else:
            ypred = 0
            ypred_train =0 
            ypred_std = ynoise_pred
            ypred_std_train = ynoise_train

        # Add mean function to prediction
        if mean_function == 'rf':
            y_pred_zmean = ypred_rf
            y_pred_train_zmean = ypred_rf_train
        elif mean_function == 'blr':
            y_pred_zmean = ypred_blr
            y_pred_train_zmean = ypred_blr_train
        elif mean_function == 'const':
            y_pred_zmean = ypred_const
            y_pred_train_zmean = ypred_const_train

        y_pred = ypred + y_pred_zmean
        y_pred_train = ypred_train + y_pred_train_zmean
        y_train += y_train_fmean

        # Calculate Residual, RMSE, R2, SSPE
        residual_test = y_pred - y_test
        rmse = np.sqrt(np.nanmean(residual_test**2))
        rmse_norm = rmse / y_test.std()
        rmedse = np.sqrt(np.median(residual_test**2))
        rmedse_norm = rmedse / y_test.std()
        #sspe = residual_test**2 / ystd_test**2
        sspe = residual_test**2 / (ypred_std**2)
        r2 = 1 - np.nanmean(residual_test**2) / np.nanmean((y_test - y_test.mean())**2)
        if not calc_mean_only:
            print("GP Marginal Log-Likelihood: ", np.round(logl,2))
        print("Normalized RMSE: ",np.round(rmse_norm,4))
        print("Normalized ROOT MEDIAN SE: ",np.round(rmedse_norm,4))
        print("R^2: ", np.round(r2,4))
        print("Mean Theta: ", np.round(np.mean(sspe),4))
        print("Median Theta: ", np.round(np.median(sspe)))

        # Save results in dataframe
        dfpred[settings.name_target + '_predict'] = y_pred
        dfpred[settings.name_target + '_stddev'] = ypred_std
        dfpred['Residual'] = residual_test
        dfpred['Residual_squared'] = residual_test**2
        dfpred['Theta'] = sspe
        dfpred.to_csv(os.path.join(outpath_nfold, settings.name_target + '_results_nfold' + str(ix) + '.csv'), index = False)
        # add to dataframe for all folds
        dfpred_all = pd.concat([dfpred_all, dfpred], axis=0, ignore_index = True)

        #Residual Map
        plt.figure(figsize=(8,6))
        plt.scatter(dftest.x.values, dftest.y.values, c=residual_test, alpha=0.3)  
        plt.axis('equal')
        plt.xlabel('Easting')                                                                                                                                                                  
        plt.ylabel('Northing')                                                                                                                                                                 
        #plt.colorbar()                                                                                                                                                                         
        plt.title('Residual Test Data ' + settings.name_target) 
        plt.colorbar()
        #plt.legend()
        plt.tight_layout()                                                                                                                                                        
        plt.savefig(os.path.join(outpath_nfold, settings.name_target + '_residualmap.png'), dpi = 300) 
        if _show:
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
        if _show:
            plt.show() 
        plt.close('all')

        plt.figure() 
        # plt.title(model_function)
        plt.errorbar(y_train, y_pred_train, ypred_std_train, linestyle='None', marker = 'o', c = 'r', label = 'Train Data', alpha =0.5)
        plt.errorbar(y_test, y_pred, ypred_std, linestyle='None', marker = 'o', c = 'b', label = 'Test Data', alpha =0.5)
        plt.legend(loc = 'upper left')
        plt.xlabel(settings.name_target + ' True')
        plt.ylabel(settings.name_target + ' Predict')
        plt.savefig(os.path.join(outpath_nfold,'pred_vs_true' + settings.name_target + '_nfold' + str(ix) + '.png'), dpi = 300)
        if _show:
            plt.show()
        plt.close('all')

        rmse_nfold.append(rmse)
        nrmse_nfold.append(rmse_norm)
        rmedse_nfold.append(rmedse)
        nrmedse_nfold.append(rmedse_norm)
        meansspe_nfold.append(np.mean(sspe))
        r2_nfold.append(r2)
        histsspe.append(sspe)
        histresidual.append(residual_test)

    # Save prediton results in dataframe
    print("Saving all predictions in dataframe ...")
    dfpred_all.to_csv(os.path.join(outpath, settings.name_target + '_results_nfold_all.csv'), index = False)

    # Plot all predictions vs true for test data
    plt.figure() 
    plt.title('Combined Test Data')
    plt.errorbar(dfpred_all[settings.name_target], 
        dfpred_all[settings.name_target + '_predict'], 
        dfpred_all[settings.name_target + '_stddev'], 
        linestyle='None', marker = 'o', c = 'b', alpha = 0.5)
    plt.xlabel(settings.name_target + ' True')
    plt.ylabel(settings.name_target + ' Predict')
    plt.savefig(os.path.join(outpath,'pred_vs_true' + settings.name_target + '_combined.png'), dpi = 300)
    if _show:
        plt.show()
    plt.close('all')

    # Save and plot summary results of residual analysis for test data:
    rmse_nfold = np.asarray(rmse_nfold)
    nrmse_nfold = np.asarray(nrmse_nfold)
    rmedse_nfold = np.asarray(rmedse_nfold)
    nrmedse_nfold = np.asarray(nrmedse_nfold)
    meansspe_nfold = np.asarray(meansspe_nfold)
    r2_nfold = np.asarray(r2_nfold)
    dfsum = pd.DataFrame({'nfold': range_nfold, 'RMSE': rmse_nfold, 'nRMSE': nrmse_nfold, 'RMEDIANSE': rmedse_nfold, 'nRMEDIANSE': nrmedse_nfold, 'R2': r2_nfold, 'Theta': meansspe_nfold})
    dfsum.to_csv(os.path.join(outpath, settings.name_target + 'nfold_summary_stats.csv'), index = False)

    print("---- X-validation Summary -----")
    print("Mean normalized RMSE: " + str(np.round(np.mean(nrmse_nfold),3)) + " +/- " + str(np.round(np.std(nrmse_nfold),3)))
    print("Median normalized RMSE: " + str(np.round(np.median(nrmedse_nfold),3)))
    print("Mean R^2: " + str(np.round(np.mean(r2_nfold),3)))
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
    if _show:
        plt.show()
    plt.close('all')

    stats_summary = (np.round(np.mean(nrmse_nfold),3), np.round(np.std(nrmse_nfold),3),  
    np.round(np.mean(meansspe_nfold),3), np.round(np.std(meansspe_nfold),3),  
    np.round(np.mean(r2_nfold),3), np.round(np.std(r2_nfold),3) )
    return dfsum, stats_summary, outpath


######################### Main Function ############################

def main(fname_settings):
    """
    Main script for running 3D cubing with Gaussian Process.
    See Documentation and comments below for more details.
    """
    # Load settings from yaml file
    with open(fname_settings, 'r') as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
    # Parse settings dictinary as namespace (settings are available as 
    # settings.variable_name rather than settings['variable_name'])
    settings = SimpleNamespace(**settings)

    # Add temporal or vertical componnet
    if settings.axistype == 'temporal':
        settings.colname_zcoord = settings.colname_tcoord
        settings.colname_zmin = settings.colname_tmin
        settings.colname_zmax =  settings.colname_tmax

    if type(settings.model_functions) != list:
        settings.model_functions = [settings.model_functions]

    # check if outpath exists, if not create direcory
    os.makedirs(settings.outpath, exist_ok = True)

    # Intialise output info file:
    print2('init')
    print2(f'--- Parameter Settings ---')
    print2(f'Selected Model Functions: {settings.model_functions}')
    print2(f'Target Name: {settings.name_target}')
    print2(f'--------------------------')

    print('Reading data into dataframe...')
    # Read in data
    dfsel = pd.read_csv(os.path.join(settings.inpath, settings.infname))

    # Rename x and y coordinates of input data
    if settings.colname_xcoord != 'x':
        dfsel.rename(columns={settings.colname_xcoord: 'x'}, inplace = True)
    if settings.colname_ycoord != 'y':
        dfsel.rename(columns={settings.colname_ycoord: 'y'}, inplace = True)
    if (settings.axistype == 'vertical') & (settings.colname_zcoord != 'z'):
        dfsel.rename(columns={settings.colname_zcoord: 'z'}, inplace = True)
    else:
        dfsel.rename(columns={settings.colname_tcoord: 'z'}, inplace = True)
        dfsel.rename(columns={settings.colname_zcoord: 'z'}, inplace = True)
    settings.name_features.append('z')
 
    # Select data between zmin and zmax
    dfsel = dfsel[(dfsel['z'] >= settings.colname_zmin) & (dfsel['z'] <= settings.colname_zmax)]

    # Generate kfold indices
    if settings.axistype == 'vertical':
        dfsel = gen_kfold(dfsel, nfold = settings.nfold, label_nfold = 'nfold', id_unique = ['x','y'], precision_unique = 0.01)
    elif settings.axistype == 'temporal':
        #dfsel = gen_kfold(dfsel, nfold = settings.nfold, label_nfold = 'nfold', id_unique = ['x', 'y', 'z'], precision_unique = 0.01)
        dfsel = gen_kfold(dfsel, nfold = settings.nfold, label_nfold = 'nfold', id_unique = ['x', 'y'], precision_unique = 0.01)

    ## Get coordinates for training data and set coord origin to (0,0)
    bound_xmin = dfsel.x.min()
    bound_xmax = dfsel.x.max()
    bound_ymin = dfsel.y.min()
    bound_ymax = dfsel.y.max()

    # Set origin to (0,0)
    dfsel['x'] = dfsel['x'] - bound_xmin
    dfsel['y'] = dfsel['y'] - bound_ymin

    nrmse_meanfunction = []
    nrmse_meanfunction_std = []
    theta_meanfunction = []
    theta_meanfunction_std = []
    r2_meanfunction = []
    r2_meanfunction_std = []

    # Loop over model functions and evaluate
    for model_function in settings.model_functions:
        # run and evaluate model
        dfsum, stats_summary, model_outpath = runmodel(dfsel, model_function, settings)
        print(f'All output files of {model_function} saved in {model_outpath}')
        print('')
        # save results
        nrmse_meanfunction.append(stats_summary[0])
        nrmse_meanfunction_std.append(stats_summary[1])
        theta_meanfunction.append(stats_summary[2])
        theta_meanfunction_std.append(stats_summary[3])
        r2_meanfunction.append(stats_summary[4])
        r2_meanfunction_std.append(stats_summary[5])

    #End of xval loop over all models
    #Print best models sorted with nRMSE
    ix_meanfunction_sorted = [nrmse_meanfunction.index(x) for x in sorted(nrmse_meanfunction)]
    print('')
    print('-------------------------------')
    print('Models ranked based on nRMSE:')
    print('')
    for ix in ix_meanfunction_sorted:
        print(f'{settings.model_functions[ix]}: Mean nRMSE = {nrmse_meanfunction[ix]} +/- {nrmse_meanfunction_std[ix]}, Mean R2= {r2_meanfunction[ix]} +/- {r2_meanfunction_std[ix]}, Theta = {theta_meanfunction[ix]} +/- {theta_meanfunction_std[ix]}')


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Prediction model for machine learning on soil data.')
    parser.add_argument('-s', '--settings', type=str, required=False,
                        help='Path and filename of settings file.',
                        default = _fname_settings)
    args = parser.parse_args()

    # Run main function
    main(args.settings)