"""
Machine Learning model for 3D Cube Soil Generator using Gaussian Process Priors with mean functions. 

Current models implemented:
- Bayesian linear regression (BLR) 
- Random forest (RF)
- Gaussian Process with bayesian linear regression (BLR) as mean function and sparse spatial covariance function
- Gaussian Process with random forest (RF) regression as mean function and sparse spatial covariance function

Core functionality:
- Training of model and GP, including hyperparameter optimization
- generating soil property predictions and uncertainties for multiple depths or time steps
- taking into account measurement errors  and uncertainties in measurement locations
- spatial support for predictions: points, volume blocks, polygons
- spatial uncertainty integration takes into account spatial covariances between points

See documentation for more details.

User settings, such as input/output paths and all other options, are set in the settings file 
(Default filename: settings_soilmodel_predict.yaml) 
Alternatively, the settings file can be specified as a command line argument with: 
'-s', or '--settings' followed by PATH-TO-FILE/FILENAME.yaml 
(e.g. python featureimportance.py -s settings_featureimportance.yaml).

This package is part of the machine learning project developed for the Agricultural Research Federation (AgReFed).
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import os
import sys
from scipy.special import erf
from scipy.interpolate import interp1d, griddata
import matplotlib.pyplot as plt
import subprocess
from sklearn.model_selection import train_test_split 
# Save and load trained models and scalers:
import pickle
import json
import yaml
import argparse
from types import SimpleNamespace  
from tqdm import tqdm

# AgReFed modules:
from utils import array2geotiff, align_nearest_neighbor, print2, truncate_data
from sigmastats import averagestats
from preprocessing import gen_kfold
import GPmodel as gp # GP model plus kernel functions and distance matrix calculation
import model_blr as blr
import model_rf as rf

# Settings yaml file
_fname_settings = 'settings_soilmod_predict.yaml'

# flag to show plot figures interactively or not (True/False)
_show = False

### Some colormap settings (Default settings)
# Choose colormap, default Matplotlib colormaps:
# add ending '_r' at end for inverse of standard color 
# For prediction maps (default 'viridis'):
colormap_pred = 'viridis' 
# For uncertainity prediction maps (default 'viridis')
colormap_pred_std =  'viridis'
# For probability exceeding treshold maps (default 'coolwarm')
colormap_prob = 'coolwarm'
# Or use seaborn colormaps 

######### Volume Block prediction #########
def model_blocks(settings):
    """
    Predict soil properties and uncertainties for block sizes rather than points.
    The predicted uncertainty takes into account spatial covariance within each block
    All output is saved in output directory as specified in settings.

    Parameters
    ----------
        settings : settings namespace

    Return:
    -------
        mu_3d: prediction cube
        std_std: corresponding cube of standard deviation

    """
    if (settings.model_function == 'blr') | (settings.model_function == 'rf'):
        # only mean function model
        calc_mean_only = True
    else:
        calc_mean_only = False
    if (settings.model_function == 'blr-gp') | (settings.model_function == 'blr'):
        mean_function = 'blr'
    if (settings.model_function == 'rf-gp') | (settings.model_function == 'rf'):
        mean_function = 'rf'

    # set conditional settings
    if calc_mean_only:
        settings.optimize_GP = False

    # set blocksizes
    settings.xblocksize = settings.yblocksize = settings.xyblocksize

    # currently assuming resolution is the same in x and y direction
    settings.xvoxsize = settings.yvoxsize = settings.xyvoxsize

    Nvoxel_per_block = settings.xblocksize * settings.yblocksize * settings.zblocksize / (settings.xvoxsize * settings.yvoxsize * settings.zvoxsize)
    print("Number of evaluation points per block: ", Nvoxel_per_block)

    # check if outpath exists, if not create direcory
    os.makedirs(settings.outpath, exist_ok = True)

    # Intialise output info file:
    print2('init')
    print2(f'--- Parameter Settings ---')
    print2(f'Model Function: {settings.model_function}')
    print2(f'Target Name: {settings.name_target}')
    print2(f'Prediction geometry: Volume')
    print2(f'x,y,z blocksize: {settings.xyblocksize,settings.xyblocksize, settings.zblocksize}')
    print2(f'--------------------------')

    print('Reading in data...')
    # Read in data for model training:
    dftrain = pd.read_csv(os.path.join(settings.inpath, settings.infname))

    # Rename x and y coordinates of input data
    if settings.colname_xcoord != 'x':
        dftrain.rename(columns={settings.colname_xcoord: 'x'}, inplace = True)
    if settings.colname_ycoord != 'y':
        dftrain.rename(columns={settings.colname_ycoord: 'y'}, inplace = True)
    if settings.colname_zcoord != 'z':
        dftrain.rename(columns={settings.colname_zcoord: 'z'}, inplace = True)
        name_features_grid = settings.name_features.copy()
        if settings.colname_zcoord in settings.name_features:
            settings.name_features.remove(settings.colname_zcoord)
            name_features_grid.remove(settings.colname_zcoord)
            settings.name_features.append('z')
            name_features = settings.name_features

    # Convert z and z_diff to meters if in cm (>10):
    if (dftrain['z'].max() > 10):
        dftrain['z'] = dftrain['z'] / 100
        if 'z_diff' in dftrain.columns:
            dftrain['z_diff'] = dftrain['z_diff'] / 100
  
    # Select data between zmin and zmax
    dftrain = dftrain[(dftrain['z'] >= settings.zmin) & (dftrain['z'] <= settings.zmax)]

    name_features = settings.name_features

    # check if z_diff is in dftrain
    if 'z_diff' not in dftrain.columns:
        dftrain['z_diff'] = 0.0

    # read in covariate grid:
    dfgrid = pd.read_csv(os.path.join(settings.inpath, settings.gridname))

    # Rename x and y coordinates of input data
    if settings.colname_xcoord != 'x':
        dfgrid.rename(columns={settings.colname_xcoord: 'x'}, inplace = True)
    if settings.colname_ycoord != 'y':
        dfgrid.rename(columns={settings.colname_ycoord: 'y'}, inplace = True)
    if settings.colname_zcoord != 'z':
        dfgrid.rename(columns={settings.colname_zcoord: 'z'}, inplace = True)
    settings.name_features.append('z')

    ## Get coordinates for training data and set coord origin to (0,0)  
    bound_xmin = dfgrid.x.min() - 0.5* settings.xvoxsize
    bound_xmax = dfgrid.x.max() + 0.5* settings.xvoxsize
    bound_ymin = dfgrid.y.min() - 0.5* settings.yvoxsize
    bound_ymax = dfgrid.y.max() + 0.5* settings.yvoxsize

    dfgrid['x'] = dfgrid.x - bound_xmin
    dfgrid['y'] = dfgrid.y - bound_ymin
	
    # Define grid coordinates:
    points3D_train = np.asarray([dftrain.z.values, dftrain.y.values - bound_ymin, dftrain.x.values - bound_xmin ]).T

    # Define y target
    y_train = dftrain[settings.name_target].values

    # spatial uncertainty of coordinates:
    Xdelta_train = np.asarray([0.5 * dftrain.z_diff.values, dftrain.y.values * 0, dftrain.x.values * 0.]).T

    # Calculate predicted mean values of training data
    X_train = dftrain[settings.name_features].values
    y_train = dftrain[settings.name_target].values
    if mean_function == 'rf':
        # Estimate GP mean function with Random Forest
        rf_model = rf.rf_train(X_train, y_train)
        ypred_rf_train, ynoise_train, nrmse_rf_train = rf.rf_predict(X_train, rf_model, y_test = y_train)
        y_train_fmean = ypred_rf_train
    elif mean_function == 'blr':
        # Scale data
        Xs_train, ys_train, scale_params = blr.scale_data(X_train, y_train)
        scaler_x, scaler_y = scale_params
        # Train BLR
        blr_model = blr.blr_train(Xs_train, y_train)
        # Predict for X_test
        ypred_blr_train, ypred_std_blr_train, nrmse_blr_train = blr.blr_predict(Xs_train, blr_model, y_test = y_train)
        ypred_blr_train = ypred_blr_train.flatten()
        y_train_fmean = ypred_blr_train
        ynoise_train = ypred_std_blr_train

    # Subtract mean function from training data 
    y_train -= y_train_fmean

    if not calc_mean_only:
        # optimise GP hyperparameters 
        # Use mean of X uncertainity for optimizing since otherwise too many local minima
        print('Optimizing GP hyperparameters...')
        Xdelta_mean = Xdelta_train * 0 + np.nanmean(Xdelta_train,axis=0)
        opt_params, opt_logl = gp.optimize_gp_3D(points3D_train, y_train, ynoise_train, 
            xymin = settings.xyvoxsize, 
            zmin = settings.zvoxsize,  
            Xdelta = Xdelta_mean)
        params_gp = opt_params

    # Set extent of prediction grid
    extent = (0,bound_xmax - bound_xmin, 0, bound_ymax - bound_ymin)

    # Set output path for figures for each depth or temporal slice
    outpath_fig = os.path.join(settings.outpath, 'Figures_zslices/')
    os.makedirs(outpath_fig, exist_ok = True)	

    xblock = np.arange(dfgrid['x'].min(), dfgrid['x'].max(), settings.xblocksize) + 0.5 * settings.xblocksize
    yblock = np.arange(dfgrid['y'].min(), dfgrid['y'].max(), settings.yblocksize) + 0.5 * settings.yblocksize
    if (len(settings.list_z_pred) > 0) & (settings.list_z_pred is not None) &  (settings.list_z_pred != 'None'):
        zblock = np.asarray(settings.list_z_pred)
    else:
        zblock = np.arange(0.5 * settings.zblocksize, settings.zmax + 0.5 * settings.zblocksize, settings.zblocksize)
    block_x, block_y = np.meshgrid(xblock, yblock)
    block_shape = block_x.shape
    block_x = block_x.flatten()
    block_y = block_y.flatten()
    mu_3d = np.zeros((len(xblock), len(yblock), len(zblock)))
    std_3d = np.zeros((len(xblock), len(yblock), len(zblock)))
    mu_block = np.zeros_like(block_x)
    std_block = np.zeros_like(block_x)
    # Set initial optimisation of hyperparamter to True
    gp_train_flag = True
    # Slice in blocks for prediction calculating per 30 km x 1cm

    for i in range(len(zblock)):
        if settings.axistype == 'vertical':
            # predict for each depth  slice
            print('Computing slice at depth: ' + str(np.round(100 * zblock[i])) + 'cm')
        if settings.axistype == 'temporal':
            # predict for each temporal slice
            print('Computing slice at date: ' + str(np.round(100 * zblock[i])))
        zrange = np.arange(zblock[i] - 0.5 * settings.zblocksize, zblock[i] + 0.5 * settings.zblocksize + settings.zvoxsize, settings.zvoxsize)
        ix_start = 0
        # Progressbar
        for j in tqdm(range(len(block_x.flatten()))):
            dftest = dfgrid[(dfgrid.x >= block_x[j] - 0.5 * settings.xblocksize) & (dfgrid.x <= block_x[j] + 0.5 * settings.xblocksize) &
                (dfgrid.y >= block_y[j] - 0.5 * settings.yblocksize) & (dfgrid.y <= block_y[j] + 0.5 * settings.yblocksize)].copy()
            if len(dftest) > 0:
                dfnew = dftest.copy()
                for z in zrange:
                    if z == zrange[0]:
                        dftest['z'] = z 
                    else:
                        dfnew['z'] = z
                        dftest = dftest.append(dfnew, ignore_index = True)									
                ysel = dftest.y.values
                xsel = dftest.x.values
                zsel = dftest.z.values
                points3D_pred = np.asarray([zsel, ysel, xsel]).T		
                # Calculate mean function for prediction

                if mean_function == 'rf':
                    X_test = dftest[settings.name_features].values
                    ypred_rf, ynoise_pred, _ = rf.rf_predict(X_test, rf_model)
                    y_pred_zmean = ypred_rf
                elif mean_function == 'blr':
                    X_test = dftest[settings.name_features].values
                    Xs_test = scaler_x.transform(X_test)
                    ypred_blr, ypred_std_blr, _ = blr.blr_predict(Xs_test, blr_model)
                    ypred_blr = ypred_blr.flatten()
                    y_pred_zmean = ypred_blr
                    ynoise_pred = ypred_std_blr

                # GP Prediction:
                if not calc_mean_only:
                    if gp_train_flag:
                        # Need to calculate matrix gp_train only once, then used subsequently for all other predictions
                        ypred, ystd, logl, gp_train, covar = gp.train_predict_3D(points3D_train, points3D_pred, y_train, ynoise_train, params_gp, 
                            Ynoise_pred = ynoise_pred, Xdelta = Xdelta_train, out_covar = True) 
                        gp_train_flag = False
                    else:
                        ypred, ystd, covar = gp.predict_3D(points3D_train, points3D_pred, gp_train, params_gp, Ynoise_pred = ynoise_pred, Xdelta = Xdelta_train, 
                            out_covar = True)
                else:
                    ypred = y_pred_zmean
                    ystd = ynoise_pred

                #### Need to calculate weighted average from covar and ypred
                if not calc_mean_only:
                    ypred_block, ystd_block = averagestats(ypred + y_pred_zmean, covar)
                else:
                    ypred_block, ystd_block = averagestats(ypred, covar)

                # Save results in block array
                mu_block[j] = ypred_block
                std_block[j] = ystd_block

            # Set blocks where there is no data to nan
            else:
                mu_block[j] = np.nan
                std_block[j] = np.nan

        # map coordinate array to image and save in 3D
        mu_img = mu_block.reshape(block_shape)
        std_img = std_block.reshape(block_shape)

        if settings.axistype == 'vertical':
            np.savetxt(os.path.join(outpath_fig, 'Pred_' + settings.name_target + '_z' + str("{:03d}".format(int(np.round(100 * zblock[i])))) + 'cm.txt'), np.round(mu_img.flatten(),3), delimiter=',')
            np.savetxt(os.path.join(outpath_fig, 'Pred_Stddev_' + settings.name_target + '_z' + str("{:03d}".format(int(np.round(100 * zblock[i])))) + 'cm.txt'), np.round(std_img.flatten(),3), delimiter=',')
        if settings.axistype == 'temporal':
            np.savetxt(os.path.join(outpath_fig, 'Pred_' + settings.name_target + '_t' + str("{:03d}".format(int(np.round(100 * zblock[i])))) + '.txt'), np.round(mu_img.flatten(),3), delimiter=',')
            np.savetxt(os.path.join(outpath_fig, 'Pred_Stddev_' + settings.name_target + '_t' + str("{:03d}".format(int(np.round(100 * zblock[i])))) + '.txt'), np.round(std_img.flatten(),3), delimiter=',')
        if i == 0:
            # Create coordinate array of x and y
            np.savetxt(os.path.join(outpath_fig, 'Pred_' + settings.name_target + '_coord_x.txt'), block_x, delimiter=',')
            np.savetxt(os.path.join(outpath_fig, 'Pred_' + settings.name_target + '_coord_y.txt'), block_y, delimiter=',')

        mu_3d[:,:,i] = mu_img.T
        std_3d[:,:,i] = std_img.T

        # Create Result Plots
        mu_3d_trim = mu_3d[:,:,i].copy()
        mu_3d_trim_max = np.percentile(mu_3d_trim[~np.isnan(mu_3d_trim)], 99.5)
        mu_3d_trim[mu_3d_trim > mu_3d_trim_max] = mu_3d_trim_max
        mu_3d_trim[mu_3d_trim < 0] = 0
        plt.figure(figsize = (8,8))
        plt.subplot(2, 1, 1)
        plt.imshow(mu_3d_trim.T,origin='lower', aspect = 'equal', extent = extent, cmap = colormap_pred)
        if settings.axistype == 'vertical':
            plt.title(settings.name_target + ' Depth ' + str(np.round(100 * zblock[i])) + 'cm')
        elif settings.axistype == 'temporal':
            plt.title(settings.name_target + ' Date ' + str(np.round(100 * zblock[i])))
        plt.ylabel('Northing [meters]')
        plt.colorbar()
        plt.subplot(2, 1, 2)
        std_3d_trim = std_3d[:,:,i].copy()
        std_3d_trim_max = np.percentile(std_3d_trim[~np.isnan(std_3d_trim)], 99.5)
        std_3d_trim[std_3d_trim > std_3d_trim_max] = std_3d_trim_max
        plt.imshow(std_3d_trim.T,origin='lower', aspect = 'equal', extent = extent, cmap = colormap_pred_std)
        if settings.axistype == 'vertical':
            plt.title('Std Dev ' + settings.name_target + ' Depth ' + str(np.round(100 * zblock[i])) + 'cm')
        elif settings.axistype == 'temporal':
            plt.title(settings.name_target + ' Date ' + str(np.round(100 * zblock[i])))
        plt.colorbar()
        plt.xlabel('Easting [meters]')
        plt.ylabel('Northing [meters]')
        plt.tight_layout()
        if settings.axistype == 'vertical':
            plt.savefig(os.path.join(outpath_fig, 'Pred_' + settings.name_target + '_z' + str("{:03d}".format(int(np.round(100 * zblock[i])))) + 'cm.png'), dpi=300)
        elif settings.axistype == 'temporal':
            plt.savefig(os.path.join(outpath_fig, 'Pred_' + settings.name_target + '_t' + str("{:03d}".format(int(np.round(100 * zblock[i])))) + '.png'), dpi=300)
        if _show:
            plt.show()
        plt.close('all')   

        #Save also as geotiff
        if settings.axistype == 'vertical':
            outfname_tif = os.path.join(outpath_fig, 'Pred_' + settings.name_target + '_z' + str("{:03d}".format(int(np.round(100 * zblock[i])))) + 'cm.tif')
            outfname_tif_std = os.path.join(outpath_fig, 'Std_' + settings.name_target + '_z' + str("{:03d}".format(int(np.round(100 * zblock[i])))) + 'cm.tif')
        elif settings.axistype == 'temporal':
            outfname_tif = os.path.join(outpath_fig, 'Pred_' + settings.name_target + '_t' + str("{:03d}".format(int(np.round(100 * zblock[i])))) + '.tif')
            outfname_tif_std = os.path.join(outpath_fig, 'Std_' + settings.name_target + '_t' + str("{:03d}".format(int(np.round(100 * zblock[i])))) + '.tif')
        #print('Saving results as geo tif...')
        tif_ok = array2geotiff(mu_img, [bound_xmin + 0.5 * settings.xblocksize,bound_ymin + 0.5 * settings.yblocksize], [settings.xblocksize,settings.yblocksize], outfname_tif, settings.project_crs)
        tif2_ok = array2geotiff(std_img, [bound_xmin + 0.5 * settings.xblocksize,bound_ymin + 0.5 * settings.yblocksize], [settings.xblocksize,settings.yblocksize], outfname_tif_std, settings.project_crs)

    return mu_3d, std_3d


######### Point prediction #########
def model_points(settings):
    """
    Predict soil properties and uncertainties for location points.
    All output is saved in output directory as specified in settings.

    Parameters
    ----------
        settings : settings namespace

    Return:
    -------
        mu_3d: prediction cube
        std_3d: standard deviation cube
    """

    if (settings.model_function == 'blr') | (settings.model_function == 'rf'):
        # only mean function model
        calc_mean_only = True
    else:
        calc_mean_only = False
    if (settings.model_function == 'blr-gp') | (settings.model_function == 'blr'):
        mean_function = 'blr'
    if (settings.model_function == 'rf-gp') | (settings.model_function == 'rf'):
        mean_function = 'rf'


    # set conditional settings
    if calc_mean_only:
        settings.optimize_GP = False

    # currently assuming resolution is the same in x and y direction
    settings.xvoxsize = settings.yvoxsize = settings.xyvoxsize

    # check if outpath exists, if not create direcory
    os.makedirs(settings.outpath, exist_ok = True)

    # Intialise output info file:
    print2('init')
    print2(f'--- Parameter Settings ---')
    print2(f'Model Function: {settings.model_function}')
    print2(f'Target Name: {settings.name_target}')
    print2(f'Prediction geometry: Point')
    print2(f'x,y,z voxsize: {settings.xyvoxsize,settings.xyvoxsize, settings.zvoxsize}')
    print2(f'--------------------------')

    print('Reading in data...')
    # Read in data for model training:
    dftrain = pd.read_csv(os.path.join(settings.inpath, settings.infname))

    # Rename x and y coordinates of input data
    if settings.colname_xcoord != 'x':
        dftrain.rename(columns={settings.colname_xcoord: 'x'}, inplace = True)
    if settings.colname_ycoord != 'y':
        dftrain.rename(columns={settings.colname_ycoord: 'y'}, inplace = True)
    if settings.colname_zcoord != 'z':
        dftrain.rename(columns={settings.colname_zcoord: 'z'}, inplace = True)
        if settings.colname_zcoord in settings.name_features:
            settings.name_features.remove(settings.colname_zcoord)
            settings.name_features.append('z')

    # Convert z and z_diff to meters if in cm (>10):
    if (dftrain['z'].max() > 10):
        dftrain['z'] = dftrain['z'] / 100
        if 'z_diff' in dftrain.columns:
            dftrain['z_diff'] = dftrain['z_diff'] / 100
  
    # Select data between zmin and zmax
    dftrain = dftrain[(dftrain['z'] >= settings.zmin) & (dftrain['z'] <= settings.zmax)]

    name_features = settings.name_features
 
    # check if z_diff is in dftrain
    if 'z_diff' not in dftrain.columns:
        dftrain['z_diff'] = 0.0

    # read in covariate grid:
    dfgrid = pd.read_csv(os.path.join(settings.inpath, settings.gridname))

    # Rename x and y coordinates of input data
    if settings.colname_xcoord != 'x':
        dfgrid.rename(columns={settings.colname_xcoord: 'x'}, inplace = True)
    if settings.colname_ycoord != 'y':
        dfgrid.rename(columns={settings.colname_ycoord: 'y'}, inplace = True)

    ## Get coordinates for training data and set coord origin to (0,0)  
    bound_xmin = dfgrid.x.min() - 0.5* settings.xvoxsize
    bound_xmax = dfgrid.x.max() + 0.5* settings.xvoxsize
    bound_ymin = dfgrid.y.min() - 0.5* settings.yvoxsize
    bound_ymax = dfgrid.y.max() + 0.5* settings.yvoxsize

    dfgrid['x'] = dfgrid.x - bound_xmin
    dfgrid['y'] = dfgrid.y - bound_ymin

    # Define grid coordinates:
    points3D_train = np.asarray([dftrain.z.values, dftrain.y.values - bound_ymin, dftrain.x.values - bound_xmin ]).T

    # Define y target
    y_train = dftrain[settings.name_target].values

    # spatial uncertainty of coordinates:
    Xdelta_train = np.asarray([0.5 * dftrain.z_diff.values, dftrain.y.values * 0, dftrain.x.values * 0.]).T

    # Calculate predicted mean values of training data
    X_train = dftrain[settings.name_features].values
    y_train = dftrain[settings.name_target].values
    if mean_function == 'rf':
        # Estimate GP mean function with Random Forest
        rf_model = rf.rf_train(X_train, y_train)
        ypred_rf_train, ynoise_train, nrmse_rf_train = rf.rf_predict(X_train, rf_model, y_test = y_train)
        y_train_fmean = ypred_rf_train
    elif mean_function == 'blr':
        # Scale data
        Xs_train, ys_train, scale_params = blr.scale_data(X_train, y_train)
        scaler_x, scaler_y = scale_params
        # Train BLR
        blr_model = blr.blr_train(Xs_train, y_train)
        # Predict for X_test
        ypred_blr_train, ypred_std_blr_train, nrmse_blr_train = blr.blr_predict(Xs_train, blr_model, y_test = y_train)
        ypred_blr_train = ypred_blr_train.flatten()
        y_train_fmean = ypred_blr_train
        ynoise_train = ypred_std_blr_train

    # Subtract mean function from training data 
    y_train -= y_train_fmean

    if not calc_mean_only:
        # optimise GP hyperparameters 
        # Use mean of X uncertainity for optimizing since otherwise too many local minima
        print('Optimizing GP hyperparameters...')
        Xdelta_mean = Xdelta_train * 0 + np.nanmean(Xdelta_train,axis=0)
        opt_params, opt_logl = gp.optimize_gp_3D(points3D_train, y_train, ynoise_train, 
            xymin = settings.xyvoxsize, 
            zmin = settings.zvoxsize,  
            Xdelta = Xdelta_mean)
        params_gp = opt_params

    # Set extent of prediction grid
    extent = (0,bound_xmax - bound_xmin, 0, bound_ymax - bound_ymin)

    # Set output path for figures for each depth or temporal slice
    outpath_fig = os.path.join(settings.outpath, 'Figures_zslices/')
    os.makedirs(outpath_fig, exist_ok = True)	

    # Need to make predictions in mini-batches and then map results with coordinates to grid with ndimage.map_coordinates
    batchsize = 500
    #def chunker(df, batchsize):
    #	return (df[pos:pos + batchsize] for pos in np.arange(0, len(df), batchsize))
    dfgrid = dfgrid.reset_index()
    dfgrid['ibatch'] = dfgrid.index // batchsize
        
    #nbatch = dfgrid['ibatch'].max()
    ixrange_batch = dfgrid['ibatch'].unique()
    nbatch = len(ixrange_batch)
    print("Number of mini-batches per depth or time slice: ", nbatch)
    mu_res = np.zeros(len(dfgrid))
    std_res = np.zeros(len(dfgrid))
    coord_x = np.zeros(len(dfgrid))
    coord_y = np.zeros(len(dfgrid))
    ix = np.arange(len(dfgrid))

    xspace = np.arange(dfgrid['x'].min(), dfgrid['x'].max(), settings.xvoxsize)
    yspace = np.arange(dfgrid['y'].min(), dfgrid['y'].max(), settings.yvoxsize)
    if (len(settings.list_z_pred) > 0) & (settings.list_z_pred is not None) &  (settings.list_z_pred != 'None'):
        zspace = np.asarray(settings.list_z_pred)
    else:
        zspace = np.arange(settings.zvoxsize, settings.zmax + settings.zvoxsize, settings.zvoxsize)
        if settings.axistype == 'vertical':
            print('Calculating for depths at: ', zspace)
        elif settings.axistype == 'temporal':
            print('Calculating for time slices at: ', zspace)
    grid_x, grid_y = np.meshgrid(xspace, yspace)
    mu_3d = np.zeros((len(xspace), len(yspace), len(zspace)))
    std_3d = np.zeros((len(xspace), len(yspace), len(zspace)))
    gp_train_flag = 0 # need to be computed only first time
    # Slice in blocks for prediction calculating per 30 km x 1cm
    for i in range(len(zspace)):
        # predict for each depth  or temporal slice
        if settings.axistype == 'vertical':
            print('Computing slices at depth: ' + str(np.round(100 * zspace[i])) + 'cm')
        elif settings.axistype == 'temporal':
            print('Computing slices at time: ' + str(np.round(100 * zspace[i])))
        ix_start = 0
        for j in tqdm(ixrange_batch):
            dftest = dfgrid[dfgrid.ibatch == j].copy()
            #Set maximum number of evaluation points to 500 
            while len(dftest) > 500:
                # if larger than 500, select only subset of sample points that are regular spaced
                # select only every second value, this reduces size to 1/2
                dftest = dftest.sort_values(['y', 'x'], ascending = [True, True])
                dftest = dftest.iloc[::2, :]
            dftest['z'] = zspace[i]
            ysel = dftest.y.values
            xsel = dftest.x.values
            zsel = dftest.z.values
            points3D_pred = np.asarray([zsel, ysel, xsel]).T
            
            # Calculate mean function for prediction
            if mean_function == 'rf':
                X_test = dftest[settings.name_features].values
                ypred_rf, ynoise_pred, _ = rf.rf_predict(X_test, rf_model)
                y_pred_zmean = ypred_rf
            elif mean_function == 'blr':
                X_test = dftest[settings.name_features].values
                Xs_test = scaler_x.transform(X_test)
                ypred_blr, ypred_std_blr, _ = blr.blr_predict(Xs_test, blr_model)
                y_pred_zmean = ypred_blr
                ynoise_pred = ypred_std_blr

            # GP Prediction:
            if not calc_mean_only:
                if gp_train_flag == 0:
                    # Need to calculate matrix gp_train only once, then used subsequently for all other predictions
                    ypred, ystd, logl, gp_train = gp.train_predict_3D(points3D_train, points3D_pred, y_train, ynoise_train, params_gp, 
                        Ynoise_pred = ynoise_pred, Xdelta = Xdelta_train)
                    gp_train_flag = 1
                else:
                    ypred, ystd = gp.predict_3D(points3D_train, points3D_pred, gp_train, params_gp, Ynoise_pred = ynoise_pred, Xdelta = Xdelta_train)
            else:
                ypred = y_pred_zmean
                ystd = ynoise_pred

            # Alterantive: Combine noise of GP and mean function for prediction (already in coavraice function):
            #ystd = np.sqrt(ystd**2 + ynoise_pred**2)	

            # Save results in 3D array
            ix_end = ix_start + len(ypred)
            if not calc_mean_only:
                mu_res[ix_start : ix_end] = ypred + y_pred_zmean #.reshape(len(xspace), len(yspace))
                std_res[ix_start : ix_end] = ystd #.reshape(len(xspace), len(yspace))
            else: 
                mu_res[ix_start : ix_end] = ypred #.reshape(len(xspace), len(yspace))
                std_res[ix_start : ix_end] = ystd #.reshape(len(xspace), len(yspace))
            if i ==0:
                coord_x[ix_start : ix_end] = np.round(dftest.x.values,2)
                coord_y[ix_start : ix_end] = np.round(dftest.y.values,2)
            ix_start = ix_end


        # Save all data for the depth or temporal layer
        print("saving data and generating plots...")
        mu_img = np.zeros_like(grid_x.flatten()) * np.nan
        std_img = np.zeros_like(grid_x.flatten()) * np.nan
        xgridflat = grid_x.flatten()
        ygridflat = grid_y.flatten()

        # Calculate nearest neighbor
        xygridflat = np.asarray([xgridflat, ygridflat]).T
        coord_xy = np.asarray([coord_x, coord_y]).T
        mu_img, std_img = align_nearest_neighbor(xygridflat, coord_xy, [mu_res, std_res], max_dist = 0.5 * settings.xvoxsize)

        if settings.axistype == 'vertical':
            np.savetxt(os.path.join(outpath_fig, 'Pred_' + settings.name_target + '_z' + str("{:03d}".format(int(np.round(100 * zspace[i])))) + 'cm.txt'), np.round(mu_img,2), delimiter=',')
            np.savetxt(os.path.join(outpath_fig, 'Pred_Stddev_' + settings.name_target + '_z' + str("{:03d}".format(int(np.round(100 * zspace[i])))) + 'cm.txt'), np.round(std_img,3), delimiter=',')
        elif settings.axistype == 'temporal':
            np.savetxt(os.path.join(outpath_fig, 'Pred_' + settings.name_target + '_t' + str("{:03d}".format(int(np.round(100 * zspace[i])))) + '.txt'), np.round(mu_img,2), delimiter=',')
            np.savetxt(os.path.join(outpath_fig, 'Pred_Stddev_' + settings.name_target + '_t' + str("{:03d}".format(int(np.round(100 * zspace[i])))) + '.txt'), np.round(std_img,3), delimiter=',')
        if i == 0:
            # Create coordinate array of x and y
            np.savetxt(os.path.join(outpath_fig, 'Pred_' + settings.name_target + '_coord_x.txt'), coord_x, delimiter=',')
            np.savetxt(os.path.join(outpath_fig, 'Pred_' + settings.name_target + '_coord_y.txt'), coord_y, delimiter=',')
        
        mu_img = mu_img.reshape(grid_x.shape)
        std_img = std_img.reshape(grid_x.shape)
        mu_3d[:,:,i] = mu_img.T
        std_3d[:,:,i] = std_img.T

        print("Creating plots...")
        mu_3d_trim = mu_3d[:,:,i].copy()
        mu_3d_trim_max = np.percentile(mu_3d_trim[~np.isnan(mu_3d_trim)], 99.5)
        mu_3d_trim[mu_3d_trim > mu_3d_trim_max] = mu_3d_trim_max
        mu_3d_trim[mu_3d_trim < 0] = 0
        plt.figure(figsize = (8,8))
        plt.subplot(2, 1, 1)
        plt.imshow(mu_3d_trim.T,origin='lower', aspect = 'equal', extent = extent, cmap = colormap_pred)
        #plt.imshow(ystd.reshape(len(yspace),len(xspace)),origin='lower', aspect = 'equal', extent = extent) 
        #plt.scatter(points3D_train[:,2],points3D_train[:,1], edgecolors = 'k',facecolors='none')
        if settings.axistype == 'vertical':
            plt.title(settings.name_target + ' Depth ' + str(np.round(100 * zspace[i])) + 'cm')
        elif settings.axistype == 'temporal':
            plt.title(settings.name_target + ' Time ' + str(np.round(100 * zspace[i])))
        plt.ylabel('Northing [meters]')
        plt.colorbar()
        plt.subplot(2, 1, 2)
        std_3d_trim = std_3d[:,:,i].copy()
        std_3d_trim_max = np.percentile(std_3d_trim[~np.isnan(std_3d_trim)], 99.5)
        std_3d_trim[std_3d_trim > std_3d_trim_max] = std_3d_trim_max
        std_3d_trim[std_3d_trim < 0] = 0
        plt.imshow(std_3d_trim.T,origin='lower', aspect = 'equal', extent = extent, cmap = colormap_pred_std)
        #plt.imshow(ystd.reshape(len(yspace),len(xspace)),origin='lower', aspect = 'equal', extent = extent) 
        #plt.scatter(points3D_train[:,2],points3D_train[:,1], edgecolors = 'k',facecolors='none')
        if settings.axistype == 'vertical':
            plt.title('Std Dev ' + settings.name_target + ' Depth ' + str(np.round(100 * zspace[i])) + 'cm')
        if settings.axistype == 'temporal':
            plt.title('Std Dev ' + settings.name_target + ' Time ' + str(np.round(100 * zspace[i])))
        plt.colorbar()
        plt.xlabel('Easting [meters]')
        plt.ylabel('Northing [meters]')
        plt.tight_layout()
        if settings.axistype == 'vertical':
            plt.savefig(os.path.join(outpath_fig, 'Pred_' + settings.name_target + '_z' + str("{:03d}".format(int(np.round(100 * zspace[i])))) + 'cm.png'), dpi=300)
        elif settings.axistype == 'temporal':
            plt.savefig(os.path.join(outpath_fig, 'Pred_' + settings.name_target + '_t' + str("{:03d}".format(int(np.round(100 * zspace[i])))) + '.png'), dpi=300)
        if _show:
            plt.show()
        plt.close('all')

        #Save also as geotiff
        if settings.axistype == 'vertical':
            outfname_tif = os.path.join(outpath_fig, 'Pred_' + settings.name_target + '_z' + str("{:03d}".format(int(np.round(100 * zspace[i])))) + 'cm.tif')
            outfname_tif_std = os.path.join(outpath_fig, 'Std_' + settings.name_target + '_z' + str("{:03d}".format(int(np.round(100 * zspace[i])))) + 'cm.tif')
        elif settings.axistype == 'temporal':
            outfname_tif = os.path.join(outpath_fig, 'Pred_' + settings.name_target + '_t' + str("{:03d}".format(int(np.round(100 * zspace[i])))) + '.tif')
            outfname_tif_std = os.path.join(outpath_fig, 'Std_' + settings.name_target + '_t' + str("{:03d}".format(int(np.round(100 * zspace[i])))) + '.tif')

        print('Saving results as geo tif...')
        tif_ok = array2geotiff(mu_img, [bound_xmin + 0.5 * settings.xvoxsize, bound_ymin + 0.5 * settings.yvoxsize], [settings.xvoxsize,settings.yvoxsize], outfname_tif, settings.project_crs)
        tif2_ok = array2geotiff(std_img, [bound_xmin + 0.5 * settings.xvoxsize, bound_ymin + 0.5 * settings.yvoxsize], [settings.xvoxsize,settings.yvoxsize], outfname_tif_std, settings.project_crs)


    # Clip stddev for images
    mu_3d_mean = mu_3d.mean(axis = 2).T
    mu_3d_mean_max = np.percentile(mu_3d_mean,99.5)
    mu_3d_mean_trim = mu_3d_mean.copy()
    mu_3d_mean_trim[mu_3d_mean > mu_3d_mean_max] = mu_3d_mean_max
    mu_3d_mean_trim[mu_3d_mean < 0] = 0
    std_3d_trim = std_3d.copy()
    std_3d_trim_max = np.percentile(std_3d_trim[~np.isnan(std_3d_trim)],99.5)
    std_3d_trim[std_3d_trim > std_3d_trim_max] = std_3d_trim_max
    std_3d_trim[std_3d_trim < 0] = 0

    # Create Result Plot of mean with locations
    plt.figure(figsize = (8,8))
    plt.subplot(2, 1, 1)
    plt.imshow(mu_3d_mean_trim,origin='lower', aspect = 'equal', extent = extent, cmap = colormap_pred)
    plt.colorbar()
    #plt.imshow(ystd.reshape(len(yspace),len(xspace)),origin='lower', aspect = 'equal', extent = extent) 
    plt.scatter(points3D_train[:,2],points3D_train[:,1], edgecolors = 'k',facecolors='none')
    plt.title(settings.name_target + ' Mean')
    plt.ylabel('Northing [meters]')

    plt.subplot(2, 1, 2)
    plt.imshow(std_3d_trim.mean(axis = 2).T,origin='lower', aspect = 'equal', extent = extent, cmap = colormap_pred_std)
    plt.colorbar()
    #plt.imshow(ystd.reshape(len(yspace),len(xspace)),origin='lower', aspect = 'equal', extent = extent) 
    plt.scatter(points3D_train[:,2],points3D_train[:,1], edgecolors = 'k',facecolors='none')
    plt.title('Std Dev ' + settings.name_target + ' Mean')
    plt.xlabel('Easting [meters]')
    plt.ylabel('Northing [meters]')
    plt.tight_layout()
    plt.savefig(os.path.join(outpath_fig, 'Pred_' + settings.name_target + '_mean.png'), dpi=300)
    if _show:
        plt.show()
    plt.close('all')

    # Create Result Plot with data colors
    plt.figure(figsize = (8,8))
    plt.subplot(2, 1, 1)
    plt.imshow(mu_3d_mean_trim,origin='lower', aspect = 'equal', extent = extent, cmap = colormap_pred)
    plt.colorbar()
    #plt.imshow(ystd.reshape(len(yspace),len(xspace)),origin='lower', aspect = 'equal', extent = extent) 
    plt.scatter(points3D_train[:,2],points3D_train[:,1], c = dftrain[settings.name_target].values, alpha =0.3, edgecolors = 'k')

    if settings.axistype == 'vertical':
        plt.title(settings.name_target + ' Depth Mean')
    elif settings.axistype == 'temporal':
        plt.title(settings.name_target + ' Time Mean')
    plt.ylabel('Northing [meters]')
    plt.subplot(2, 1, 2)
    plt.imshow(std_3d_trim.mean(axis = 2).T,origin='lower', aspect = 'equal', extent = extent, cmap = colormap_pred_std)
    #plt.imshow(np.sqrt((std_3d.mean(axis = 2).T)**2 + params_gp[1]**2 *  ytrain.std()**2),origin='lower', aspect = 'equal', extent = extent)
    plt.colorbar()
    #plt.imshow(ystd.reshape(len(yspace),len(xspace)),origin='lower', aspect = 'equal', extent = extent) 
    plt.scatter(points3D_train[:,2],points3D_train[:,1], edgecolors = 'k',facecolors='none')
    plt.title('Std Dev ' + settings.name_target + ' Mean')
    plt.xlabel('Easting [meters]')
    plt.ylabel('Northing [meters]')
    plt.tight_layout()
    plt.savefig(os.path.join(outpath_fig, 'Pred_' + settings.name_target + '_mean2.png'), dpi=300)
    if _show:
        plt.show()
    plt.close('all')

    return mu_3d, std_3d


######### Polygon prediction #########
def model_polygons(settings):
    """
    Predict soil properties and uncertainties for polygons.
    All output is saved in output directory as specified in settings.

    Note: This is an experimental function and is not tested yet.

    Parameters
    ----------
        settings : settings namespace
    """

    from preprocessing import preprocess_grid_poly

    if (settings.model_function == 'blr') | (settings.model_function == 'rf'):
        # only mean function model
        calc_mean_only = True
    else:
        calc_mean_only = False
    if (settings.model_function == 'blr-gp') | (settings.model_function == 'blr'):
        mean_function = 'blr'
    if (settings.model_function == 'rf-gp') | (settings.model_function == 'rf'):
        mean_function = 'rf'

    # set conditional settings
    if calc_mean_only:
        settings.optimize_GP = False

    # currently assuming resolution is the same in x and y direction
    settings.xvoxsize = settings.yvoxsize = settings.xyvoxsize

    # check if outpath exists, if not create direcory
    os.makedirs(settings.outpath, exist_ok = True)

    # Intialise output info file:
    print2('init')
    print2(f'--- Parameter Settings ---')
    print2(f'Model Function: {settings.model_function}')
    print2(f'Target Name: {settings.name_target}')
    print2(f'Prediction geometry: Polygon')
    print2(f'--------------------------')

    print('Reading in data...')
    # Read in data for model training:
    dftrain = pd.read_csv(os.path.join(settings.inpath, settings.infname))

    # Rename x and y coordinates of input data
    if settings.colname_xcoord != 'x':
        dftrain.rename(columns={settings.colname_xcoord: 'x'}, inplace = True)
    if settings.colname_ycoord != 'y':
        dftrain.rename(columns={settings.colname_ycoord: 'y'}, inplace = True)
    if settings.colname_zcoord != 'z':
        dftrain.rename(columns={settings.colname_zcoord: 'z'}, inplace = True)
        name_features_grid = settings.name_features.copy()
        if settings.colname_zcoord in settings.name_features:
            settings.name_features.remove(settings.colname_zcoord)
            name_features_grid.remove(settings.colname_zcoord)
            settings.name_features.append('z')
            name_features = settings.name_features

    # Convert z and z_diff to meters if in cm (>10):
    if (dftrain['z'].max() > 10):
        dftrain['z'] = dftrain['z'] / 100
        if 'z_diff' in dftrain.columns:
            dftrain['z_diff'] = dftrain['z_diff'] / 100
  
    # Select data between zmin and zmax
    dftrain = dftrain[(dftrain['z'] >= settings.zmin) & (dftrain['z'] <= settings.zmax)]

    # check if z_diff is in dftrain
    if 'z_diff' not in dftrain.columns:
        dftrain['z_diff'] = 0.0

    # Read in polygon data:
    dfgrid, dfpoly, name_features_grid2 = preprocess_grid_poly(settings.inpath, settings.gridname, settings.polyname, 
        name_features = name_features_grid,  grid_crs = settings.project_crs, 
        grid_colname_Easting = settings.colname_xcoord, grid_colname_Northing = settings.colname_ycoord)

    # Rename x and y coordinates of input data
    if settings.colname_xcoord != 'x':
        dfgrid.rename(columns={settings.colname_xcoord: 'x'}, inplace = True)
    if settings.colname_ycoord != 'y':
        dfgrid.rename(columns={settings.colname_ycoord: 'y'}, inplace = True)
    if settings.colname_zcoord != 'z':
        dfgrid.rename(columns={settings.colname_zcoord: 'z'}, inplace = True)

    ## Get coordinates for training data and set coord origin to (0,0)  
    bound_xmin = dfgrid.x.min() - 0.5* settings.xvoxsize
    bound_xmax = dfgrid.x.max() + 0.5* settings.xvoxsize
    bound_ymin = dfgrid.y.min() - 0.5* settings.yvoxsize
    bound_ymax = dfgrid.y.max() + 0.5* settings.yvoxsize

    dfgrid['x'] = dfgrid.x - bound_xmin
    dfgrid['y'] = dfgrid.y - bound_ymin
	
    # Define grid coordinates:
    points3D_train = np.asarray([dftrain.z.values, dftrain.y.values - bound_ymin, dftrain.x.values - bound_xmin ]).T

    # Define y target
    y_train = dftrain[settings.name_target].values

    # spatial uncertainty of coordinates:
    if 'z_diff' in list(dftrain):
        Xdelta_train = np.asarray([0.5 * dftrain.z_diff.values, dftrain.y.values * 0, dftrain.x.values * 0.]).T
    else:
        Xdelta_train = np.asarray([0 * dftrain.z.values, dftrain.y.values * 0, dftrain.x.values * 0.]).T

    # Calculate predicted mean values of training data
    X_train = dftrain[settings.name_features].values
    y_train = dftrain[settings.name_target].values
    if mean_function == 'rf':
        # Estimate GP mean function with Random Forest
        rf_model = rf.rf_train(X_train, y_train)
        ypred_rf_train, ynoise_train, nrmse_rf_train = rf.rf_predict(X_train, rf_model, y_test = y_train)
        y_train_fmean = ypred_rf_train
    elif mean_function == 'blr':
        # Scale data
        Xs_train, ys_train, scale_params = blr.scale_data(X_train, y_train)
        scaler_x, scaler_y = scale_params
        # Train BLR
        blr_model = blr.blr_train(Xs_train, y_train)
        # Predict for X_test
        ypred_blr_train, ypred_std_blr_train, nrmse_blr_train = blr.blr_predict(Xs_train, blr_model, y_test = y_train)
        ypred_blr_train = ypred_blr_train.flatten()
        y_train_fmean = ypred_blr_train
        ynoise_train = ypred_std_blr_train

    # Subtract mean function from training data 
    y_train -= y_train_fmean

    if not calc_mean_only:
        # optimise GP hyperparameters 
        # Use mean of X uncertainity for optimizing since otherwise too many local minima
        print('Optimizing GP hyperparameters...')
        Xdelta_mean = Xdelta_train * 0 + np.nanmean(Xdelta_train,axis=0)
        opt_params, opt_logl = gp.optimize_gp_3D(points3D_train, y_train, ynoise_train, 
            xymin = settings.xyvoxsize, 
            zmin = settings.zvoxsize,  
            Xdelta = Xdelta_mean)
        params_gp = opt_params

    # Set extent of prediction grid
    extent = (0,bound_xmax - bound_xmin, 0, bound_ymax - bound_ymin)

    # Set output path for figures for each depth or time slice
    outpath_fig = os.path.join(settings.outpath, 'Figures_zslices/')
    os.makedirs(outpath_fig, exist_ok = True)	

    dfout_poly =  dfgrid[['ibatch']].copy()
    dfout_poly.drop_duplicates(inplace=True, ignore_index=True)
        
    #nbatch = dfgrid['ibatch'].max()
    ixrange_batch = dfgrid['ibatch'].unique()
    nbatch = len(ixrange_batch)
    print("Number of mini-batches per depth or time slice: ", nbatch)
    mu_res = np.zeros(len(dfgrid))
    std_res = np.zeros(len(dfgrid))
    coord_x = np.zeros(len(dfgrid))
    coord_y = np.zeros(len(dfgrid))
    ix = np.arange(len(dfgrid))

    xspace = np.arange(dfgrid['x'].min(), dfgrid['x'].max(), settings.xvoxsize)
    yspace = np.arange(dfgrid['y'].min(), dfgrid['y'].max(), settings.yvoxsize)
    if (len(settings.list_z_pred) > 0) & (settings.list_z_pred is not None) &  (settings.list_z_pred != 'None'):
        zspace = np.asarray(settings.list_z_pred)
    else:
        zspace = np.arange(settings.zvoxsize, settings.zmax + settings.zvoxsize, settings.zvoxsize)
        if settings.axistype == 'vertical':
            print('Calculating for depths at: ', zspace)
        elif settings.axistype == 'temporal':
            print('Calculating for time slices at: ', zspace)
    grid_x, grid_y = np.meshgrid(xspace, yspace)
    gp_train_flag = 0 # need to be computed only first time
    # Slice in blocks for prediction calculating per 30 km x 1cm
    for i in range(len(zspace)):
        # predict for each depth  or time slice
        if settings.axistype == 'vertical':
            print('Computing slices at depth: ' + str(np.round(100 * zspace[i])) + 'cm')
        elif settings.axistype == 'temporal':
            print('Computing slices at time: ' + str(np.round(100 * zspace[i])))
        ix_start = 0
        dfout_poly['Mean'] = np.nan
        dfout_poly['Std'] = np.nan
        for j in tqdm(ixrange_batch):
            dftest = dfgrid[dfgrid.ibatch == j].copy()
            #Set maximum number of evaluation points to 500 
            while len(dftest) > 500:
                # if larger than 500, select only subset of sample points that are regular spaced
                # select only every second value, this reduces size to 1/2
                dftest = dftest.sort_values(['y', 'x'], ascending = [True, True])
                dftest = dftest.iloc[::2, :]
            dftest['z'] = zspace[i]
            ysel = dftest.y.values
            xsel = dftest.x.values
            zsel = dftest.z.values
            points3D_pred = np.asarray([zsel, ysel, xsel]).T
            
            # Calculate mean function for prediction
            if mean_function == 'rf':
                X_test = dftest[settings.name_features].values
                ypred_rf, ynoise_pred, _ = rf.rf_predict(X_test, rf_model)
                y_pred_zmean = ypred_rf
            elif mean_function == 'blr':
                X_test = dftest[settings.name_features].values
                Xs_test = scaler_x.transform(X_test)
                ypred_blr, ypred_std_blr, _ = blr.blr_predict(Xs_test, blr_model)
                y_pred_zmean = ypred_blr
                ynoise_pred = ypred_std_blr

            # GP Prediction:
            if not calc_mean_only:
                if gp_train_flag == 0:
                    # Need to calculate matrix gp_train only once, then used subsequently for all other predictions
                    ypred, ystd, logl, gp_train, covar = gp.train_predict_3D(points3D_train, points3D_pred, y_train, ynoise_train, params_gp, 
                        Ynoise_pred = ynoise_pred, Xdelta = Xdelta_train, out_covar = True)
                    gp_train_flag = 1
                else:
                    ypred, ystd, covar = gp.predict_3D(points3D_train, points3D_pred, gp_train, params_gp, Ynoise_pred = ynoise_pred, 
                        Xdelta = Xdelta_train, out_covar = True)
            else:
                ypred = y_pred_zmean
                ystd = ynoise_pred

            # Now calculate mean and standard deviation for polygon area
            # Need to calculate weighted average from covar and ypred
            if not calc_mean_only:
                ypred_poly, ystd_poly = averagestats(ypred + y_pred_zmean, covar)
            else:
                ypred_poly, ystd_poly = averagestats(ypred, covar)
            dfout_poly.loc[dfout_poly['ibatch'] == j, 'Mean'] = ypred_poly
            dfout_poly.loc[dfout_poly['ibatch'] == j, 'Std'] = ystd_poly

        # Save all data for the slice
        dfpoly_z = dfpoly.merge(dfout_poly, how = 'left', on = 'ibatch')
        # Save results with polygon shape as Geopackage (can e.g. visualised in QGIS)
        if settings.axistype == 'vertical':
            dfpoly_z.to_file(os.path.join(outpath_fig, 'Prediction_poly_' + settings.name_target + '_z' + str("{:03d}".format(int(np.round(100 * zspace[i])))) + 'cm.gpkg'), driver='GPKG')
        elif settings.axistype == 'temporal':
            dfpoly_z.to_file(os.path.join(outpath_fig, 'Prediction_poly_' + settings.name_target + '_t' + str("{:03d}".format(int(np.round(100 * zspace[i])))) + '.gpkg'), driver='GPKG')
        # make some plots with geopandas
        print("Plotting polygon map ...")
        fig, (ax1, ax2) = plt.subplots(ncols = 1, nrows=2, sharex=True, sharey=True, figsize = (10,10))
        dfpoly_z.plot(column='Mean', legend=True, ax = ax1, cmap = colormap_pred)#, legend_kwds={'label': "",'orientation': "vertical"}))
        if settings.axistype == 'vertical':
            ax1.title.set_text('Mean ' + settings.name_target + ' Depth ' + str(np.round(100 * zspace[i])) + 'cm')
        elif settings.axistype == 'temporal':
            ax1.title.set_text('Mean ' + settings.name_target + ' Time ' + str(np.round(100 * zspace[i])))
        ax1.set_ylabel('Northing [meters]')
        #plt.xlabel('Easting [meters]')
        #plt.savefig(os.path.join(outpath_fig, 'Pred_Mean_Poly_' + name_target + '_z' + str("{:03d}".format(int(np.round(100 * zspace[i])))) + 'cm.png'), dpi=300)
        dfpoly_z.plot(column='Std', legend=True,  ax = ax2, cmap = colormap_pred_std)#, legend_kwds={'label': "",'orientation': "vertical"}))
        if settings.axistype == 'vertical':
            ax2.title.set_text('Std Dev ' + settings.name_target + ' Depth ' + str(np.round(100 * zspace[i])) + 'cm')
        elif settings.axistype == 'temporal':
            ax2.title.set_text('Std Dev ' + settings.name_target + ' Time ' + str(np.round(100 * zspace[i])))
        ax2.set_xlabel('Easting [meters]')
        ax2.set_ylabel('Northing [meters]')
        plt.tight_layout()
        if settings.axistype == 'vertical':
            plt.savefig(os.path.join(outpath_fig, 'Pred_Poly_' + settings.name_target + '_z' + str("{:03d}".format(int(np.round(100 * zspace[i])))) + 'cm.png'), dpi=300)
        elif settings.axistype == 'temporal':
            plt.savefig(os.path.join(outpath_fig, 'Pred_Poly_' + settings.name_target + '_t' + str("{:03d}".format(int(np.round(100 * zspace[i])))) + '.png'), dpi=300)
        if _show:
            plt.show()  
        plt.close('all')



######################### Main Function ############################
def main(fname_settings):	
    """
    Main function for running soilmodel predictions.

    Input:
        fname_settings: path and filename to settings file
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
        settings.zmin = settings.tmin
        settings.zmax =  settings.tmax
        settings.list_z_pred = settings.list_t_pred 
        settings.zblocksize = settings.tblocksize

    if settings.integrate_polygon:
        model_polygons(settings)
    else:
        if settings.integrate_block:
            mu_3d, std_3d = model_blocks(settings)
        else:
            mu_3d, std_3d = model_points(settings)
        print("Prediction Mean, Median, Std, 25Perc, 75Perc:", np.round([np.nanmean(mu_3d), np.median(mu_3d[~np.isnan(mu_3d)]), 
            np.nanstd(mu_3d), np.percentile(mu_3d[~np.isnan(mu_3d)],25), np.percentile(mu_3d[~np.isnan(mu_3d)],75)] 
            ,3))
        print("Uncertainty Mean, Median, Std, 25Perc, 75Perc:", np.round([np.nanmean(std_3d), np.median(std_3d[~np.isnan(std_3d)]),
            np.nanstd(std_3d), np.percentile(std_3d[~np.isnan(std_3d)],25), np.percentile(std_3d[~np.isnan(std_3d)],75)],3))
    print('')
    print('Prediction finished')
    print(f'All results are saved in output directory {settings.outpath}')


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Prediction model for machine learning on soil data.')
    parser.add_argument('-s', '--settings', type=str, required=False,
                        help='Path and filename of settings file.',
                        default = _fname_settings)
    args = parser.parse_args()

    # Run main function
    main(args.settings)