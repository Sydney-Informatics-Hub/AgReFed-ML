"""
Machine Learning model for Change prediction and uncertainties using Gaussian Processes.

Current models implemented:
- Bayesian linear regression (BLR) 
- Random forest (RF)
- Gaussian Process with bayesian linear regression (BLR) as mean function and sparse spatial covariance function
- Gaussian Process with random forest (RF) regression as mean function and sparse spatial covariance function

User settings, such as input/output paths and all other options, are set in the settings file 
(Default filename: settings_soilmodel_predict.yaml) 
Alternatively, the settings file can be specified as a command line argument with: 
'-s', or '--settings' followed by PATH-TO-FILE/FILENAME.yaml 
(e.g. python featureimportance.py -s settings_featureimportance.yaml).

This package is part of the machine learning project developed for the Agricultural Research Federation (AgReFed).

Copyright Sydney Informatics Hub (SIH), The University of Sydney

This open-source software is released under the LGPL-3.0 License.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import os
import sys
from scipy.special import erf
from scipy.interpolate import interp1d, griddata
import matplotlib.pyplot as plt
#import pyvista as pv # helper module for the Visualization Toolkit (VTK)
import subprocess
from sklearn.model_selection import train_test_split 
# Save and load trained models and scalers:
import pickle
import json
import yaml
import argparse
from types import SimpleNamespace  
from tqdm import tqdm

# Custom local libraries:
from utils import array2geotiff, align_nearest_neighbor, print2, truncate_data
from sigmastats import averagestats, calc_change
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

######### Change model #########
def model_change(settings):
    """
    Predict soil properties and uncertainties for two dates and temporal covariance.
    The predicted uncertainty takes into account spatial covariance between the dates

    All output is saved in output directory as specified in settings.

    Parameters
    ----------
    settings : settings namespace

    Return
    ------
    mu_3d: prediction maps for two dates
    std_3d: standard deviation maps for two dates
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

    # Check that only two dates selected for change prediction
    assert len(settings.list_t_pred) == 2, 'length of list for setting list_t_pred must be two'

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
    settings.name_features.append('z')

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

    if (len(settings.list_z_pred) ==  2):
        zblock = np.asarray(settings.list_z_pred)
    else:
        raise ValueError('list_z_pred must be a list of length 2')
    block_x, block_y = np.meshgrid(xblock, yblock)
    block_shape = block_x.shape
    block_x = block_x.flatten()
    block_y = block_y.flatten()
    mu_3d = np.zeros((len(xblock), len(yblock), len(zblock)))
    std_3d = np.zeros((len(xblock), len(yblock), len(zblock)))
    mu_block1 = np.zeros_like(block_x)
    std_block1 = np.zeros_like(block_x)
    mu_block2 = np.zeros_like(block_x)
    std_block2 = np.zeros_like(block_x)
    ydelta_block = np.zeros_like(block_x)
    ydelta_std_block = np.zeros_like(block_x)
    # Set initial optimisation of hyperparamter to True
    gp_train_flag = True

    # predict for both temporal slices and their covariance
    print('Computing block average and time change ... ')
    #zrange = np.arange(zblock[i] - 0.5 * settings.zblocksize, zblock[i] + 0.5 * settings.zblocksize + settings.zvoxsize, settings.zvoxsize)
    ix_start = 0
    # Progressbar
    for j in tqdm(range(len(block_x.flatten()))):
        dftest = dfgrid[(dfgrid.x >= block_x[j] - 0.5 * settings.xblocksize) & (dfgrid.x <= block_x[j] + 0.5 * settings.xblocksize) &
            (dfgrid.y >= block_y[j] - 0.5 * settings.yblocksize) & (dfgrid.y <= block_y[j] + 0.5 * settings.yblocksize)].copy()
        if len(dftest) > 0:
            dfnew = dftest.copy()
            for z in zblock:
                if z == zblock[0]:
                    dftest['z'] = z 
                else:
                    dfnew['z'] = z
                    dftest = dftest.append(dfnew, ignore_index = True)									
            ysel = dftest.y.values
            xsel = dftest.x.values
            zsel = dftest.z.values
            #zz, yy = np.meshgrid(zrange, ysel)
            #zz, xx = np.meshgrid(zrange, xsel)
            #points3D_pred = np.asarray([zz.flatten(), yy.flatten(), xx.flatten()]).T
            points3D_pred = np.asarray([zsel, ysel, xsel]).T	# shape (nsamples, 3))	shape[:,0] = z
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
                    # ypred.shape = (nsample,); nsample 0:nsamepl/2 = t1, nsample/2:nsample = t2
                    # covar.shape = (nsample, nsample)
                else:
                    ypred, ystd, covar = gp.predict_3D(points3D_train, points3D_pred, gp_train, params_gp, Ynoise_pred = ynoise_pred, Xdelta = Xdelta_train, 
                        out_covar = True)
            else:
                ypred = y_pred_zmean
                ystd = ynoise_pred

            # Split prediction into t1 and t2
            nsplit = int(len(ypred)/2)
            ypred1 = ypred[:nsplit]
            ypred2 = ypred[nsplit:]
            ystd1 = ystd[:nsplit]
            ystd2 = ystd[nsplit:]
            covar1 = covar[:nsplit, :nsplit]
            covar2 = covar[nsplit:, nsplit:]
            y_pred_zmean1 = y_pred_zmean[:nsplit]
            y_pred_zmean2 = y_pred_zmean[nsplit:]
            ynoise_pred1 = ynoise_pred[:nsplit]
            ynoise_pred2 = ynoise_pred[nsplit:]

            #### Need to calculate weighted average from covar and ypred
            if not calc_mean_only:
                ypred_block1, ystd_block1 = averagestats(ypred1 + y_pred_zmean1, covar1)
                ypred_block2, ystd_block2 = averagestats(ypred2 + y_pred_zmean2, covar2)
            else:
                ypred_block1, ystd_block1 = averagestats(ypred1, covar1)
                ypred_block2, ystd_block2 = averagestats(ypred2, covar2)

            # Calculate Change 
            covar_delta = np.asarray([covar[i, i + nsplit] for i in range(len(ypred1))])
            ydelta, _ = calc_change(ypred1 +  y_pred_zmean1, ypred2 +  y_pred_zmean2, ystd1**2, ystd2**2, covar_delta)
            # Calculate combined covariance for delta t as sum of both covariances (t1, t2) and cross-covariance:
            covar_combined = covar1 + covar2 - 2*covar[:nsplit, nsplit:]
            # Calculate over spatial block:
            ydelta_block[j], ydelta_std_block[j] = averagestats(ydelta, covar_combined)

            # Save results in block array
            mu_block1[j] = ypred_block1
            std_block1[j] = ystd_block1
            mu_block2[j] = ypred_block2
            std_block2[j] = ystd_block2

        # Set blocks where there is no data to nan
        else:
            mu_block1[j] = np.nan
            std_block1[j] = np.nan
            mu_block2[j] = np.nan
            std_block2[j] = np.nan

    # map coordinate array to image and save in 3D
    mu_img1 = mu_block1.reshape(block_shape)
    std_img1 = std_block1.reshape(block_shape)
    mu_img2 = mu_block2.reshape(block_shape)
    std_img2 = std_block2.reshape(block_shape)
    delta_img = ydelta_block.reshape(block_shape)
    delta_std_img = ydelta_std_block.reshape(block_shape)

    # Create coordinate array of x and y
    np.savetxt(os.path.join(outpath_fig, 'Pred_' + settings.name_target + '_coord_x.txt'), block_x, delimiter=',')
    np.savetxt(os.path.join(outpath_fig, 'Pred_' + settings.name_target + '_coord_y.txt'), block_y, delimiter=',')

    mu_3d[:,:,0] = mu_img1.T
    std_3d[:,:,0] = std_img1.T
    mu_3d[:,:,1] = mu_img2.T
    std_3d[:,:,1] = std_img2.T

    for i in range(len(zblock)):
        # Create Result Plots
        mu_3d_trim = mu_3d[:,:,i].copy()
        mu_3d_trim_max = np.percentile(mu_3d_trim[~np.isnan(mu_3d_trim)], 99.5)
        mu_3d_trim[mu_3d_trim > mu_3d_trim_max] = mu_3d_trim_max
        mu_3d_trim[mu_3d_trim < 0] = 0
        plt.figure(figsize = (8,8))
        plt.subplot(2, 1, 1)
        plt.imshow(mu_3d_trim.T,origin='lower', aspect = 'equal', extent = extent, cmap = colormap_pred)
        plt.title(settings.name_target + ' Date ' + str(np.round(100 * zblock[i])))
        plt.ylabel('Northing [meters]')
        plt.colorbar()
        plt.subplot(2, 1, 2)
        std_3d_trim = std_3d[:,:,i].copy()
        std_3d_trim_max = np.percentile(std_3d_trim[~np.isnan(std_3d_trim)], 99.5)
        std_3d_trim[std_3d_trim > std_3d_trim_max] = std_3d_trim_max
        plt.imshow(std_3d_trim.T,origin='lower', aspect = 'equal', extent = extent, cmap = colormap_pred_std)
        plt.title(settings.name_target + ' Date ' + str(np.round(100 * zblock[i])))
        plt.colorbar()
        plt.xlabel('Easting [meters]')
        plt.ylabel('Northing [meters]')
        plt.tight_layout()
        plt.savefig(os.path.join(outpath_fig, 'Pred_' + settings.name_target + '_t' + str("{:03d}".format(int(np.round(100 * zblock[i])))) + '.png'), dpi=300)
        if _show:
            plt.show()
        plt.close('all')    

        #Save also as geotiff
        outfname_tif = os.path.join(outpath_fig, 'Pred_' + settings.name_target + '_t' + str("{:03d}".format(int(np.round(100 * zblock[i])))) + '.tif')
        outfname_tif_std = os.path.join(outpath_fig, 'Std_' + settings.name_target + '_t' + str("{:03d}".format(int(np.round(100 * zblock[i])))) + '.tif')
        #print('Saving results as geo tif...')
        tif_ok = array2geotiff(mu_3d[:,:,i].T, [bound_xmin + 0.5 * settings.xblocksize,bound_ymin + 0.5 * settings.yblocksize], [settings.xblocksize,settings.yblocksize], outfname_tif, settings.project_crs)
        tif_ok = array2geotiff(std_3d[:,:,i].T, [bound_xmin + 0.5 * settings.xblocksize,bound_ymin + 0.5 * settings.yblocksize], [settings.xblocksize,settings.yblocksize], outfname_tif_std, settings.project_crs)
    
    
    # Save delta image
    plt.figure(figsize = (8,8))
    plt.subplot(2, 1, 1)
    plt.imshow(delta_img,origin='lower', aspect = 'equal', extent = extent, cmap = colormap_pred)
    plt.title(settings.name_target + ' Change ' + str(settings.list_z_pred[1]) + '-' + str(settings.list_z_pred[0]))
    plt.ylabel('Northing [meters]')
    plt.colorbar()
    plt.subplot(2, 1, 2)
    plt.imshow(delta_std_img,origin='lower', aspect = 'equal', extent = extent, cmap = colormap_pred_std)
    plt.title(settings.name_target + ' Std Change ' + str(settings.list_z_pred[1]) + '-' + str(settings.list_z_pred[0]))
    plt.colorbar()
    plt.xlabel('Easting [meters]')
    plt.ylabel('Northing [meters]')
    plt.tight_layout()
    plt.savefig(os.path.join(outpath_fig, 'Pred_' + settings.name_target + '_dt_' + str(settings.list_z_pred[1]) + '-' + str(settings.list_z_pred[0]) +'.png'), dpi=300)
    if _show:
        plt.show()
    plt.close('all') 

    # save as tif
    outfname_dt_tif = os.path.join(outpath_fig, 'Pred_' + settings.name_target + '_dt.tif')
    outfname_dt_tif_std = os.path.join(outpath_fig, 'Std_' + settings.name_target + '_dt.tif')
    tif_ok = array2geotiff(delta_img, [bound_xmin + 0.5 * settings.xblocksize,bound_ymin + 0.5 * settings.yblocksize], [settings.xblocksize,settings.yblocksize], outfname_dt_tif, settings.project_crs)
    tif_ok = array2geotiff(delta_std_img, [bound_xmin + 0.5 * settings.xblocksize,bound_ymin + 0.5 * settings.yblocksize], [settings.xblocksize,settings.yblocksize], outfname_dt_tif_std, settings.project_crs)

    return mu_3d, std_3d




######################### Main Script ############################
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

    # Assume covariate grid file has same covariate names as training data
    settings.name_features_grid = settings.name_features.copy()

    # Add temporal component
    settings.colname_zcoord = settings.colname_tcoord
    settings.zmin = settings.tmin
    settings.zmax =  settings.tmax
    settings.list_z_pred = settings.list_t_pred 
    settings.zblocksize = settings.tblocksize

    mu_3d, std_3d = model_change(settings)

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