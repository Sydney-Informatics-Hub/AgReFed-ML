"""
Preprocessing functions for geospatial soil data 


Requirements:
- python>=3.9
- matplotlib>=3.5.1
- numpy>=1.22.0
- pandas>=1.3.5
- PyYAML>=6.0
- geopandas>=0.7.0

For more package details see conda environment file: environment.yaml

This package is part of the machine learning project developed for the Agricultural Research Federation (AgReFed).

Copyright 2022 Sebastian Haan, Sydney Informatics Hub (SIH), The University of Sydney

This open-source software is released under the AGPL-3.0 License.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import os
import sys
import argparse
import yaml
from types import SimpleNamespace  

# Default settings yaml file name:
_fname_settings = 'settings_preprocessing.yaml'


def preprocess(inpath, infname, outpath, outfname, name_target, name_features, 
name_ixval = None, zmin = None, zmax = None, gen_gpkg = False, categorical = None,
colname_depthmin = 'lower_depth', colname_depthmax = 'upper_depth'):
    """
    Converts input dataframe into more useable data and saves as new dataframe on disk
    Automatically converts categorical features to binary feature representations.

    Input:
        inpath: input path
        infname: input file name
        outpath: output path
        outfname: output file name
        name_target: String, Name of target for prediction (column names in infname)
        name_features: String or list of strings of covariate features (column names in infname)
        name_ixval: string, column name for index for X-fold crossvalidation
        zmin: in centimeters, if not None, data only larger than zmin will be selected
        zmax: in centimeters, if not None, data only smaller than zmax will be selected
        gen_gpkg: if True, data will be also saved as georeferenced geopackage
        categorical: name of categorical feature, converts categorical feature into additional features with binary coding

    Return:
        processed dataframe
        names of features with updated categorical (if none, then return original input name_features)
    """
    name_features2 = name_features.copy()
    df = pd.read_csv(os.path.join(path, infname))
    # Convert categorical features to binary features
    names_categorical = df.select_dtypes(include=['object']).columns.tolist()
    #names_categorical = df.select_dtypes(include=['categorical']).columns.tolist()

    if categorical is not None:
        # Add categories as features with binary coding
        cat_levels = df[categorical].unique()
        print('Adding following categories as binary features: ', cat_levels)
        name_features2.extend(cat_levels)
        for level in cat_levels:
            df[level] = 0
            df.loc[df[categorical].values == level, level] = 1
        name_features2.remove(categorical)
    # Calculate mid point and convert to cm
    df['z'] = 0.5 * (df[colname_depthmin] + df[colname_depthmax]) / 100.
    df['z_diff'] = (df[colname_depthmin] - df[colname_depthmax]) / 100.
    if ('Easting' in list(df)) &  ('Northing' in list(df)) & ('x' not in list(df)) & ('y' not in list(df)):
        df.rename(columns={"Easting": "x", "Northing": "y"}, inplace=True)
    if isinstance(name_features2, list):
        selcols = ['x', 'y', 'z', 'z_diff']
        selcols.extend(name_features2)
    else:
        selcols = ['Sample.ID','x', 'y', 'z', 'z_diff', name_features2]
    selcols.extend([name_target])
    dfout = df[selcols].copy()
    if zmin:
        dfout = dfout[dfout.z >= zmin]
    #Check for only finite values
    dfout.dropna(inplace = True)
    if name_ixval:
        ### Splitting in N-fold cross-validation
        if name_ixval not in list(dfout):
            print('Creating 10-fold cross-validation samples...')
            nsplit = 10
            dfout['site_id'] = dfout['x'].round(0).astype(int).map(str) + '_' + dfout['y'].round(0).astype(int).map(str)
            nunique = dfout['site_id'].unique()
            dfout['new_id'] = 0
            ix = 1
            for unique in nunique:
                dfout.loc[dfout['site_id'] == unique, 'new_id'] = ix
                ix += 1
            size = int(len(nunique)/nsplit)

            np.random.shuffle(nunique)
            dfout[name_ixval] = 0
            start = 0
            for i in range(nsplit - 1):
                stop = start + size
                sub = nunique[start:stop]
                dfout.loc[dfout['site_id'].isin(sub),name_ixval] = i + 1
                start = stop
            dfout.loc[dfout[name_ixval] == 0, name_ixval] = nsplit
        selcols.extend([name_ixval])
        #dfout = dfout[selcols].copy()
    dfout.to_csv(os.path.join(outpath, outfname), index = False)

    # save also as geopackage for visualisation  etc:
    # find zone from center point
    if gen_gpkg:
        if ('Latitude' in df) & ('Longitude' in df):
            lat_cen = df.Latitude.mean()
            lng_cen = df.Longitude.mean()
            zone, crs = find_zone(lat_cen, lng_cen)
            crs_epsg = 'EPSG:' + str(crs)
            gdf = gpd.GeoDataFrame(dfout.copy(), geometry=gpd.points_from_xy(dfout['x'], dfout['y']))
            gdf.crs = crs_epsg
        else:
            gdf = gpd.GeoDataFrame(dfout.copy(), geometry=gpd.points_from_xy(dfout['x'], dfout['y']))
            gdf.crs = project_crs
        #gdf.crs = {'init' :'epsg:4326'} # Depreciated with geopandas >= 0.7
        gdf.to_file(os.path.join(outpath, outfname + '.gpkg'), driver='GPKG')

    return dfout, name_features2


def preprocess_grid(inpath, infname, outpath, outfname, name_features, categorical = None):
    """
    Converts input dataframe into more useable data and saves as new dataframe on disk
    categorical variables are converted into binary features

    Input:
        inpath: input path
        infname: input file name
        outpath: output path
        outfname: output file name
        name_features: String or list of strings of covariate features (column names in infname)
        categorical: name of categorical feature, converts categorical feature into additional features with binary coding

    Return:
        processed dataframe
        names of features with updated categorical (if none, then return original input name_features)
    """
    print("Preprocessing grid covariates...")
    name_features2 = name_features.copy()
    df = pd.read_csv(os.path.join(path, infname))
    if categorical is not None:
        # Add categories as features with binary coding
        cat_levels = df[categorical].unique()
        print('Adding following categories as binary features: ', cat_levels)
        name_features2.extend(cat_levels)
        for level in cat_levels:
            df[level] = 0
            df.loc[df[categorical].values == level, level] = 1
        name_features2.remove(categorical)
    if ('Easting' in list(df)) &  ('Northing' in list(df)) & ('x' not in list(df)) & ('y' not in list(df)):
        df.rename(columns={"Easting": "x", "Northing": "y"}, inplace=True)
    if isinstance(name_features2, list): 
        selcols = ['x', 'y']
        selcols.extend(name_features2)
    else:
        selcols = ['x', 'y', name_features2]
    dfout = df[selcols].copy()
    dfout.to_csv(os.path.join(outpath, outfname), index = False)   
    return dfout, name_features2


def preprocess_grid_poly(path, infname_grid, infname_poly, name_features, 
	grid_crs, grid_colname_Easting = 'x', grid_colname_Northing = 'y',
	categorical = None):
	"""
	Converts input dataframe into more useable data and saves as new dataframe on disk.
	Grid points will be spatially joioned with polygons for prediction
	To Do add conversion to categorical

	Input:
        path: input path (same used for output)
        infname_grid: input file name of grid covariates
        infname_poly: input file name of polygon shape for predictions
        name_features: String or list of strings of covariate features (column names in infname)
        grid_crs: coordinate reference system of grid points, e.g. 'EPSG:28355'
        grid_colname_Easting: column name of Easting coordinate (or Longitude)
        grid_colname_Northing: coumn name for Northing coodinate (or Latitude)
        categorical: name of categorical feature, converts categorical feature into additional features with binary coding
	
    Return:
        processed dataframe
        names of features with updated categorical (if none, then return original input name_features)
	"""
	from shapely.geometry import Point
	print("Preprocessing grid covariates and joining with polygon geometry...")
	name_features2 = name_features.copy()
	df = pd.read_csv(os.path.join(path, infname_grid))
	if categorical is not None:
		# Add categories as features with binary coding
		cat_levels = df[categorical].unique()
		print('Adding following categories as binary features: ', cat_levels)
		name_features2.extend(cat_levels)
		for level in cat_levels:
			df[level] = 0
			df.loc[df[categorical].values == level, level] = 1
		name_features2.remove(categorical)
	# Convert grid covariates to geospatial point data
	geometry = [Point(xy) for xy in zip(df[grid_colname_Easting], df[grid_colname_Northing])]
	gdf = gpd.GeoDataFrame(df, crs=grid_crs, geometry=geometry)
	# Read in polygon data
	dfpoly =  gpd.read_file(os.path.join(path, infname_poly))
	dfpoly['ibatch'] = dfpoly.index.values.astype(int)
	# Check before merging that grid and poly are in same crs, of not, convert poly to same crs as grid
	if not (dfpoly.crs == gdf.crs):
		dfpoly = dfpoly.to_crs(gdf.crs)
	# Spatially merge points that are within polygons and keep point grid geometry
	dfcomb = gpd.sjoin(gdf, dfpoly, how="left", op="within") 
	# alternatively use op='intersects' to includfe also points that are on boundary of polygon
	# Remove all points that are not in a polygon
	dfcomb.dropna(subset = ['ibatch'], inplace=True)
	dfcomb['ibatch'] = dfcomb['ibatch'].astype(int) # Need to be converted again to int since merge changes type
	if len(dfcomb) == 0:
		print('WARNING: NO GRID POINTS IN POLYGONS!')
	
	# if convert_to_crs is not None:
	# 	# Convert to meters and get batch x,y coordinate list for each polygon:
	# 	dfcomb = dfcomb.to_crs(epsgcrs)
	# 	dfpoly = dfpoly.to_crs(epsgcrs)
	
	""" If evalutaion points are not given by grid points, following approaches can be tried
	evaluation  points need to subdivide polygon in equal areas (voronoi), otherwise not equal spatial weight 
	options a) rasterize polygon  and then turn pixels in points, b) use pygridtools https://github.com/Geosyntec/pygridtools
	c) use Centroidal Voronoi tessellation (use vorbin or pytess), d) use iterative inwards buffering, e) find points by dynmaic simulation
	or split in halfs iteratively: https://snorfalorpagus.net/blog/2016/03/13/splitting-large-polygons-for-faster-intersections/
	use centroid, then any line through centroid splits polygon in half
	Porpose new algorithm: centroidal hexagonal grid (densest cirle packing), 
	this need optimise only one parameter: rotational angle so that a) maximal number of poinst are in polygon and 
	b) that distance of point to nearest polygon edge is maximised for all points (which aligns points towards center)
	"""
	if isinstance(name_features2, list): 
		selcols = ['x', 'y', 'ibatch']
		selcols.extend(name_features2)
	else:
		selcols = ['Sample.ID','x', 'y', 'ibatch', name_features2]
	return dfcomb[selcols].copy(), dfpoly, name_features2


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


def main(fname_settings):
    """
    Main function for running the script.

    Input:
        fname_settings: path and filename to settings file
    """
    # Load settings from yaml file
    with open(fname_settings), 'r') as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
    # Parse settings dictinary as namespace (settings are available as 
    # settings.variable_name rather than settings['variable_name'])
    settings = SimpleNamespace(**settings)

    # Verify output directory and make it if it does not exist
    os.makedirs(settings.outpath, exist_ok = True)

    # Preprocess data
    df, name_features = preprocess(settings.inpath, settings.infname, settings.outfname, settings.name_target, settings.name_features, zmin = 100*settings.zmin, zmax= 100*settings.zmax, categorical = 'Soiltype',
	colname_depthmin = settings.colname_depthmin, colname_depthmax = settings.colname_depthmax)


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Preprocess data for machine learning on soil data.')
    parser.add_argument('-s', '--settings', type=str, required=False,
                        help='Path and filename of settings file.',
                        default = _fname_settings)
    args = parser.parse_args()

    # Run main function
    main(args.settings)