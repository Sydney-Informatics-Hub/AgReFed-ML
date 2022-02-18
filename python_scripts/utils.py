"""
Some useful functions

This package is part of the machine learning project developed for the Agricultural Research Federation (AgReFed).

Copyright 2022 Sebastian Haan, Sydney Informatics Hub (SIH), The University of Sydney

This open-source software is released under the AGPL-3.0 License.
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import Affine
from scipy import spatial


def project_data(infname, outfname, colname_lng_source, colname_lat_source, tocrs_epsg):
    """
    Projects csv file into new coordinate reference system (crs) such as from Lat/Long to meter
    The projected file with convertd coordinates are saved as csv.

    Input
    ----
        infname: inoput path + filename
        infname: output path + filename

    Result
    ------
    Csv file saved to disk

    """
    df_soil = pd.read_csv(infname)
    #df_soil = pd.read_csv(os.path.join(inpath,'ESP_original_depths_covariates_NAMOI.csv'))
    #First convert WGS84 ( or epsg:4326) to meters, here we choose default web merkator projection (epgs:3857) 
    gdf = gpd.GeoDataFrame(df_soil.copy(), geometry=gpd.points_from_xy(df_soil[colname_lng_source], df_soil[colname_lat_source]))
    #gdf.crs = {'init' :'epsg:4326'} # Depreciated with geopandas >= 0.7
    gdf.crs = "EPSG:4326"
    #gdf = gdf.to_crs('EPSG:3857')
    gdf = gdf.to_crs(tocrs_epsg)
    gdf['Easting'] = gdf.geometry.x
    gdf['Northing'] = gdf.geometry.y
    #gdf.to_file(os.path.join(inpath, 'clipped_ESP_llara_COVARIATES_proj.gpkg'), driver='GPKG')
    # Save without geometry column as csv
    gdf.drop('geometry', axis=1).to_csv(outfname)


def array2geotiff(array, xy0, xyres, outfname, epsg_crs):
    """
    Converts numpy array into geopastial refernced tiff image (geotiff)

    Input
    -----
        array: numpy 2-dim array with height = shape[0] and width = shape[1]
        xy0: Coordinate origin [x0,y0] in given crs
        xyres: resolution [xres, yres]
        outfname: output path + filename.tif
        crs: Coordinate reference system, e.g. 'EPSG:28355'

    Return
    ------
        Saves geotiff file
        returns True/False
    """
    crs = rasterio.crs.CRS.from_string(epsg_crs)
    transform = Affine.translation(xy0[0] - xyres[0] / 2, xy0[1] - xyres[1] / 2) * Affine.scale(xyres[0], xyres[1])
    # Write to file:
    try:
        with rasterio.open(
            outfname,
            'w',
            driver='GTiff',
            height=array.shape[0],
            width=array.shape[1],
            count=1,
            dtype=array.dtype,
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(array, 1)
        return True
    except:
        print("ERROR: Writing array to Geotiff failed!")
        return False


def align_nearest_neighbor(xy_orig, xy_ref, data_orig, max_dist):
    """
    Find nearest neighbor in reference coordinates and 
    return matching data at locations and with shape of reference grid

    Input
    -----
        xy_orig: original positions, 2D array with shape (n_orig,2)
        xy_ref: reference positions, 2D array with shape (n_ref, 2)
        data_orig: array with n_orig data values, or list with multiple arrays
        max_dist: maximum distance for nearest neighbor, 
        reference array with distances larger than that are set to nan values

    Return
    ------
        array or list of arrays with values that match refernce grid
    """
    assert xy_orig.shape[1] == xy_ref.shape[1] == 2
    tree = spatial.cKDTree(xy_ref)
    ind_nearest = tree.query(xy_orig)[1]
    nearest_dist = np.sqrt((xy_orig[:,0] - xy_ref[ind_nearest][:,0])**2 
        + (xy_orig[:,1] - xy_ref[ind_nearest][:,1])**2)
    if isinstance(data_orig, list):
        res = []
        for data in data_orig:
            ires = data[ind_nearest]
            ires[nearest_dist > max_dist] = np.nan
            res.append(ires)
    else:
        res = data_orig[ind_nearest]
        res[nearest_dist > max_dist] = np.nan
    return res

def print2(text, fname_out = os.path.join('loginfo.txt')):
	"""
	Prints text to standard output (typically terminal) and to file simultaneously
	Note text: Advise to use f-string for complex text
	Alternative: Use logger info

	INPUT:
	text: string (if 'init' new header will be written, and old contentwill be overwritten)
	fname_out: path + file name (Default: {outpath}/info.txt)
	""" 
	if text == 'init':
		# Initialise a new file (e.g., if program is restarted)
		if os.path.isfile(fname_out):
			with open(fname_out, 'w') as f:
				print('OUTPUT INFO', file = f) 
				print('-----------', file = f) 
			pass
	else:
		print(text) 
		if os.path.isfile(fname_out):
			arg = 'a' 
		else:
			arg = 'w' 
		with open(fname_out, arg) as f:
			print(text, file = f) 

def truncate_data(data, cutperc):
	"""
	Combines lits of arrays and truncates with lower and upper percentile

	INPUT
	data: list of numpy arrays
	cutperc: percent to cut off from percentile

	Return
	truncated flattened data array
	"""
	datacomb = data[0]
	for i in range(len(data)-1):
		datacomb = np.concatenate((datacomb, data[i+1]))
	low = np.percentile(datacomb, cutperc)
	high = np.percentile(datacomb, 100-cutperc)
	return datacomb[(datacomb >= low) & (datacomb <= high)]


def create_vtkcube(data, origin, voxelsize, fname):
    """
    Export Cube as VTK file (can be used in e.g. ParaView)
    and create a range of 3D cube plots with pyvista

    INPUT
    :param data: 3D cube in shape (xdim, ydim, zdim)
    :param origin: origin cooridnates of cube
    :param voxelsize: voxel sizes in (xsize, ysize, zsize)
    :param fname: path + filename for files
    """
    grid = pv.UniformGrid()
    grid.dimensions = np.array(data.shape) + 1
    grid.origin = origin
    grid.spacing = voxelsize
    grid.cell_arrays["values"] = data.flatten(order="F")
    grid.save(fname)

