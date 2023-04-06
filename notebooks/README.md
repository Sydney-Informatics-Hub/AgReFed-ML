# Notebook Guide

This folder contains a series of Jupyter notebooks that demonstrate the workflows for [three use case scenarios](#notebooks-ordered-by-use-case) for agricultural research. Each scenario is described by a reproducible workflow process that typically includes the following three steps:
1. Feature selection
2. Model selection via multi-model training and cross-validation
3. Generating maps/cubes of soil properties and uncertainties

For the prediction workflows, i.e., generating soil prediction maps and uncertainty maps, it is recommended to test the predictions first by generating maps at low resolution. This is to verify that the map output is working properly and to reduce the computational time. The resolution can then be increased to the desired resolution. Note that the compute-time for block and polygon type prediction will take longer than point type prediction due to the complexity of computing the spatial covariance matrix of all support points within each polygon or block. This covariance is needed for computing the spatial-averaged uncertainty for each block or polygon.

## Sample data

The sample data used in the notebooks is available in the zipped file samples.zip. Please download or fork github repo and unzip the file in the same folder as the notebooks.

The sample data are provided by the University of Sydney and includes 3D soil data of organic carbon stock and soil moisture plus multiple surface covariates (e.g., DEM, NDVI, rain etc.). An overview of the surface covariance data and their selection based on their importance is demonstrated in the notebook examples (see steps for features importance and selection). In addition, covariate grid data are provided for the prediction mapping. The covariate grids have been supplemented with data generated via the [Geodata-Harvester](https://github.com/Sydney-Informatics-Hub/geodata-harvester).

All data are provided as CSV files.

## Settings files

The settings files are used to configure the workflows and to make the work process reproducible. The settings files are in YAML format and are used to configure the parameters for each workflow. Typical parameters include:
- Input and output paths 
- Input data filename
- Feature names
- Model function names
- Cooordinate reference system
- model resolution

## Notebooks ordered by use case

### A) Static Soil Model

The static model is a spatial model for generating prediction maps of soil properties for one given time. The output are geo-referenced prediction and uncertainty maps (2D) at multiple soil depths.

Relevant notebooks and settings files:

- Feature selection: [`feature_importance_static.ipynb`](feature_importance_static.ipynb) + [`settings_featureimportance_static.yaml`](settings/settings_featureimportance_static.yaml)
- Model testing: [`testmodels_static.ipynb`](testmodels_static.ipynb) + [`settings_testmodels_static.yaml`](settings/settings_testmodels_static.yaml)
- 3D soil predictions: [`prediction_static3d_OC.ipynb`](prediction_static3d_OC.ipynb) + [`settings_soilmod_predict_static.yaml`](settings/settings_soilmod_predict_static.yaml)
- 3D soil predictions for different spatial types (points, polygons, blocks): [`prediction_staticpoly_OC.ipynb`](prediction_staticpoly_OC.ipynb) + [`settings_soilmod_predict_polystatic.yaml`](settings/settings_soilmod_predict_polystatic.yaml)


### B) Change Model for Carbon Accounting Mapping

This workflow generates prediction and uncertainty maps for the change of soil properties within a certain period of time. The use-case goal for this example is to model the change of the Organic Carbon (OC) stock volume for a farm. A particular focus is to model the uncertainty of the change, which needs to take into account the covariances of the prediction in space and time. The model training data is based on laboratory measurements of Organic Carbon stock for 2018 and 2021 from sample sites.

Relevant notebooks and settings files:

- notebook: [`model_soilchange_OC.ipynb`](model_soilchange_OC.ipynb)
- settings files: 
    - for step feature selection: [`settings_featureimportance_OC.yaml`](settings/settings_featureimportance_OC.yaml), 
    - for step model selection: [`settings_soilmod_xval_OC.yaml`](settings/settings_soilmod_xval_OC.yaml), 
    - for step prediction mapping: [`settings_soilmod_changepredict_OC.yaml`](settings/settings_soilmod_changepredict_OC.yaml)

### C) Spatial-Temporal Model

This workflow generates soil moisture prediction maps and their uncertainty for multiple time intervals. For this example we focus on the top-soil properties only. Model training data is based on daily and weekly averaged data from soil moisture probes and multiple spatiotemporal dependent covariates for 2020-2022 from sample sites. Soil moisture data is provided by the University of Sydney and multiple spatial-temporal covariates are extracted with the [Geodata-Harvester](https://github.com/Sydney-Informatics-Hub/geodata-harvester).

Relevant notebooks and settings files:

- Feature selection: [`feature_selection_moisture.ipynb`](feature_selection_moisture.ipynb) (this notebook includes functions for generating settings file)
- Model testing: [`testmodels_st_moisture.ipynb`](testmodels_st_moisture.ipynb) + `settings_soilmod_xval_moisture_20xx.yaml`
- Prediction notebooks for 2020 and 2021: 
    - GPR plus Random Forest: [`prediction_st_moisture.ipynb`](prediction_st_moisture.ipynb) + `settings_soilmod_moisture_predict_20xx.yaml`
    - GPR plus Bayesian Linear Regression: [`prediction_st_blr-gp_moisture.ipynb`](prediction_st_blr-gp_moisture.ipynb) + `settings_soilmod_moisture_predict_blr-gp_20xx.yaml` 