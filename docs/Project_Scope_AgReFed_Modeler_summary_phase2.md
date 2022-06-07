# Project Scope: AgReFed Machine Learning Tools II

## Project Administration

Title: **Machine learning models under uncertainty for agricultural systems**

Prepared by: Sebastian Haan

Prepared for: AgReFed

Date Prepared: June 2022

Affiliation: Sydney Informatics Hub, The University of Sydney



## Project Summary

### Research Context

Problem: Currently agricultural researchers have models which are of high reuse value to the agricultural community. These models require inter-operable data flows of appropriately calibrated, cleaned data variables. Understanding the model limitations, assumptions and then interpreting the outputs is required. This takes time and a high level of expertise across a number of areas depending on the data type/s, data condition and model complexity.


This project will contribute software scripts that provide multiple machine learning workflows and tools for agriculture researchers. One first pilot project will be to develop a software tool to map soil properties under sparse and uncertain input. While this tool will be tested first on mapping soil bulk density and carbon concentration, it can be used for a diverse range of soil property predictions such as sodicity, salinity, pH-values and many more.

### Client Needs

Software tool to predict soil properties and uncertainties.

The modelling approach should ideally have the following features: 

- accommodate the spatial (-temporal) support of the observations
- accommodate the spatial (-temporal) auto-correlation of the observations
- accommodate measurement error of the observations
- incorporate cheap to measure and numerous variables as predictors
- accommodate measurement error of the covariates
- when predicting give both a point and uncertainty (confidence interval) estimate 
- be able to predict at any spatial (-temporal) support
- Optional: model multiple prediction targets simultaneously by taken into account the correlations between them.


## Project Implementation


### Workpackages for ML:

1) Static: Focus on spatial extraction for one given time (as tested by current prototype, L'lara spatial, see below)

2) Two time points for long-term change (e.g., change in organic carbon); needs multiple averages of covariates before and after measurements

3) Multiple time points regular space at smaller intervals (e.g., for soil moisture)


### Project Summary of ML package 1

For this project we developed a workflow around the use of Gaussian Process Regression (GPR) which includes:
- a mean function relating the response to a data cube of predictors through a regression/ML model;
- a GPR on the residual to accommodate the measurement error and the spatial structure in the observations.

The output maps soil properties on a grid of user define resolution including uncertainty predictions.

The current package includes:

- tools for covariate feature selection and model evaluation
- pre-processing tools, e.g. for data conversion and geo-referencing
- Implementation and test of a range mean function models, i.e. Random Forest and Bayesian Linear Regression
- GPR with custom spatial 3D kernel functions to include measurement uncertainties 
- test of integrated GPR model on synthetic data set with spatial correlated fields and simulated uncertainties
- test on use-case scenario of spatial map for farm data of L'lara


### Project Plan for ML package 2

The following additional functionalities are needed:

- spatial-temporal modeling with focus on covariance between two time dates
- extraction of covariates with multiple time window averages
- use-case: Map change of soil properties between 2 time points
- estimate the uncertainty of change to allow hypothesis testing, including modeling of cross-correlation between 2 time points
- predictions for custom shapes: estimate carbon change for a grid, blocks, polygons representing zones and/paddocks


### Data Availability
Soil data is available for L'lara and covariates can be extracted from public data-sources.


### Deliverables
- Python software package
- Documentation of package including functionality and installation
- Examples and use-case scenarios


### In Scope
- Software package for data aggregation and processing
- Software testing and review
- Documentation of package including functionality and installation guide


### Out of Scope & Exclusions
- Long-term maintenance and updates of delivered Software tools
- Installation and software assistance beyond testing phase and documentation
- Publication (can be considered in future depending on availability and need)
- User interface (potential as future optional update)


## Time Estimate

next milestone delivery for mid-Nov 2022