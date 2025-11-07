# Tutorial: AgReFed-ML

**AgReFed-ML** is a machine learning framework specifically designed for **agricultural soil modeling**. It transforms *sparse soil measurements* into detailed **spatial-temporal prediction maps** of soil properties and their **uncertainties** using advanced *Gaussian Process regression*. The system combines traditional machine learning models (like Random Forest and Bayesian Linear Regression) as **mean functions** with sophisticated *spatial correlation modeling* to make accurate predictions even with limited data. It's particularly valuable for **precision agriculture** applications like carbon accounting, soil moisture mapping, and property change detection over time.


**Source Repository:** [https://github.com/Sydney-Informatics-Hub/AgReFed-ML](https://github.com/Sydney-Informatics-Hub/AgReFed-ML)

```mermaid
flowchart TD
    A0["Gaussian Process Models
"]
    A1["Mean Function Models  
"]
    A2["Prediction Workflows
"]
    A3["Data Preprocessing Pipeline
"]
    A4["Model Evaluation and Cross-Validation
"]
    A5["Spatial-Temporal Modeling Framework
"]
    A6["Uncertainty Quantification System
"]
    A7["Notebook-Based Workflows
"]
    A8["Synthetic Data Generation
"]
    A2 -- "Uses for prediction" --> A0
    A2 -- "Uses as baseline" --> A1
    A0 -- "Incorporates" --> A6
    A2 -- "Uses for preparation" --> A3
    A4 -- "Evaluates" --> A0
    A4 -- "Tests performance" --> A1
    A5 -- "Extends with 4D kernels" --> A0
    A7 -- "Demonstrates usage" --> A2
    A8 -- "Provides test data" --> A4
```

## Chapters

1. [Notebook-Based Workflows
](01_notebook_based_workflows_.html)
2. [Prediction Workflows
](02_prediction_workflows_.html)
3. [Data Preprocessing Pipeline
](03_data_preprocessing_pipeline_.html)
4. [Mean Function Models  
](04_mean_function_models___.html)
5. [Gaussian Process Models
](05_gaussian_process_models_.html)
6. [Uncertainty Quantification System
](06_uncertainty_quantification_system_.html)
7. [Spatial-Temporal Modeling Framework
](07_spatial_temporal_modeling_framework_.html)
8. [Synthetic Data Generation
](08_synthetic_data_generation_.html)
9. [Model Evaluation and Cross-Validation
](09_model_evaluation_and_cross_validation_.html)


---
