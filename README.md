# fire_prediction

# Wildfire Prediction Models

This repository contains Python scripts for wildfire prediction using meteorological and environmental data.  
The project focuses on training deep learning and machine learning models, generating fire-risk heatmaps, and applying explainability techniques.

## Files

### `unet_train.py`
Trains an improved UNet model for wildfire prediction on gridded meteorological data.  
It includes preprocessing, feature engineering, gap-aware 7-day windowing, masked focal loss, train/validation/test split, evaluation metrics, and explainability outputs.

### `unet_sensitivity.py`
Performs sensitivity analysis for the trained UNet model.  
It is used to understand how input features or spatial regions affect the wildfire prediction output.

### `export_heatmap.py`
Generates an interactive HTML heatmap dashboard using predictions from the trained UNet model.  
It can automatically select high-fire-risk days or use specific dates provided by the user.

### `elevated_unet.py`
Implements an uncertainty-aware extension of the trained UNet model using a residual conditional VAE.  
It produces mean fire-risk predictions and uncertainty maps.

### `elevated_unet_occlusion_sensitivity.py`
Runs occlusion sensitivity analysis on the elevated UNet model.  
It shows which spatial areas influence the base risk prediction, elevated risk prediction, and uncertainty estimation.

### `conv2dupdatetdmodern.py`
Contains a Vision Transformer / Conv2D-based wildfire prediction model.  
It uses spatio-temporal meteorological grids and compares deep learning behavior against UNet-style models.

### `train_hybrid_rocket.py`
Trains a Hybrid MiniRocket model for cell-level wildfire prediction.  
It uses time-series transformations, spatial neighbor features, calibration, threshold tuning, and evaluation on validation/test data.

### `train_lstm.py`
Trains classical and neural machine learning models for wildfire classification.  
It includes feature engineering, SMOTE balancing, scaling, and comparison between Logistic Regression, Random Forest, XGBoost, LightGBM, MLP, and ensemble models.

## Goal

The goal of the repository is to compare different approaches for wildfire prediction:

- tabular machine learning models
- time-series models
- UNet-based spatial prediction
- transformer-based spatial prediction
- uncertainty-aware wildfire risk mapping
- explainability and sensitivity analysis

## Requirements

Main libraries used:

```bash
pip install numpy pandas tensorflow scikit-learn matplotlib xgboost lightgbm imbalanced-learn sktime joblib
