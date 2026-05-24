# Hemispheric Functional Connectivity Based Brain Age Prediction for Alzheimer's Disease using fMRIs

## Overview

This project implements multiple machine learning approaches to predict brain age from functional MRI data, with a focus on detecting Alzheimer's Disease using hemispheric functional connectivity patterns. The project compares several modeling techniques including CNN, GNN, SVR, and Ridge Regression to determine the most effective approach for brain age prediction.

## Project Structure

preprocessing_pipeline/
  Contains the data preprocessing pipeline for fMRI data normalization and feature extraction.
  
ridge_regression/
  Implementation of Ridge Regression and SVM classification models for brain age prediction.
  
CNN:GNN/
  Convolutional Neural Network and Graph Neural Network implementations for learning from connectivity patterns.
  
SVR and SVM/
  Support Vector Regression and Support Vector Machine models for classification and regression tasks.

Dataset/
  Links and references to the fMRI dataset used in this project.

## Project Objectives

1. Develop multiple machine learning models to predict brain age from fMRI data
2. Identify hemispheric functional connectivity patterns associated with Alzheimer's Disease
3. Compare model performance across different architectures (CNN, GNN, SVR, Ridge Regression)
4. Create interpretable biomarkers for disease detection and prognosis

## Requirements

Python 3.7+
NumPy
Pandas
Scikit-learn
TensorFlow/Keras (for CNN and GNN models)
PyTorch (alternative deep learning framework)
Matplotlib
Seaborn
Nibabel (for neuroimaging data)
Scipy

## Installation

1. Clone the repository:
   git clone https://github.com/RushilGargash08/Hemispheric-Functional-Connectivity-Based-Brain-Age-Prediction-for-Alzheimer-s-Disease-using-fMRIs.git
   cd mlpr-2

2. Create a virtual environment:
   python -m venv env
   source env/bin/activate

3. Install required packages:
   pip install -r requirements.txt

## Dataset

The project uses fMRI data from Alzheimer's Disease research. Please refer to Dataset/Dataset_link.txt for download instructions and data availability.

## Usage

### Preprocessing

Run the preprocessing pipeline to prepare raw fMRI data:
   jupyter notebook preprocessing_pipeline/Preprocessing_Pipeline.ipynb

### Ridge Regression Model

Execute the ridge regression models:
   python ridge_regression/run_ridge.py
   python ridge_regression/run_svm_classification.py

### CNN and GNN Models

Run the deep learning models:
   jupyter notebook CNN:GNN/brain_age_models.ipynb

### Support Vector Models

Execute SVR and SVM implementations:
   jupyter notebook "SVR and SVM/SVR_SVM.ipynb"

## Results

The project evaluates model performance based on:
- Mean Absolute Error (MAE) for age prediction
- R-squared score for regression tasks
- Classification accuracy for disease detection
- AUC-ROC scores for binary classification

Refer to ridge_regression/ridge_regression_results.md for detailed results and comparisons.

## Model Descriptions

Ridge Regression: Linear regression with L2 regularization for brain age prediction from connectivity features.

Support Vector Regression: Non-linear regression using kernel methods to capture complex relationships in connectivity data.

Convolutional Neural Networks: Deep learning model for learning spatial patterns from fMRI connectivity matrices.

Graph Neural Networks: Advanced architecture leveraging graph structure of brain connectivity networks for improved predictions.

Support Vector Machines: Classification model for distinguishing between Alzheimer's and healthy control subjects.

## Project Highlights

Multi-modal approach comparing traditional machine learning with deep learning techniques
Hemispheric analysis focusing on left-right brain connectivity asymmetries
Comprehensive evaluation across multiple metrics and validation strategies
Interpretable biomarkers for clinical applications

## Contributing

For contributing to this project, please submit pull requests with detailed descriptions of changes and improvements.

## Authors

Rushil Gargash

## License

This project is provided as-is for research purposes. Please cite appropriately if used in academic work.