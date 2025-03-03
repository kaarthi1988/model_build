# ML Model Builder & Report Generator

## Overview

ML Model Builder is an end-to-end machine learning project that streamlines the process of building, evaluating, and interpreting both regression and classification models. The project features a user-friendly graphical interface (GUI) built with Plotly Dash and Bootstrap components. Users can easily upload data, provide inputs, and generate interactive reports.

The system performs:
- **Data Preprocessing:**  
  Generates a detailed preprocessing report covering data cleaning, feature engineering, and exploratory data analysis (EDA).
- **Model Training & Selection:**  
  Trains multiple models for regression or classification using [PyCaret](https://pycaret.org/), evaluates their performance using key metrics, and automatically saves the best-performing model.
- **Explainable AI:**  
  Integrates SHAP to provide visual explanations (e.g., waterfall, summary plots) for model predictions.
- **Reporting:**  
  Builds two interactive reports:
  - A **Preprocessing Report** to help understand the data preparation steps.
  - An **ML Model Build Report** that compares model metrics and includes SHAP-based explainability for the chosen model.

## Features

- **User-Friendly GUI:**  
  A dashboard built using Plotly Dash and Bootstrap allows users to:
  - Upload and explore datasets.
  - Select between regression and classification tasks.
  - Input parameters for data preprocessing and model training.
  - View real-time interactive charts and reports.
    <img width="704" alt="Screenshot 2025-03-03 at 2 54 03 PM" src="https://github.com/user-attachments/assets/d8da4897-3110-4055-917b-8592cb304e12" />


- **Automated Modeling with PyCaret:**  
  Leverage PyCaret to quickly train, evaluate, and compare multiple machine learning models. PyCaret simplifies model selection by automatically:
  - Comparing different models.
  - Optimizing hyperparameters.
  - Selecting and saving the best model based on user-defined metrics.

- **Flexible Modeling:**  
  Supports both regression and classification tasks. Multiple models are trained and compared based on performance metrics. The final, best-performing model is automatically saved for deployment or further use.

- **Interactive Reports:**  
  - **Preprocessing Report:**  
    Provides insights into data quality, feature distributions, and preprocessing steps.
  - **ML Model Build Report:**  
    Displays performance comparisons (using bar charts, pie charts, etc.) and explainable AI visualizations with SHAP.

- **Explainable AI:**  
  Uses SHAP to generate detailed visual explanations (e.g., waterfall, summary, and dependence plots) that help users understand feature contributions to predictions.

## Technologies

- **Programming Language:** Python 3.x  
- **Frameworks & Libraries:**  
  - [Plotly Dash](https://dash.plotly.com/) & [Dash Bootstrap Components](https://dash-bootstrap-components.opensource.faculty.ai/)
  - [PyCaret](https://pycaret.org/) for rapid model training and evaluation  
  - [scikit-learn](https://scikit-learn.org/) for additional machine learning support  
  - [Pandas](https://pandas.pydata.org/) and [NumPy](https://numpy.org/) for data manipulation  
  - [SHAP](https://github.com/slundberg/shap) for explainability  
- **Visualization:** Plotly Express and Plotly Graph Objects

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your_username/ml-model-builder.git
   cd ml-model-builder
