"""
Model evaluation functions for the Bayou_Sp25 project.

This module contains functions to evaluate model performance,
calculate error metrics, and compare different models.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os
import sys

# Add the project root to sys.path to allow imports from other modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import settings
from models.poisson_model import predict

def calculate_error_metrics(actual, predicted, logger=None):
    """
    Calculate error metrics for model evaluation.
    
    Parameters:
    -----------
    actual : array-like
        Actual values
    predicted : array-like
        Predicted values
        
    Returns:
    --------
    dict
        Dictionary of error metrics
    """
    if actual is None or predicted is None:
        message = "Error: Cannot calculate metrics - actual or predicted values are missing"
        if logger:
            logger.info(message)
            print(message)
        else:
            print(message)
        return None
    
    # Calculate error metrics
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    
    # Calculate mean absolute percentage error (MAPE)
    # Avoid division by zero by adding a small constant
    mape = np.mean(np.abs((actual - predicted) / (actual + 1e-10))) * 100
    
    # Return metrics as a dictionary
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }
    
    return metrics

def calculate_accuracy_metrics(actual, predicted, logger=None):
    """
    Calculate accuracy metrics for count data.
    
    Parameters:
    -----------
    actual : array-like
        Actual count values
    predicted : array-like
        Predicted count values (float)
        
    Returns:
    --------
    dict
        Dictionary of accuracy metrics
    """
    if actual is None or predicted is None:
        message = "Error: Cannot calculate metrics - actual or predicted values are missing"
        if logger:
            logger.info(message)
            print(message)
        else:
            print(message)
        return None
    
    # Round predictions to nearest integer
    predicted_rounded = np.round(predicted)
    
    # Calculate accuracy metrics for count data
    exact_match = np.mean(predicted_rounded == actual)
    within_1 = np.mean(abs(predicted_rounded - actual) <= 1)
    within_2 = np.mean(abs(predicted_rounded - actual) <= 2)
    
    # Return metrics as a dictionary
    metrics = {
        'Exact_Match': exact_match,
        'Within_1': within_1,
        'Within_2': within_2
    }
    
    return metrics

def evaluate_model(model_result, data, response_var='overflow', data_label='Test', logger=None):
    """
    Evaluate model performance on a dataset.
    
    Parameters:
    -----------
    model_result : statsmodels.genmod.generalized_linear_model.GLMResultsWrapper
        Fitted model object
    data : pandas.DataFrame
        Data for evaluation
    response_var : str
        Response variable name (default: 'overflow')
    data_label : str
        Label for the dataset (default: 'Test')
        
    Returns:
    --------
    tuple
        (data_with_predictions, error_metrics, accuracy_metrics)
    """
    if model_result is None or data is None:
        message = "Error: Cannot evaluate model - model or data is missing"
        if logger:
            logger.info(message)
            print(message)
        else:
            print(message)
        return None, None, None
    
    # Generate predictions
    data_with_predictions = data.copy()
    data_with_predictions['predicted'] = predict(model_result, data)
    
    # Calculate error metrics
    error_metrics = calculate_error_metrics(
        data_with_predictions[response_var], 
        data_with_predictions['predicted']
    )
    
    # Calculate accuracy metrics
    accuracy_metrics = calculate_accuracy_metrics(
        data_with_predictions[response_var], 
        data_with_predictions['predicted']
    )
    
    # Print performance metrics
    if logger:
        logger.info(f"\n{data_label} Set Performance Metrics:")
        logger.info(f"Mean Squared Error: {error_metrics['MSE']:.4f}")
        logger.info(f"Root Mean Squared Error: {error_metrics['RMSE']:.4f}")
        logger.info(f"Mean Absolute Error: {error_metrics['MAE']:.4f}")
        logger.info(f"Exact Count Match Accuracy: {accuracy_metrics['Exact_Match']:.4f} ({accuracy_metrics['Exact_Match']*100:.1f}%)")
        logger.info(f"Within ±1 Count Accuracy: {accuracy_metrics['Within_1']:.4f} ({accuracy_metrics['Within_1']*100:.1f}%)")
        logger.info(f"Within ±2 Count Accuracy: {accuracy_metrics['Within_2']:.4f} ({accuracy_metrics['Within_2']*100:.1f}%)")
        
        print(f"\n{data_label} Set Performance Metrics:")
        print(f"Mean Squared Error: {error_metrics['MSE']:.4f}")
        print(f"Root Mean Squared Error: {error_metrics['RMSE']:.4f}")
        print(f"Mean Absolute Error: {error_metrics['MAE']:.4f}")
        print(f"Exact Count Match Accuracy: {accuracy_metrics['Exact_Match']:.4f} ({accuracy_metrics['Exact_Match']*100:.1f}%)")
    else:
        print(f"\n{data_label} Set Performance Metrics:")
        print(f"Mean Squared Error: {error_metrics['MSE']:.4f}")
        print(f"Root Mean Squared Error: {error_metrics['RMSE']:.4f}")
        print(f"Mean Absolute Error: {error_metrics['MAE']:.4f}")
        print(f"Exact Count Match Accuracy: {accuracy_metrics['Exact_Match']:.4f} ({accuracy_metrics['Exact_Match']*100:.1f}%)")
        
    
    return data_with_predictions, error_metrics, accuracy_metrics

def compare_models(models_dict, test_data, response_var='overflow', logger=None):
    """
    Compare multiple models based on performance metrics.
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary of model results with model names as keys
    test_data : pandas.DataFrame
        Test data for evaluation
    response_var : str
        Response variable name (default: 'overflow')
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with performance metrics for each model
    """
    if not models_dict or test_data is None:
        print("Error: Cannot compare models - models or test data is missing")
        return None
    
    # Initialize lists to store results
    model_names = []
    mse_values = []
    rmse_values = []
    mae_values = []
    exact_match_values = []
    within_1_values = []
    
    # Evaluate each model
    for model_name, model_result in models_dict.items():
        if logger:
            logger.info(f"\nEvaluating model: {model_name}")
            print(f"\nEvaluating model: {model_name}")
        else:
            print(f"\nEvaluating model: {model_name}")
        
        # Evaluate model
        _, error_metrics, accuracy_metrics = evaluate_model(
            model_result, 
            test_data, 
            response_var=response_var, 
            data_label=f"{model_name}"
        )
        
        # Store results
        model_names.append(model_name)
        mse_values.append(error_metrics['MSE'])
        rmse_values.append(error_metrics['RMSE'])
        mae_values.append(error_metrics['MAE'])
        exact_match_values.append(accuracy_metrics['Exact_Match'])
        within_1_values.append(accuracy_metrics['Within_1'])
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame({
        'Model': model_names,
        'MSE': mse_values,
        'RMSE': rmse_values,
        'MAE': mae_values,
        'Exact_Match': exact_match_values,
        'Within_1': within_1_values
    })
    
    # Sort by MSE (lower is better)
    comparison_df = comparison_df.sort_values('MSE')
    
    # Print comparison table
    if logger:
        logger.info("\nModel Comparison:")
        logger.info(comparison_df)
        print("\nModel Comparison:")
        print(comparison_df)
    else:
        print("\nModel Comparison:")
        print(comparison_df)
    
    return comparison_df

def analyze_zipcode_performance(data_with_predictions, response_var='overflow', logger=None):
    """
    Analyze model performance by zipcode.
    
    Parameters:
    -----------
    data_with_predictions : pandas.DataFrame
        DataFrame with actual and predicted values
    response_var : str
        Response variable name (default: 'overflow')
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with performance metrics by zipcode
    """
    if data_with_predictions is None:
        message = "Error: Cannot analyze zipcode performance - data is missing"
        if logger:
            logger.info(message)
            print(message)
        else:
            print(message)
        return None
    
    # Calculate metrics by zipcode
    zipcode_performance = data_with_predictions.groupby('Zipcode').agg({
        response_var: 'mean',
        'predicted': 'mean',
        'median_income': 'first',
        'percent_white': 'first',
        'percent_black': 'first',
        'percent_asian': 'first',
        'population': 'first'
    })
    
    # Calculate error by zipcode
    zipcode_performance['error'] = zipcode_performance['predicted'] - zipcode_performance[response_var]
    zipcode_performance['abs_error'] = np.abs(zipcode_performance['error'])
    
    # Find zipcodes with best and worst predictions
    if logger:
        logger.info("\nZipcodes with Most Accurate Predictions:")
        logger.info(zipcode_performance.nsmallest(5, 'abs_error'))
        print("\nZipcodes with Most Accurate Predictions:")
        print(zipcode_performance.nsmallest(5, 'abs_error'))
    else:
        print("\nZipcodes with Most Accurate Predictions:")
        print(zipcode_performance.nsmallest(5, 'abs_error'))
    
    if logger:
        logger.info("\nZipcodes with Least Accurate Predictions:")
        logger.info(zipcode_performance.nlargest(5, 'abs_error'))
        print("\nZipcodes with Least Accurate Predictions:")
        print(zipcode_performance.nlargest(5, 'abs_error'))
    else:
        print("\nZipcodes with Least Accurate Predictions:")
        print(zipcode_performance.nlargest(5, 'abs_error'))
    
    return zipcode_performance