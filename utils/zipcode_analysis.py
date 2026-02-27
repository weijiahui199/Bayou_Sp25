"""
Zipcode analysis utilities for the Bayou_Sp25 project.

This module contains functions for analyzing model performance by zipcode,
including identifying best/worst performing zipcodes and demographic effects.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import sys

# Add the project root to sys.path to allow imports from other modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import settings

def identify_best_worst_zipcodes(zipcode_performance, n=5):
    """
    Identify best and worst performing zipcodes.
    
    Parameters:
    -----------
    zipcode_performance : pandas.DataFrame
        DataFrame with performance metrics by zipcode
    n : int
        Number of zipcodes to identify (default: 5)
        
    Returns:
    --------
    tuple
        (best_zipcodes, worst_zipcodes) DataFrames
    """
    if zipcode_performance is None:
        print("Error: Cannot identify zipcodes - data is missing")
        return None, None
    
    # Find zipcodes with lowest absolute error
    best_zipcodes = zipcode_performance.nsmallest(n, 'abs_error')
    
    # Find zipcodes with highest absolute error
    worst_zipcodes = zipcode_performance.nlargest(n, 'abs_error')
    
    # Print the results
    print("\nZipcodes with Most Accurate Predictions:")
    print(best_zipcodes)
    
    print("\nZipcodes with Least Accurate Predictions:")
    print(worst_zipcodes)
    
    return best_zipcodes, worst_zipcodes

def analyze_demographic_effect_on_errors(zipcode_performance):
    """
    Analyze relationship between demographics and prediction errors.
    
    Parameters:
    -----------
    zipcode_performance : pandas.DataFrame
        DataFrame with performance metrics by zipcode
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with correlation results
    """
    if zipcode_performance is None:
        print("Error: Cannot analyze demographic effects - data is missing")
        return None
    
    # Demographic variables to analyze
    demographic_vars = [
        'median_income',
        'population',
        'percent_white',
        'percent_black',
        'percent_asian',
        'percent_other'
    ]
    
    # Ensure all demographic variables exist in the data
    existing_vars = [var for var in demographic_vars if var in zipcode_performance.columns]
    
    if not existing_vars:
        print("Error: No demographic variables found in the data")
        return None
    
    # Calculate correlations with error
    correlations = []
    p_values = []
    
    for var in existing_vars:
        # Calculate Pearson correlation
        corr, p_val = stats.pearsonr(zipcode_performance[var], zipcode_performance['abs_error'])
        correlations.append(corr)
        p_values.append(p_val)
    
    # Create DataFrame with results
    results = pd.DataFrame({
        'Variable': existing_vars,
        'Correlation': correlations,
        'P_Value': p_values,
        'Significant': [p < 0.05 for p in p_values]
    })
    
    # Sort by absolute correlation value
    results = results.reindex(results['Correlation'].abs().sort_values(ascending=False).index)
    
    # Print results
    print("\nDemographic Factors Correlation with Prediction Error:")
    print(results)
    
    # Create bar plot of correlations
    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        results['Variable'],
        results['Correlation'],
        color=[
            'green' if c > 0 and s else 'lightgreen' if c > 0 else 
            'red' if s else 'lightcoral' 
            for c, s in zip(results['Correlation'], results['Significant'])
        ]
    )
    
    # Add value labels
    for bar, corr, p_val in zip(bars, results['Correlation'], results['P_Value']):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.02 if bar.get_height() > 0 else bar.get_height() - 0.08,
            f"{corr:.3f}\n(p={p_val:.3f})",
            ha='center',
            va='bottom' if bar.get_height() > 0 else 'top',
            fontsize=9
        )
    
    # Customize plot
    plt.axhline(0, color='black', linestyle='-', linewidth=0.5)
    plt.ylabel('Correlation with Absolute Error')
    plt.title('Demographic Factors Correlation with Prediction Error')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save and show
    fig_path = os.path.join(settings.FIGURE_DIR, 'demographic_correlations.png')
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=settings.DPI, bbox_inches='tight')
    print(f"Demographic correlations plot saved to {fig_path}")
    
    plt.show()
    
    return results

def create_zipcode_summary(model_data, predictions):
    """
    Create summary statistics by zipcode.
    
    Parameters:
    -----------
    model_data : pandas.DataFrame
        Model data with all features
    predictions : pandas.Series
        Model predictions
        
    Returns:
    --------
    pandas.DataFrame
        Summary statistics by zipcode
    """
    if model_data is None or predictions is None:
        print("Error: Cannot create zipcode summary - data is missing")
        return None
    
    # Create a copy of the data with predictions
    data_with_predictions = model_data.copy()
    
    if len(predictions) != len(data_with_predictions):
        print("Error: Length of predictions does not match data")
        return None
    
    # Add predictions to the data
    data_with_predictions['predicted'] = predictions
    
    # Identify the response variable
    response_var = None
    for var in ['overflow', '311_calls']:
        if var in data_with_predictions.columns:
            response_var = var
            break
    
    if response_var is None:
        print("Error: Could not identify response variable")
        return None
    
    # Calculate error
    data_with_predictions['error'] = data_with_predictions['predicted'] - data_with_predictions[response_var]
    data_with_predictions['abs_error'] = np.abs(data_with_predictions['error'])
    
    # Create summary by zipcode
    zipcode_summary = data_with_predictions.groupby('Zipcode').agg({
        response_var: ['mean', 'std', 'count'],
        'predicted': ['mean', 'std'],
        'error': ['mean', 'std'],
        'abs_error': ['mean', 'median'],
        'rainfall': ['mean', 'max'],
        'median_income': 'first',
        'population': 'first',
        'percent_white': 'first',
        'percent_black': 'first'
    })
    
    # Flatten the column names
    zipcode_summary.columns = [
        '_'.join(col).strip() for col in zipcode_summary.columns.values
    ]
    
    # Calculate additional metrics
    zipcode_summary['mape'] = (
        zipcode_summary['abs_error_mean'] / 
        (zipcode_summary[f"{response_var}_mean"] + 1e-10)  # Avoid division by zero
    ) * 100
    
    # Sort by absolute error (worst performing first)
    zipcode_summary = zipcode_summary.sort_values('abs_error_mean', ascending=False)
    
    return zipcode_summary