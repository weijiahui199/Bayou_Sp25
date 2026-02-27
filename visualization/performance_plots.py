"""
Performance plotting functions for the Bayou_Sp25 project.

This module contains functions to visualize model performance,
including actual vs. predicted plots and error distributions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add the project root to sys.path to allow imports from other modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import settings

def plot_actual_vs_predicted(actual, predicted, title='Actual vs Predicted', 
                            filename='actual_vs_predicted.png'):
    """
    Plot actual vs predicted values.
    
    Parameters:
    -----------
    actual : array-like
        Actual values
    predicted : array-like
        Predicted values
    title : str
        Title for the plot (default: 'Actual vs Predicted')
    filename : str
        Filename for saving the plot (default: 'actual_vs_predicted.png')
        
    Returns:
    --------
    None
    """
    if actual is None or predicted is None:
        print("Error: Cannot plot actual vs predicted - data is missing")
        return
    
    # Set up figure
    plt.figure(figsize=settings.FIGURE_SIZE)
    
    # Scatter plot of actual vs predicted
    plt.scatter(actual, predicted, alpha=0.3)
    
    # Add perfect prediction line
    max_val = max(actual.max(), predicted.max())
    plt.plot([0, max_val], [0, max_val], 'r--')
    
    # Customize plot
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save figure
    fig_path = os.path.join(settings.FIGURE_DIR, filename)
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=settings.DPI, bbox_inches='tight')
    print(f"Actual vs predicted plot saved to {fig_path}")
    
    # Display the plot
    plt.show()

def plot_error_distribution(errors, title='Error Distribution', 
                           filename='error_distribution.png'):
    """
    Plot distribution of prediction errors.
    
    Parameters:
    -----------
    errors : array-like
        Prediction errors (actual - predicted)
    title : str
        Title for the plot (default: 'Error Distribution')
    filename : str
        Filename for saving the plot (default: 'error_distribution.png')
        
    Returns:
    --------
    None
    """
    if errors is None:
        print("Error: Cannot plot error distribution - data is missing")
        return
    
    # Set up figure
    plt.figure(figsize=settings.FIGURE_SIZE)
    
    # Create histogram of errors
    sns.histplot(errors, kde=True)
    
    # Add vertical line at zero
    plt.axvline(0, color='r', linestyle='--')
    
    # Add summary statistics
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    
    plt.text(0.95, 0.95, 
             f"Mean: {mean_error:.4f}\nMedian: {median_error:.4f}",
             transform=plt.gca().transAxes,
             ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Customize plot
    plt.xlabel('Error (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save figure
    fig_path = os.path.join(settings.FIGURE_DIR, filename)
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=settings.DPI, bbox_inches='tight')
    print(f"Error distribution plot saved to {fig_path}")
    
    # Display the plot
    plt.show()

def plot_performance_metrics(metrics_df, title='Performance Metrics Comparison', 
                           filename='performance_metrics.png'):
    """
    Plot performance metrics comparison.
    
    Parameters:
    -----------
    metrics_df : pandas.DataFrame
        DataFrame with performance metrics
    title : str
        Title for the plot (default: 'Performance Metrics Comparison')
    filename : str
        Filename for saving the plot (default: 'performance_metrics.png')
        
    Returns:
    --------
    None
    """
    if metrics_df is None or len(metrics_df) == 0:
        print("Error: Cannot plot performance metrics - data is missing")
        return
    
    # Set up figure
    plt.figure(figsize=settings.FIGURE_SIZE)
    
    # Select numeric columns for plotting
    metric_cols = metrics_df.select_dtypes(include=['float64', 'int64']).columns
    
    # Create heatmap of metrics
    sns.heatmap(metrics_df[metric_cols], annot=True, fmt='.4f', cmap='YlGnBu')
    
    # Customize plot
    plt.title(title)
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(settings.FIGURE_DIR, filename)
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=settings.DPI, bbox_inches='tight')
    print(f"Performance metrics plot saved to {fig_path}")
    
    # Display the plot
    plt.show()

def plot_residuals_vs_fitted(actual, predicted, title='Residuals vs Fitted Values', 
                            filename='residuals_vs_fitted.png'):
    """
    Plot residuals vs fitted values.
    
    Parameters:
    -----------
    actual : array-like
        Actual values
    predicted : array-like
        Predicted values
    title : str
        Title for the plot (default: 'Residuals vs Fitted Values')
    filename : str
        Filename for saving the plot (default: 'residuals_vs_fitted.png')
        
    Returns:
    --------
    None
    """
    if actual is None or predicted is None:
        print("Error: Cannot plot residuals vs fitted - data is missing")
        return
    
    # Calculate residuals
    residuals = actual - predicted
    
    # Set up figure
    plt.figure(figsize=settings.FIGURE_SIZE)
    
    # Scatter plot of residuals vs fitted
    plt.scatter(predicted, residuals, alpha=0.3)
    
    # Add horizontal line at zero
    plt.axhline(0, color='r', linestyle='--')
    
    # Add smoothed trendline
    try:
        # Use lowess smoothing if statsmodels is available
        from statsmodels.nonparametric.smoothers_lowess import lowess
        z = lowess(residuals, predicted)
        plt.plot(z[:, 0], z[:, 1], 'r-', linewidth=2)
    except ImportError:
        # Fall back to a simple moving average
        indices = np.argsort(predicted)
        sorted_pred = predicted[indices]
        sorted_resid = residuals[indices]
        window_size = max(10, len(predicted) // 20)
        smoothed = pd.Series(sorted_resid).rolling(window=window_size, center=True).mean()
        plt.plot(sorted_pred, smoothed, 'r-', linewidth=2)
    
    # Customize plot
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save figure
    fig_path = os.path.join(settings.FIGURE_DIR, filename)
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=settings.DPI, bbox_inches='tight')
    print(f"Residuals vs fitted plot saved to {fig_path}")
    
    # Display the plot
    plt.show()