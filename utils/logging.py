"""
Logging utilities for the Bayou_Sp25 project.

This module contains functions for logging output to both console and files.
"""

import os
import sys
import logging
from datetime import datetime
import pandas as pd

# Add the project root to sys.path to allow imports from other modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import settings

def setup_logger(name, log_file, level=logging.INFO):
    """
    Set up a logger that writes to both console and file.
    
    Parameters:
    -----------
    name : str
        Name of the logger
    log_file : str
        Path to the log file
    level : logging level
        Logging level (default: logging.INFO)
        
    Returns:
    --------
    logging.Logger
        Configured logger object
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create file handler
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def get_results_logger(analysis_type='overflow', resolution=1):
    """
    Get a logger for analysis results.
    
    Parameters:
    -----------
    analysis_type : str
        Type of analysis ('overflow', '311', 'comparison')
    resolution : int
        Time resolution in days
        
    Returns:
    --------
    logging.Logger
        Configured logger object
    """
    # Create timestamp for unique log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Define log file path
    if analysis_type == 'comparison':
        log_file = os.path.join(settings.RESULTS_DIR, f'comparison_analysis_{timestamp}.log')
    else:
        log_file = os.path.join(settings.RESULTS_DIR, 
                              f'{analysis_type}_{resolution}day_{timestamp}.log')
    
    # Set up and return logger
    return setup_logger(f'{analysis_type}_{resolution}day', log_file)

def save_model_results(results_dict, analysis_type='overflow', resolution=1):
    """
    Save model results to CSV files.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary containing model results
    analysis_type : str
        Type of analysis ('overflow', '311')
    resolution : int
        Time resolution in days
        
    Returns:
    --------
    None
    """
    # Create timestamp for unique file names
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save significant variables
    if 'significant_variables' in results_dict and isinstance(results_dict['significant_variables'], pd.DataFrame):
        sig_vars_file = os.path.join(
            settings.RESULTS_DIR, 
            f'{analysis_type}_{resolution}day_significant_vars_{timestamp}.csv'
        )
        results_dict['significant_variables'].to_csv(sig_vars_file, index=False)
        print(f"Significant variables saved to {sig_vars_file}")
    
    # Save performance metrics
    metrics = {}
    
    if 'test_error_metrics' in results_dict:
        metrics.update({f'test_{k}': v for k, v in results_dict['test_error_metrics'].items()})
    
    if 'test_accuracy_metrics' in results_dict:
        metrics.update({f'test_{k}': v for k, v in results_dict['test_accuracy_metrics'].items()})
    
    if 'train_error_metrics' in results_dict:
        metrics.update({f'train_{k}': v for k, v in results_dict['train_error_metrics'].items()})
    
    if 'train_accuracy_metrics' in results_dict:
        metrics.update({f'train_{k}': v for k, v in results_dict['train_accuracy_metrics'].items()})
    
    if metrics:
        metrics_file = os.path.join(
            settings.RESULTS_DIR, 
            f'{analysis_type}_{resolution}day_metrics_{timestamp}.csv'
        )
        pd.DataFrame([metrics]).to_csv(metrics_file, index=False)
        print(f"Performance metrics saved to {metrics_file}")

def save_comparison_results(all_results):
    """
    Save comparison results to CSV files.
    
    Parameters:
    -----------
    all_results : dict
        Dictionary with results for all models
        
    Returns:
    --------
    None
    """
    # Create timestamp for unique file names
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Prepare data for summary table
    rows = []
    
    for resolution, results_by_type in all_results.items():
        for data_type, results in results_by_type.items():
            if 'test_error_metrics' in results and 'test_accuracy_metrics' in results:
                row = {
                    'Resolution': resolution,
                    'Data_Type': data_type,
                    'MSE': results['test_error_metrics']['MSE'],
                    'RMSE': results['test_error_metrics']['RMSE'],
                    'MAE': results['test_error_metrics']['MAE'],
                    'Exact_Match': results['test_accuracy_metrics']['Exact_Match'],
                    'Within_1': results['test_accuracy_metrics']['Within_1'],
                    'Within_2': results['test_accuracy_metrics']['Within_2']
                }
                rows.append(row)
    
    if rows:
        # Create summary DataFrame and save
        summary_df = pd.DataFrame(rows)
        summary_file = os.path.join(
            settings.RESULTS_DIR, 
            f'comparison_summary_{timestamp}.csv'
        )
        summary_df.to_csv(summary_file, index=False)
        print(f"Comparison summary saved to {summary_file}")