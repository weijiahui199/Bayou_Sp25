"""
Time series utilities for the Bayou_Sp25 project.

This module contains functions for time series cross-validation
and related time series operations.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import os
import sys

# Add the project root to sys.path to allow imports from other modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import settings

def create_time_series_folds(data, n_splits=5, date_col='date'):
    """
    Create time series cross-validation folds.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input DataFrame
    n_splits : int
        Number of folds for cross-validation (default: 5)
    date_col : str
        Name of the date column (default: 'date')
        
    Returns:
    --------
    list
        List of dictionaries containing train/test indices for each fold
    """
    if data is None:
        return None
        
    # Sort data by date
    data_sorted = data.sort_values(date_col).reset_index(drop=True)
    
    # Create TimeSeriesSplit object
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Generate folds
    folds = []
    for i, (train_idx, test_idx) in enumerate(tscv.split(data_sorted)):
        fold_dict = {
            'fold': i + 1,
            'train_idx': train_idx,
            'test_idx': test_idx,
            'train_min_date': data_sorted.iloc[train_idx][date_col].min(),
            'train_max_date': data_sorted.iloc[train_idx][date_col].max(),
            'test_min_date': data_sorted.iloc[test_idx][date_col].min(),
            'test_max_date': data_sorted.iloc[test_idx][date_col].max(),
            'train_size': len(train_idx),
            'test_size': len(test_idx)
        }
        folds.append(fold_dict)
        
    # Print fold information
    for fold in folds:
        print(f"Fold {fold['fold']}:")
        print(f"  Train: {fold['train_size']} observations, {fold['train_min_date']} to {fold['train_max_date']}")
        print(f"  Test: {fold['test_size']} observations, {fold['test_min_date']} to {fold['test_max_date']}")
        
    return folds

def get_fold_data(data, fold_index, n_splits=5, date_col='date'):
    """
    Extract train and test data for a specific fold.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input DataFrame
    fold_index : int
        Index of the fold to extract (1-based)
    n_splits : int
        Number of folds for cross-validation (default: 5)
    date_col : str
        Name of the date column (default: 'date')
        
    Returns:
    --------
    dict
        Dictionary containing train and test DataFrames
    """
    if data is None:
        return None
        
    if fold_index < 1 or fold_index > n_splits:
        print(f"Error: fold_index must be between 1 and {n_splits}")
        return None
    
    # Sort data by date
    data_sorted = data.sort_values(date_col).reset_index(drop=True)
    
    # Create TimeSeriesSplit object
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Find the specified fold
    for i, (train_idx, test_idx) in enumerate(tscv.split(data_sorted)):
        if i + 1 == fold_index:
            train_data = data_sorted.iloc[train_idx].copy()
            test_data = data_sorted.iloc[test_idx].copy()
            
            print(f"\n=== Time Series CV: Fold {fold_index} ===")
            print(f"Training set: {len(train_data)} observations")
            print(f"Test set: {len(test_data)} observations")
            print(f"Training date range: {train_data[date_col].min()} to {train_data[date_col].max()}")
            print(f"Test date range: {test_data[date_col].min()} to {test_data[date_col].max()}")
            
            return {
                'train_data': train_data,
                'test_data': test_data
            }
    
    # This should never happen if the fold_index is valid
    print("Error: Could not find the specified fold")
    return None

def get_fifth_fold_data(data, date_col='date'):
    """
    Extract train and test data for the 5th fold of time series cross-validation.
    
    This is a convenience function for the commonly used 5th fold in the analysis.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input DataFrame
    date_col : str
        Name of the date column (default: 'date')
        
    Returns:
    --------
    dict
        Dictionary containing train and test DataFrames
    """
    return get_fold_data(
        data, 
        fold_index=settings.USE_CV_FOLD, 
        n_splits=settings.CV_FOLDS, 
        date_col=date_col
    )