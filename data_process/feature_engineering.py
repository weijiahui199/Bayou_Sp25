"""
Feature engineering functions for the Bayou_Sp25 project.

This module contains functions to create features from the preprocessed data,
including lagged rainfall, cumulative rainfall, and temporal features.
"""

import pandas as pd
import numpy as np
import os
import sys

# Add the project root to sys.path to allow imports from other modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import settings

def create_lagged_features(data, column='rainfall', group_col='Zipcode', lag_periods=3):
    """
    Create lagged features for a specific column.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input DataFrame
    column : str
        Name of the column to create lags for (default: 'rainfall')
    group_col : str
        Name of the grouping column (default: 'Zipcode')
    lag_periods : int
        Number of lag periods to create (default: 3)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added lagged features
    """
    if data is None:
        return None
        
    # Make a copy of the input data
    result = data.copy()
    
    # Sort data by group and date
    result = result.sort_values([group_col, 'date'])
    
    # Create lagged features
    for i in range(1, lag_periods + 1):
        lag_col_name = f"{column}_lag{i}"
        result[lag_col_name] = result.groupby(group_col)[column].shift(i)
        print(f"Created lagged feature: {lag_col_name}")
    
    return result

def create_cumulative_features(data, column='rainfall', group_col='Zipcode', windows=None):
    """
    Create cumulative sum features over different windows.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input DataFrame
    column : str
        Name of the column to create cumulative features for (default: 'rainfall')
    group_col : str
        Name of the grouping column (default: 'Zipcode')
    windows : list
        List of window sizes for cumulative calculation (default: [3, 7])
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added cumulative features
    """
    if data is None:
        return None
        
    if windows is None:
        windows = settings.RAINFALL_CUM_WINDOWS
        
    # Make a copy of the input data
    result = data.copy()
    
    # Sort data by group and date
    result = result.sort_values([group_col, 'date'])
    
    # Create cumulative features
    for window in windows:
        cum_col_name = f"{column}_cum{window}"
        result[cum_col_name] = result.groupby(group_col)[column].transform(
            lambda x: x.rolling(window=window, min_periods=1).sum()
        )
        print(f"Created cumulative feature: {cum_col_name}")
    
    return result

def create_temporal_features(data, date_col='date'):
    """
    Create temporal features from date column.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input DataFrame
    date_col : str
        Name of the date column (default: 'date')
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added temporal features
    """
    if data is None:
        return None
        
    # Make a copy of the input data
    result = data.copy()
    
    # Extract year and month
    result['year'] = result[date_col].dt.year
    result['month'] = result[date_col].dt.month
    print("Created temporal features: year, month")
    
    # Create season feature
    result['season'] = pd.cut(
        result['month'],
        bins=[0, 3, 6, 9, 12],
        labels=['Winter', 'Spring', 'Summer', 'Fall'],
        include_lowest=True
    )
    print("Created temporal feature: season")
    
    # Optional: Create day of week, week of year, etc.
    result['dayofweek'] = result[date_col].dt.dayofweek
    result['quarter'] = result[date_col].dt.quarter
    print("Created temporal features: dayofweek, quarter")
    
    return result

def scale_demographic_features(data):
    """
    Scale demographic features for better model interpretation.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input DataFrame with demographic variables
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with scaled demographic features
    """
    if data is None:
        return None
        
    # Make a copy of the input data
    result = data.copy()
    
    # Scale demographic variables
    if 'population' in result.columns:
        result['population_scaled'] = result['population'] / 1000
        print("Scaled population to thousands")
    
    if 'median_income' in result.columns:
        result['income_scaled'] = result['median_income'] / 1000
        print("Scaled median income to thousands")
    
    return result

def create_interaction_features(data):
    """
    Create interaction features between rainfall and demographic variables.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input DataFrame with rainfall and demographic variables
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added interaction features
    """
    if data is None:
        return None
        
    # Make a copy of the input data
    result = data.copy()
    
    # Define variables for interaction
    rainfall_vars = ['rainfall', 'rainfall_cum7']
    demographic_vars = ['income_scaled', 'population_scaled']
    
    # Create interaction features
    for rain_var in rainfall_vars:
        for demo_var in demographic_vars:
            if rain_var in result.columns and demo_var in result.columns:
                interaction_col = f"{rain_var}_x_{demo_var}"
                result[interaction_col] = result[rain_var] * result[demo_var]
                print(f"Created interaction feature: {interaction_col}")
    
    return result

def create_features(data):
    """
    Apply all feature engineering steps to the dataset.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Preprocessed input DataFrame
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with all engineered features
    """
    if data is None:
        print("Error: Cannot create features - input data is missing")
        return None
    
    print("Starting feature engineering process...")
    
    # Create lagged features
    data_with_lags = create_lagged_features(
        data, 
        column='rainfall', 
        lag_periods=settings.RAINFALL_LAG_DAYS
    )
    
    # Create cumulative features
    data_with_cum = create_cumulative_features(
        data_with_lags, 
        column='rainfall', 
        windows=settings.RAINFALL_CUM_WINDOWS
    )
    
    # Create temporal features
    data_with_temporal = create_temporal_features(data_with_cum)
    
    # Scale demographic features
    data_with_scaled = scale_demographic_features(data_with_temporal)
    
    # Create interaction features (optional)
    data_with_interactions = create_interaction_features(data_with_scaled)
    
    # Drop rows with missing values created during feature engineering
    result = data_with_interactions.dropna()
    print(f"Removed {len(data_with_interactions) - len(result)} rows with missing values")
    print(f"Final dataset has {len(result)} rows and {len(result.columns)} columns")
    
    return result

def prepare_model_data(resolution=1, data_type='overflow', include_interactions=False):
    """
    Complete data preparation pipeline including preprocessing and feature engineering.
    
    Parameters:
    -----------
    resolution : int
        Time resolution in days (default: 1)
    data_type : str
        Type of event data to preprocess ('overflow' or '311')
    include_interactions : bool
        Whether to include interaction features (default: False)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame ready for modeling
    """
    from data_process.preprocessors import preprocess_data
    
    # Preprocess data
    preprocessed_data = preprocess_data(resolution, data_type)
    
    if preprocessed_data is None:
        print("Error: Preprocessing failed")
        return None
    
    # Create features
    if include_interactions:
        model_data = create_features(preprocessed_data)
    else:
        # Skip interaction features
        model_data = create_lagged_features(preprocessed_data)
        model_data = create_cumulative_features(model_data)
        model_data = create_temporal_features(model_data)
        model_data = scale_demographic_features(model_data)
        model_data = model_data.dropna()
    
    print(f"Model data preparation complete: {len(model_data)} rows, {len(model_data.columns)} columns")
    return model_data