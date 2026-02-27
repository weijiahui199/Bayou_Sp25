"""
Data preprocessing functions for the Bayou_Sp25 project.

This module contains functions to clean, reshape, and merge datasets.
"""

import pandas as pd
import numpy as np
import os
import sys

# Add the project root to sys.path to allow imports from other modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import settings

def clean_zipcodes(df, zipcode_col='Zipcode'):
    """
    Remove rows with invalid zipcodes and standardize zipcode format.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with zipcode column
    zipcode_col : str
        Name of the zipcode column (default: 'Zipcode')
        
    Returns:
    --------
    pandas.DataFrame
        Cleaned DataFrame with valid zipcodes
    """
    if df is None:
        return None
        
    # Convert zipcode column to string if it's not already
    df[zipcode_col] = df[zipcode_col].astype(str)
    
    # Filter out invalid zipcodes
    valid_df = df[
        (df[zipcode_col] != '0') & 
        (df[zipcode_col] != '0.0') & 
        (df[zipcode_col] != 'nan') &
        (df[zipcode_col] != 'None') &
        (df[zipcode_col].str.strip() != '')
    ].copy()
    
    # Standardize zipcode format (remove any trailing .0)
    valid_df[zipcode_col] = valid_df[zipcode_col].str.replace('.0', '', regex=False)
    
    print(f"Removed {len(df) - len(valid_df)} rows with invalid zipcodes from {zipcode_col}")
    return valid_df

def reshape_to_long(df):
    """
    Convert data from wide format (dates as columns) to long format.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame in wide format with 'Zipcode' and date columns
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame in long format with 'Zipcode', 'date', 'date_period', and 'value' columns
    """
    if df is None:
        return None
        
    # Keep only Zipcode and date columns
    id_vars = ['Zipcode']
    
    # Filter out non-date columns (like 'Unnamed: 0')
    # Try to parse each column name as a date, skip those that fail
    date_cols = []
    for col in df.columns:
        if col != 'Zipcode':
            try:
                # Try to parse the column name as a date
                pd.to_datetime(col.split('/')[0])
                date_cols.append(col)
            except:
                print(f"Skipping non-date column: {col}")
    
    if not date_cols:
        print("Error: No date columns found after filtering")
        return None
    
    # Melt the dataframe
    df_long = pd.melt(
        df, 
        id_vars=id_vars, 
        value_vars=date_cols,
        var_name='date_period', 
        value_name='value'
    )
    
    # Extract date from period string
    df_long['date'] = df_long['date_period'].str.split('/').str[0]
    
    # Convert to datetime with error handling
    try:
        df_long['date'] = pd.to_datetime(df_long['date'])
    except Exception as e:
        print(f"Error converting dates: {str(e)}")
        # Try to convert with more explicit format
        try:
            df_long['date'] = pd.to_datetime(df_long['date'], format='%Y-%m-%d')
            print("Successfully converted dates using explicit format")
        except Exception as e2:
            print(f"Error with explicit format: {str(e2)}")
            print("Sample date strings:")
            print(df_long['date'].head(10).tolist())
            return None
    
    return df_long

def merge_datasets(rainfall_long, event_long, demographics=None):
    """
    Merge rainfall, event, and demographic datasets.
    
    Parameters:
    -----------
    rainfall_long : pandas.DataFrame
        Rainfall data in long format
    event_long : pandas.DataFrame
        Event data (overflow or 311 calls) in long format
    demographics : pandas.DataFrame, optional
        Demographic data by zipcode
        
    Returns:
    --------
    pandas.DataFrame
        Merged dataset with rainfall, event, and demographic data
    """
    if rainfall_long is None or event_long is None:
        print("Error: Cannot merge datasets - input data is missing")
        return None
    
    # Determine the event type from column names
    event_type = 'overflow' if 'overflow' in event_long.columns else '311_calls'
    
    # Rename value column in event data if needed
    if 'value' in event_long.columns:
        event_long = event_long.rename(columns={'value': event_type})
    
    # Merge rainfall and event data
    merged_data = pd.merge(
        rainfall_long[['Zipcode', 'date', 'date_period', 'rainfall']], 
        event_long[['Zipcode', 'date', event_type]], 
        on=['Zipcode', 'date'], 
        how='inner'
    )
    
    # Merge with demographics if provided
    if demographics is not None:
        merged_data = pd.merge(
            merged_data,
            demographics,
            on='Zipcode',
            how='left'
        )
    
    # Drop rows with missing values
    merged_data = merged_data.dropna()
    
    print(f"Merged dataset has {len(merged_data)} rows and {len(merged_data.columns)} columns")
    return merged_data

def preprocess_data(resolution=1, data_type='overflow'):
    """
    Complete preprocessing pipeline for a specific resolution and data type.
    
    Parameters:
    -----------
    resolution : int
        Time resolution in days (default: 1)
    data_type : str
        Type of event data to preprocess ('overflow' or '311')
        
    Returns:
    --------
    pandas.DataFrame
        Preprocessed dataset ready for feature engineering
    """
    from data_process.loaders import load_all_data
    
    # Load data
    rainfall_data, event_data, demographics = load_all_data(resolution, data_type)
    
    if rainfall_data is None or event_data is None or demographics is None:
        print("Error: Cannot preprocess data - one or more datasets are missing")
        return None
    
    # Print column names for debugging
    print("\nRainfall data columns:", rainfall_data.columns.tolist())
    print("Event data columns:", event_data.columns.tolist())
    
    # Clean zipcodes
    rainfall_data_clean = clean_zipcodes(rainfall_data)
    event_data_clean = clean_zipcodes(event_data)
    demographics_clean = clean_zipcodes(demographics)
    
    # Reshape to long format
    rainfall_long = reshape_to_long(rainfall_data_clean)
    if rainfall_long is None:
        print("Error: Failed to reshape rainfall data")
        return None
        
    rainfall_long.rename(columns={'value': 'rainfall'}, inplace=True)
    
    event_long = reshape_to_long(event_data_clean)
    if event_long is None:
        print("Error: Failed to reshape event data")
        return None
        
    event_type = 'overflow' if data_type.lower() == 'overflow' else '311_calls'
    event_long.rename(columns={'value': event_type}, inplace=True)
    
    # Merge datasets
    merged_data = merge_datasets(rainfall_long, event_long, demographics_clean)
    
    return merged_data