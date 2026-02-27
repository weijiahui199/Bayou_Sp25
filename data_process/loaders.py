"""
Data loading functions for the Bayou_Sp25 project.

This module contains functions to load rainfall, overflow, 311 call,
and demographic data from various sources.
"""

import pandas as pd
from pathlib import Path
import os
import sys

# Add the project root to sys.path to allow imports from other modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import settings

def load_rainfall_data(resolution=1):
    """
    Load rainfall data for a specific time resolution.
    
    Parameters:
    -----------
    resolution : int
        Time resolution in days (default: 1)
        
    Returns:
    --------
    pandas.DataFrame
        Rainfall data with zipcode and daily measurements
    """
    file_path = settings.RAINFALL_DATA_DIR / settings.RAINFALL_FILE_PATTERN.format(resolution)
    print(f"Loading rainfall data from: {file_path}")
    
    try:
        rainfall_data = pd.read_csv(file_path)
        rainfall_data['Zipcode'] = rainfall_data['Zipcode'].astype(str)
        print(f"Successfully loaded rainfall data: {len(rainfall_data)} records")
        return rainfall_data
    except FileNotFoundError:
        print(f"Error: Rainfall data file not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading rainfall data: {str(e)}")
        return None

def load_overflow_data(resolution=1):
    """
    Load sewer overflow data for a specific time resolution.
    
    Parameters:
    -----------
    resolution : int
        Time resolution in days (default: 1)
        
    Returns:
    --------
    pandas.DataFrame
        Overflow event data with zipcode and daily counts
    """
    file_path = settings.EVENT_COUNT_DIR / settings.OVERFLOW_FILE_PATTERN.format(resolution)
    print(f"Loading overflow data from: {file_path}")
    
    try:
        overflow_data = pd.read_csv(file_path)
        overflow_data['Zipcode'] = overflow_data['Zipcode'].astype(str)
        print(f"Successfully loaded overflow data: {len(overflow_data)} records")
        return overflow_data
    except FileNotFoundError:
        print(f"Error: Overflow data file not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading overflow data: {str(e)}")
        return None

def load_311_data(resolution=1):
    """
    Load 311 call data for a specific time resolution.
    
    Parameters:
    -----------
    resolution : int
        Time resolution in days (default: 1)
        
    Returns:
    --------
    pandas.DataFrame
        311 call data with zipcode and daily counts
    """
    file_path = settings.EVENT_COUNT_DIR / settings.CALLS311_FILE_PATTERN.format(resolution)
    print(f"Loading 311 call data from: {file_path}")
    
    try:
        calls_311_data = pd.read_csv(file_path)
        calls_311_data['Zipcode'] = calls_311_data['Zipcode'].astype(str)
        print(f"Successfully loaded 311 call data: {len(calls_311_data)} records")
        return calls_311_data
    except FileNotFoundError:
        print(f"Error: 311 call data file not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading 311 call data: {str(e)}")
        return None

def load_demographics():
    """
    Load demographic data by zipcode.
    
    Returns:
    --------
    pandas.DataFrame
        Demographic data with zipcode as index
    """
    file_path = settings.DEMOGRAPHICS_PATH
    print(f"Loading demographic data from: {file_path}")
    
    try:
        demographics = pd.read_csv(file_path)
        
        # Clean demographic data
        demographics_clean = demographics.rename(columns={
            'zip code': 'Zipcode',
            'Median earnings (dollars)': 'median_income',
            'total population': 'population',
            'White (%)': 'percent_white',
            'Black or African American (%)': 'percent_black',
            'Asian (%)': 'percent_asian',
            'Some Other Race (%)': 'percent_other'
        })
        
        demographics_clean['Zipcode'] = demographics_clean['Zipcode'].astype(str)
        print(f"Successfully loaded demographic data: {len(demographics_clean)} records")
        return demographics_clean
    except FileNotFoundError:
        print(f"Error: Demographic data file not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading demographic data: {str(e)}")
        return None

def load_all_data(resolution=1, data_type='overflow'):
    """
    Load all required datasets for a specific resolution and data type.
    
    Parameters:
    -----------
    resolution : int
        Time resolution in days (default: 1)
    data_type : str
        Type of event data to load ('overflow' or '311')
        
    Returns:
    --------
    tuple
        (rainfall_data, event_data, demographics_data)
    """
    rainfall_data = load_rainfall_data(resolution)
    demographics_data = load_demographics()
    
    if data_type.lower() == 'overflow':
        event_data = load_overflow_data(resolution)
    elif data_type.lower() in ['311', '311_calls']:
        event_data = load_311_data(resolution)
    else:
        print(f"Error: Unknown data_type '{data_type}'. Use 'overflow' or '311'.")
        event_data = None
    
    return rainfall_data, event_data, demographics_data