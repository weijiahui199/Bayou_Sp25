#!/usr/bin/env python3
"""
Periodic Data Processor for Bayou_Sp25 Project

This script aggregates raw daily data (rainfall, overflow events, 311 calls) into periodic summaries
based on a specified time resolution (days_per_group). These summaries are the input files required
by the main analysis pipeline (via data_process/loaders.py).

Usage:
    python scripts/generate_periodic_summaries.py -r RESOLUTION

    where RESOLUTION is the time aggregation in days (e.g., 1, 2, 3, 7)

Example:
    python scripts/generate_periodic_summaries.py -r 7
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

# Add the project root to sys.path to allow importing project modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from data_process.preprocessors import clean_zipcodes
from config import settings

def load_data():
    """
    Load and perform initial cleaning of the raw data files.
    
    Returns:
    --------
    tuple
        (rainfall_data_clean, overflow_merged_clean, formatted_311_clean, demographics_clean)
    """
    # Load site locations first (needed for both rainfall processing and zipcode mapping)
    site_locations = pd.read_csv(settings.SITE_LOCATIONS_PATH)
    site_locations = site_locations.assign(Zipcode=site_locations['Zipcode'].astype(str))
    
    # Process rainfall data from raw files
    rainfall_data = _process_raw_rainfall(
        settings.RAW_RAINFALL_2223_PATH,
        settings.RAW_RAINFALL_2324_PATH,
        settings.SITE_LOCATIONS_PATH
    )
    
    # Process raw overflow events data to pivot tables
    private_events = _process_raw_event_to_daily_pivot(
        settings.ACTUAL_RAW_PRIVATE_EVENTS_PATH, 
        settings.PRIVATE_EVENTS_DATE_COL
    )
    
    public_events = _process_raw_event_to_daily_pivot(
        settings.ACTUAL_RAW_PUBLIC_EVENTS_PATH, 
        settings.PUBLIC_EVENTS_DATE_COL
    )
    
    # Load formatted 311 calls data
    try:
        formatted_311 = pd.read_csv(settings.RAW_311_CALLS_PATH)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"311 data file not found at {settings.RAW_311_CALLS_PATH}. "
            f"Please make sure the file exists or update the path in config/settings.py."
        )
    
    # Load demographics data
    demographics = pd.read_csv(settings.DEMOGRAPHICS_PATH)
    
    # Fix column names in demographics
    demographics_cols = {
        'zip code': 'Zipcode',
        'Median earnings (dollars)': 'median_income',
        'total population': 'population',
        'White (%)': 'percent_white',
        'Black or African American (%)': 'percent_black',
        'Asian (%)': 'percent_asian',
        'Some Other Race (%)': 'percent_other'
    }
    
    demographics_clean = demographics[list(demographics_cols.keys())].copy()
    demographics_clean = demographics_clean.rename(columns=demographics_cols)
    demographics_clean['Zipcode'] = demographics_clean['Zipcode'].astype(str)
    
    # Fix 311 data
    formatted_311 = formatted_311.assign(Zipcode=formatted_311['Zip Code'].astype(str))
    
    
    # Clean zipcodes (using the imported function from data_process.preprocessors)
    rainfall_data_clean = clean_zipcodes(rainfall_data)
    overflow_merged = merge_overflow_data(public_events, private_events)
    overflow_merged_clean = clean_zipcodes(overflow_merged)
    formatted_311_clean = clean_zipcodes(formatted_311, zipcode_col='Zipcode')
    
    return (rainfall_data_clean, overflow_merged_clean, formatted_311_clean, demographics_clean)

def _process_raw_rainfall(raw_path_2223, raw_path_2324, site_locations_path):
    """
    Process raw rainfall data from two CSV files into a single daily rainfall dataset.
    
    Parameters:
    -----------
    raw_path_2223 : str or Path
        Path to 2022-2023 rainfall data CSV
    raw_path_2324 : str or Path
        Path to 2023-2024 rainfall data CSV
    site_locations_path : str or Path
        Path to site locations CSV with mapping info
        
    Returns:
    --------
    pandas.DataFrame
        Processed daily rainfall data with site information
    """
    print(f"Loading rainfall data from {raw_path_2223} and {raw_path_2324}")
    
    # Load the raw rainfall data files
    rainfall_2223 = pd.read_csv(raw_path_2223)
    rainfall_2324 = pd.read_csv(raw_path_2324)
    
    # Concatenate the datasets
    rainfall_combined = pd.concat([rainfall_2223, rainfall_2324])
    
    # Load site location data
    site_codes = pd.read_csv(site_locations_path)
    
    # Convert wide format to long format
    rainfall_long = rainfall_combined.melt(
        id_vars=['Begin time', 'End time'],
        var_name='Source Address',
        value_name='Rainfall'
    )
    
    # Ensure string type for joining
    rainfall_long['Source Address'] = rainfall_long['Source Address'].astype(str)
    site_codes['Source Address'] = site_codes['Source Address'].astype(str)
    
    # Merge rainfall data with site locations
    rainfall_complete = rainfall_long.merge(site_codes, on='Source Address')
    
    # Extract date from begin time
    rainfall_complete["Date"] = rainfall_complete["Begin time"].str.split().str[0]
    
    # Sum rainfall at each source address for each day
    daily_rainfall = rainfall_complete.groupby(["Date", "Source Address"], as_index=False).agg({
        "Date": 'first',
        "Latitude": 'first',
        "Longitude": 'first',
        "Zipcode": 'first',
        "Rainfall": 'sum'
    })
    
    print(f"Processed {len(rainfall_complete)} hourly records into {len(daily_rainfall)} daily records")
    
    # Optional: Save intermediate file if needed
    # Define the path using the settings module
    intermediate_rainfall_path = settings.INTERMEDIATE_RAINFALL_PATH
    print(f"Saving intermediate rainfall data to {intermediate_rainfall_path}")
    # Ensure the directory exists before saving
    os.makedirs(intermediate_rainfall_path.parent, exist_ok=True)
    daily_rainfall.to_csv(intermediate_rainfall_path, index=False)
    
    return daily_rainfall

def _process_raw_event_to_daily_pivot(raw_file_path, date_column_name):
    """
    Process raw event data file into a daily pivot table by zipcode.
    
    Parameters:
    -----------
    raw_file_path : str or Path
        Path to the raw events data file
    date_column_name : str
        Name of the date column in the raw file
        
    Returns:
    --------
    pandas.DataFrame
        Pivot table with Zipcode as index and dates as columns
    """
    print(f"Loading file: {raw_file_path}")
    events_df = pd.read_csv(raw_file_path)
    
    # Print available columns for debugging
    print(f"Available columns in {raw_file_path}:")
    for idx, col in enumerate(events_df.columns):
        print(f"  {idx}: '{col}'")
    
    # If the specified column doesn't exist, let's try to find a suitable date column
    if date_column_name not in events_df.columns:
        print(f"WARNING: Column '{date_column_name}' not found. Attempting to identify a date column...")
        
        # Look for columns with 'date' in the name (case insensitive)
        date_cols = [col for col in events_df.columns if 'date' in col.lower()]
        if date_cols:
            print(f"Found potential date columns: {date_cols}")
            date_column_name = date_cols[0]  # Use the first one
            print(f"Using '{date_column_name}' as the date column")
        else:
            # If we can't find any date column, show a preview of the data
            print("Could not identify a date column automatically. Here's a preview of the data:")
            print(events_df.head())
            raise ValueError(f"Could not find date column '{date_column_name}' or any column with 'date' in the name.")
    
    # Convert date column to datetime
    print(f"Converting column '{date_column_name}' to datetime")
    events_df['Date'] = pd.to_datetime(events_df[date_column_name], errors='coerce')
    
    # Check for NaT values that might indicate conversion problems
    nat_count = events_df['Date'].isna().sum()
    if nat_count > 0:
        print(f"WARNING: {nat_count} rows have invalid date values and will be excluded")
    
    # Drop rows with NaT dates
    events_df = events_df.dropna(subset=['Date'])
    
    # Ensure Zipcode column exists
    if 'Zipcode' not in events_df.columns:
        # Try to find a zipcode-like column
        zip_cols = [col for col in events_df.columns if any(x in col.lower() for x in ['zip', 'postal', 'code'])]
        if zip_cols:
            print(f"Found potential zipcode column: {zip_cols[0]}")
            events_df['Zipcode'] = events_df[zip_cols[0]]
        else:
            raise ValueError("No 'Zipcode' column found in the data")
    
    # Ensure Zipcode is a string
    events_df['Zipcode'] = events_df['Zipcode'].astype(str)
    
    print(f"Grouping by Zipcode and Date")
    # Group by Zipcode and Date, count events
    daily_zipcode_events = events_df.groupby(['Zipcode', 'Date']).size().reset_index(name='Event_Count')
    
    # Convert to period for better pivoting
    daily_zipcode_events['Date'] = daily_zipcode_events['Date'].dt.to_period('D')
    
    print(f"Creating pivot table")
    # Create pivot table
    event_pivot = daily_zipcode_events.pivot_table(
        index='Zipcode', 
        columns='Date',
        values='Event_Count', 
        aggfunc='sum', 
        fill_value=0
    )
    
    # Convert column names to strings
    event_pivot.columns = [str(col) for col in event_pivot.columns]
    
    # Reset index to make Zipcode a regular column
    event_pivot.reset_index(inplace=True)
    
    print(f"Processed {len(events_df)} events into {len(event_pivot)} zipcode entries")
    return event_pivot

def merge_overflow_data(public_data, private_data):
    """
    Merge public and private sewage overflow data.
    
    Parameters:
    -----------
    public_data : pandas.DataFrame
        Public overflow data
    private_data : pandas.DataFrame
        Private overflow data
        
    Returns:
    --------
    pandas.DataFrame
        Merged overflow data
    """
    # Get all columns except 'Zipcode' from both dataframes
    public_cols = [col for col in public_data.columns if col != 'Zipcode']
    private_cols = [col for col in private_data.columns if col != 'Zipcode']
    
    # Set up base dataframe with just Zipcode
    result_df = pd.DataFrame({'Zipcode': public_data['Zipcode']})
    
    # Get the union of all date columns
    all_overflow_cols = sorted(list(set(public_cols).union(set(private_cols))))
    
    # Prepare data for each dataset to be concatenated
    public_overflow = public_data[list(set(all_overflow_cols) & set(public_data.columns))].copy()
    private_overflow = private_data[list(set(all_overflow_cols) & set(private_data.columns))].copy()
    
    # For columns in both, we need to add values
    common_cols = list(set(public_cols) & set(private_cols))
    merged_common = public_data[common_cols].add(private_data[common_cols], fill_value=0)
    
    # For columns only in public
    public_only = list(set(public_cols) - set(private_cols))
    public_only_data = public_data[public_only].copy() if public_only else None
    
    # For columns only in private
    private_only = list(set(private_cols) - set(public_cols))
    private_only_data = private_data[private_only].copy() if private_only else None
    
    # Combine all parts using concat
    dfs_to_concat = [result_df]
    if not merged_common.empty:
        dfs_to_concat.append(merged_common)
    if public_only_data is not None and not public_only_data.empty:
        dfs_to_concat.append(public_only_data)
    if private_only_data is not None and not private_only_data.empty:
        dfs_to_concat.append(private_only_data)
    
    overflow_merged = pd.concat(dfs_to_concat, axis=1)
    
    return overflow_merged

def summarize_rainfall_by_period(rainfall_data, days_per_group=7):
    """
    Summarize rainfall data by grouping days together for each zipcode.
    
    Takes average of non-zero rainfall values within each period.
    
    Parameters:
    -----------
    rainfall_data : pandas.DataFrame
        Rainfall data with Date and Zipcode columns
    days_per_group : int
        Number of days to group together
        
    Returns:
    --------
    pandas.DataFrame
        Summarized rainfall data by period
    """
    # Convert rainfall data date column to datetime if it's not already
    rainfall_data['Date'] = pd.to_datetime(rainfall_data['Date'])
    
    # Create date ranges for periods
    min_date = pd.to_datetime(settings.AGGREGATION_START_DATE)
    max_date = pd.to_datetime(settings.AGGREGATION_END_DATE)
    
    # Generate period start dates
    period_starts = pd.date_range(start=min_date, end=max_date, freq=f'{days_per_group}D')
    
    # Create all zipcodes list
    all_zipcodes = sorted(rainfall_data['Zipcode'].unique())
    
    # Create a list to store period data
    period_data_list = []
    
    # Process each period
    for start_date in period_starts:
        end_date = start_date + pd.Timedelta(days=days_per_group-1)
        if end_date > max_date:
            end_date = max_date
            
        # Create period column name in the format 'start_date/end_date'
        period_name = f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
        
        # Filter data for this period
        period_mask = (rainfall_data['Date'] >= start_date) & (rainfall_data['Date'] <= end_date)
        period_data = rainfall_data[period_mask]
        
        # Calculate average non-zero rainfall by zipcode
        period_values = {}
        
        for zipcode in all_zipcodes:
            zipcode_data = period_data[period_data['Zipcode'] == zipcode]
            if len(zipcode_data) > 0:
                non_zero_data = zipcode_data[zipcode_data['Rainfall'] > 0]
                if len(non_zero_data) > 0:
                    # Calculate average of non-zero rainfall values
                    period_values[zipcode] = non_zero_data['Rainfall'].mean()
                else:
                    # All values are zero
                    period_values[zipcode] = 0
            else:
                # No data for this zipcode and period
                period_values[zipcode] = 0
        
        # Create a DataFrame for this period
        period_df = pd.DataFrame({
            'Zipcode': all_zipcodes,
            period_name: [period_values.get(zipcode, 0) for zipcode in all_zipcodes]
        })
        
        period_data_list.append(period_df)
    
    # Merge all period DataFrames
    result_df = period_data_list[0]
    for df in period_data_list[1:]:
        result_df = pd.merge(result_df, df, on='Zipcode', how='outer')
    
    return result_df

def summarize_overflow_by_period(overflow_data, days_per_group=7):
    """
    Summarize overflow data by grouping days together for each zipcode.
    
    Sums overflow counts within each period.
    
    Parameters:
    -----------
    overflow_data : pandas.DataFrame
        Overflow data with date columns and Zipcode
    days_per_group : int
        Number of days to group together
        
    Returns:
    --------
    pandas.DataFrame
        Summarized overflow data by period
    """
    # Create date ranges for periods
    min_date = pd.to_datetime(settings.AGGREGATION_START_DATE)
    max_date = pd.to_datetime(settings.AGGREGATION_END_DATE)
    
    # Generate period start dates
    period_starts = pd.date_range(start=min_date, end=max_date, freq=f'{days_per_group}D')
    
    # Create all zipcodes list
    all_zipcodes = sorted(overflow_data['Zipcode'].unique())
    
    # Get all date columns
    date_cols = [col for col in overflow_data.columns if col != 'Zipcode']
    
    # Create a list to store period data
    period_data_list = []
    
    # Process each period
    for start_date in period_starts:
        end_date = start_date + pd.Timedelta(days=days_per_group-1)
        if end_date > max_date:
            end_date = max_date
            
        # Create period column name in the format 'start_date/end_date'
        period_name = f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
        
        # Find date columns that fall within this period
        period_cols = []
        for col in date_cols:
            try:
                col_date = pd.to_datetime(col)
                if start_date <= col_date <= end_date:
                    period_cols.append(col)
            except:
                # Skip columns that can't be parsed as dates
                continue
        
        # Calculate total overflow by zipcode for this period
        period_values = {}
        
        for zipcode in all_zipcodes:
            zipcode_data = overflow_data[overflow_data['Zipcode'] == zipcode]
            if len(zipcode_data) > 0 and len(period_cols) > 0:
                # Sum overflow counts for all days in this period
                period_values[zipcode] = zipcode_data[period_cols].sum(axis=1).iloc[0]
            else:
                # No data for this zipcode and period
                period_values[zipcode] = 0
        
        # Create a DataFrame for this period
        period_df = pd.DataFrame({
            'Zipcode': all_zipcodes,
            period_name: [period_values.get(zipcode, 0) for zipcode in all_zipcodes]
        })
        
        period_data_list.append(period_df)
    
    # Merge all period DataFrames efficiently
    result_df = period_data_list[0]
    for df in period_data_list[1:]:
        result_df = pd.merge(result_df, df, on='Zipcode', how='outer')
    
    return result_df

def summarize_311_by_period(calls_311_data, days_per_group=7):
    """
    Summarize 311 call data by grouping days together for each zipcode.
    
    Sums 311 call counts within each period.
    
    Parameters:
    -----------
    calls_311_data : pandas.DataFrame
        311 call data with date columns and Zipcode
    days_per_group : int
        Number of days to group together
        
    Returns:
    --------
    pandas.DataFrame
        Summarized 311 call data by period
    """
    # Create date ranges for periods
    min_date = pd.to_datetime(settings.AGGREGATION_START_DATE)
    max_date = pd.to_datetime(settings.AGGREGATION_END_DATE)
    
    # Generate period start dates
    period_starts = pd.date_range(start=min_date, end=max_date, freq=f'{days_per_group}D')
    
    # Create all zipcodes list
    all_zipcodes = sorted(calls_311_data['Zipcode'].unique())
    
    # Get all date columns
    date_cols = [col for col in calls_311_data.columns if col != 'Zipcode']
    
    # Create a list to store period data
    period_data_list = []
    
    # Process each period
    for start_date in period_starts:
        end_date = start_date + pd.Timedelta(days=days_per_group-1)
        if end_date > max_date:
            end_date = max_date
            
        # Create period column name in the format 'start_date/end_date'
        period_name = f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
        
        # Find date columns that fall within this period
        period_cols = []
        for col in date_cols:
            try:
                col_date = pd.to_datetime(col)
                if start_date <= col_date <= end_date:
                    period_cols.append(col)
            except:
                # Skip columns that can't be parsed as dates
                continue
        
        # Calculate total 311 calls by zipcode for this period
        period_values = {}
        
        for zipcode in all_zipcodes:
            zipcode_data = calls_311_data[calls_311_data['Zipcode'] == zipcode]
            if len(zipcode_data) > 0 and len(period_cols) > 0:
                # Sum 311 call counts for all days in this period
                period_values[zipcode] = zipcode_data[period_cols].sum(axis=1).iloc[0]
            else:
                # No data for this zipcode and period
                period_values[zipcode] = 0
        
        # Create a DataFrame for this period
        period_df = pd.DataFrame({
            'Zipcode': all_zipcodes,
            period_name: [period_values.get(zipcode, 0) for zipcode in all_zipcodes]
        })
        
        period_data_list.append(period_df)
    
    # Merge all period DataFrames efficiently
    result_df = period_data_list[0]
    for df in period_data_list[1:]:
        result_df = pd.merge(result_df, df, on='Zipcode', how='outer')
    
    return result_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate periodic summary files from raw daily data.")
    parser.add_argument(
        "-r", "--resolution",
        type=int,
        required=True,
        help="Time resolution in days (e.g., 1, 2, 3, 7)."
    )
    args = parser.parse_args()
    days_per_group = args.resolution

    print(f"Starting summary generation for {days_per_group}-day resolution...")

    # 1. Load data using load_data()
    print("Loading and cleaning data...")
    (rainfall_data_clean, overflow_merged_clean,
     formatted_311_clean, demographics_clean) = load_data()

    # 2. Generate summaries using summarize functions, passing days_per_group
    print(f"Summarizing rainfall for {days_per_group}-day periods...")
    rainfall_summary = summarize_rainfall_by_period(rainfall_data_clean, days_per_group)

    print(f"Summarizing overflow for {days_per_group}-day periods...")
    overflow_summary = summarize_overflow_by_period(overflow_merged_clean, days_per_group)

    print(f"Summarizing 311 calls for {days_per_group}-day periods...")
    calls_311_summary = summarize_311_by_period(formatted_311_clean, days_per_group)

    # 3. Save the summaries
    # Ensure output directories exist (might be redundant if settings already does it)
    os.makedirs(settings.RAINFALL_DATA_DIR, exist_ok=True)
    os.makedirs(settings.EVENT_COUNT_DIR, exist_ok=True)

    # Define output file paths using settings
    rainfall_output_path = settings.RAINFALL_DATA_DIR / settings.RAINFALL_FILE_PATTERN.format(days_per_group)
    overflow_output_path = settings.EVENT_COUNT_DIR / settings.OVERFLOW_FILE_PATTERN.format(days_per_group)
    calls_311_output_path = settings.EVENT_COUNT_DIR / settings.CALLS311_FILE_PATTERN.format(days_per_group)

    # Save the dataframes
    print(f"Saving rainfall summary to {rainfall_output_path}")
    rainfall_summary.to_csv(rainfall_output_path, index=False)

    print(f"Saving overflow summary to {overflow_output_path}")
    overflow_summary.to_csv(overflow_output_path, index=False)

    print(f"Saving 311 calls summary to {calls_311_output_path}")
    calls_311_summary.to_csv(calls_311_output_path, index=False)

    print(f"Summary generation complete for {days_per_group}-day resolution.")