"""
Configuration settings for the Bayou_Sp25 project.

This module contains all configuration parameters, file paths,
and settings used throughout the project.
"""

import os
from pathlib import Path


# Base directories
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "Data" 
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Data directories
RAINFALL_DATA_DIR = DATA_DIR / "Rainfall Data by Zipcode/rainfall_summary"
EVENT_COUNT_DIR = DATA_DIR / "Event Count Data by Zipcode/event_count_summary"
DEMOGRAPHICS_PATH = DATA_DIR / "Demographic Data by Zipcode/race and income by zipcode.csv"

# Data directories
RAINFALL_DATA_DIR = DATA_DIR / "Rainfall Data by Zipcode/rainfall_summary"
EVENT_COUNT_DIR = DATA_DIR / "Event Count Data by Zipcode/event_count_summary"
DEMOGRAPHICS_PATH = DATA_DIR / "Demographic Data by Zipcode/race and income by zipcode.csv"

ACTUAL_RAW_PRIVATE_EVENTS_PATH = DATA_DIR / "BCW Public and Private Original Data/Public and Private csv/combined_private_data.csv"
ACTUAL_RAW_PUBLIC_EVENTS_PATH = DATA_DIR / "BCW Public and Private Original Data/Public and Private csv/all_public_data.csv"

# Define date column names in raw files
PRIVATE_EVENTS_DATE_COL = "Date of WW Release"
PUBLIC_EVENTS_DATE_COL = "Start_Date"

# Raw data paths for data aggregation
RAW_PUBLIC_EVENTS_PATH = DATA_DIR / "Daily_Event_Counts/public_daily_events_by_zipcode.csv"
RAW_PRIVATE_EVENTS_PATH = DATA_DIR / "Daily_Event_Counts/private_daily_events_by_zipcode.csv"
RAW_311_CALLS_PATH = DATA_DIR / "Daily_Event_Counts/formatted_311_daily_events_by_zipcode.csv"
SITE_LOCATIONS_PATH = DATA_DIR / "WeatherData/Site Locations.csv"
RAW_RAINFALL_PATH = DATA_DIR / "Rainfall Complete/rainfall_complete.csv"

RAW_RAINFALL_2223_PATH = DATA_DIR / "WeatherData/2022.0101 thru 2023.0101_cleaned.csv"
RAW_RAINFALL_2324_PATH = DATA_DIR / "WeatherData/2023.0101 thru 2024.0701_cleaned.csv"

INTERMEDIATE_RAINFALL_PATH = DATA_DIR / "Rainfall Complete/rainfall_complete.csv"

# Date range for aggregation
AGGREGATION_START_DATE = '2021-01-01'
AGGREGATION_END_DATE = '2024-12-31'

# Output directories
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"
RESULTS_DIR = OUTPUT_DIR / "results"

# Ensure output directories exist
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# File naming patterns
RAINFALL_FILE_PATTERN = "rainfall_summary_{}.csv"
OVERFLOW_FILE_PATTERN = "overflow_summary_{}.csv"
CALLS311_FILE_PATTERN = "311_summary_{}.csv"

# Model parameters
TIME_RESOLUTIONS = [1, 2, 3, 7]  # Days per group for analysis
MODEL_FORMULA = (
    'overflow ~ rainfall + rainfall_lag1 + rainfall_cum7 + '
    'income_scaled + population_scaled + percent_white + percent_black + percent_other + '
    'C(season) + C(year)'
)
CALLS311_MODEL_FORMULA = MODEL_FORMULA.replace('overflow', '311_calls')

# Feature engineering parameters
RAINFALL_LAG_DAYS = 1  # Number of lag days for rainfall
RAINFALL_CUM_WINDOWS = [3, 7]  # Cumulative rainfall windows

# Cross-validation parameters
CV_FOLDS = 5
USE_CV_FOLD = 5  # Specifically use the 5th fold

# Visualization settings
FIGURE_SIZE = (10, 6)
FIGURE_SIZE_LARGE = (12, 8)
DPI = 300

# Variable category mapping
VARIABLE_CATEGORIES = {
    'rainfall': 'Rainfall',
    'rainfall_lag1': 'Rainfall',
    'rainfall_lag2': 'Rainfall',
    'rainfall_lag3': 'Rainfall',
    'rainfall_cum3': 'Rainfall',
    'rainfall_cum7': 'Rainfall',
    'income_scaled': 'Demographics',
    'population_scaled': 'Demographics',
    'percent_white': 'Demographics',
    'percent_black': 'Demographics',
    'percent_asian': 'Demographics',
    'percent_other': 'Demographics'
}

# Custom variable display names
CUSTOM_VARIABLE_NAMES = {
    'rainfall': 'Current Date Rainfall',
    'rainfall_lag1': 'Rainfall With 1 Day Lag',
    'rainfall_lag2': 'Rainfall With 2 Day Lag',
    'rainfall_lag3': 'Rainfall With 3 Day Lag',
    'rainfall_cum3': '3-Day Cumulative Rainfall',
    'rainfall_cum7': '7-Day Cumulative Rainfall',
    'percent_white': 'White Population (%)',
    'percent_black': 'Black Population (%)',
    'percent_asian': 'Asian Population (%)',
    'percent_other': 'Other Races (%)',
    'income_scaled': 'Median Income (Thousands)',
    'population_scaled': 'Population (Thousands)'
}

# Color settings for visualization
CATEGORY_COLORS = {
    'Rainfall': 'blue',
    'Demographics': 'green',
    'Season': 'purple',
    'Year': 'orange'
}