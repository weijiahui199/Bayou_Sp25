"""
Poisson regression model implementation for the Bayou_Sp25 project.

This module contains functions to build, fit, and use Poisson regression models
for analyzing sewer overflow events and 311 calls.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pickle
import os
import sys

# Add the project root to sys.path to allow imports from other modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import settings

def create_model_formula(response_var='overflow', include_zipcode_fe=False, logger=None):
    """
    Create model formula string.
    
    Parameters:
    -----------
    response_var : str
        Response variable name (default: 'overflow')
    include_zipcode_fe : bool
        Whether to include zipcode fixed effects (default: False)
        
    Returns:
    --------
    str
        Model formula string
    """
    # Base formula from settings
    base_formula = settings.MODEL_FORMULA
    formula = base_formula.replace('overflow', response_var)
    
    if response_var == '311_calls':
        
        formula = ('311_calls ~ rainfall + rainfall_lag1 + rainfall_cum7 + '
                  'income_scaled + population_scaled + percent_white + percent_black + '
                  'percent_other + C(season) + C(year)')
    
    # Replace response variable if needed
    formula = base_formula.replace('overflow', response_var)
    
    # Add zipcode fixed effects if requested
    if include_zipcode_fe:
        formula = formula + ' + C(Zipcode)'
    
    if logger:
        logger.info(f"Model formula: {formula}")
    else:
        print(f"Model formula: {formula}")

    return formula

def build_poisson_model(train_data, response_var='overflow', include_zipcode_fe=False, logger=None):
    """
    Build and fit a Poisson regression model.
    
    Parameters:
    -----------
    train_data : pandas.DataFrame
        Training data for model fitting
    response_var : str
        Response variable name (default: 'overflow')
    include_zipcode_fe : bool
        Whether to include zipcode fixed effects (default: False)
        
    Returns:
    --------
    statsmodels.genmod.generalized_linear_model.GLMResultsWrapper
        Fitted model object
    """
    if train_data is None:
        message = "Error: Cannot build model - training data is missing"
        if logger:
            print(message)
            logger.info(message)
        else:
            print(message)
        return None
    
    # Create model formula
    formula = create_model_formula(response_var, include_zipcode_fe)
    
    # Create and fit the model
    message = "Fitting Fixed Effects Poisson Model..."
    if logger:
        print(message)
        logger.info(message)
    else:
        print(message)


    try:
        poisson_model = smf.glm(
            formula=formula,
            data=train_data,
            family=sm.families.Poisson()
        )
        
        # Fit the model with robust standard errors
        poisson_result = poisson_model.fit(cov_type='HC0')
        
        # Print model summary
        if logger:
            print("Fitting Fixed Effects Poisson Model...")
            logger.info("Fitting Fixed Effects Poisson Model...")
        else:
            print("Fitting Fixed Effects Poisson Model...")

        if logger:
            print(poisson_result.summary())
            logger.info(poisson_result.summary())
        else:
            print(poisson_result.summary())   
        
        # Calculate and display pseudo R-squared
        null_model = smf.glm(
            f'{response_var} ~ 1', 
            data=train_data,
            family=sm.families.Poisson()
        ).fit()
        
        pseudo_r2 = 1 - (poisson_result.llf / null_model.llf)
        print(f"Pseudo R-squared: {pseudo_r2:.4f}")
        
        return poisson_result
    
    except Exception as e:
        print(f"Error building Poisson model: {str(e)}")
        return None

def get_significant_variables(model_result, alpha=0.05, logger=None):
    """
    Extract significant variables from model results.
    
    Parameters:
    -----------
    model_result : statsmodels.genmod.generalized_linear_model.GLMResultsWrapper
        Fitted model object
    alpha : float
        Significance level (default: 0.05)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with significant variables and their coefficients
    """

    if model_result is None:
        print("Error: Cannot extract significant variables - model result is missing")
        return None
    
    # Extract model parameters
    params = model_result.params
    conf_int = model_result.conf_int(alpha=alpha)
    pvalues = model_result.pvalues
    
    # Create a DataFrame for significant variables
    sig_vars = pd.DataFrame({
        'Variable': params.index,
        'Coefficient': params.values,
        'P_Value': pvalues.values,
        'CI_Lower': conf_int.iloc[:, 0].values,
        'CI_Upper': conf_int.iloc[:, 1].values,
        'Significant': pvalues.values < alpha
    })
    
    # Filter for significant variables only
    sig_vars_only = sig_vars[sig_vars['Significant']].copy()
    
    # Sort by coefficient magnitude
    sig_vars_only = sig_vars_only.sort_values('Coefficient', ascending=False)
    
    
    if logger:
        print(f"Found {len(sig_vars_only)} significant variables at alpha={alpha}")
        logger.info(f"Found {len(sig_vars_only)} significant variables at alpha={alpha}")
    else:
        print(f"Found {len(sig_vars_only)} significant variables at alpha={alpha}")
    return sig_vars_only

def predict(model_result, data, logger=None):
    """
    Generate predictions from a fitted model.
    
    Parameters:
    -----------
    model_result : statsmodels.genmod.generalized_linear_model.GLMResultsWrapper
        Fitted model object
    data : pandas.DataFrame
        Data for prediction
        
    Returns:
    --------
    pandas.Series
        Predicted values
    """
    if model_result is None or data is None:
        message = "Error: Cannot make predictions - model or data is missing"
        if logger:
            logger.info(message)
        else:
            print(message)

        return None
    
    try:
        predictions = model_result.predict(data)
        return predictions
    except Exception as e:
        print(f"Error generating predictions: {str(e)}")
        return None

def save_model(model_result, filename, directory=None, logger=None):
    """
    Save model to disk.
    
    Parameters:
    -----------
    model_result : statsmodels.genmod.generalized_linear_model.GLMResultsWrapper
        Fitted model object
    filename : str
        Base filename for saved model
    directory : str or pathlib.Path, optional
        Directory to save model in (default: settings.MODEL_DIR)
        
    Returns:
    --------
    str
        Path to saved model file
    """
    if model_result is None:
        message = "Error: Cannot save model - model result is missing"
        if logger:
            logger.info(message)
        else:
            print(message)
        return None
    
    if directory is None:
        directory = settings.MODEL_DIR
    
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Create full file path
    file_path = os.path.join(directory, filename)
    
    try:
        # Save model using pickle
        with open(file_path, 'wb') as f:
            pickle.dump(model_result, f)
        
        if logger:
            logger.info(f"Model saved to {file_path}")
            print(f"Model saved to {file_path}")
        else:
            print(f"Model saved to {file_path}")
        return file_path
    
    except Exception as e:
        message = f"Error saving model: {str(e)}"
        if logger:
            print(message)
            logger.info(message)
        else:
            print(message)

        return None

def load_model(filename, directory=None, logger=None):
    """
    Load model from disk.
    
    Parameters:
    -----------
    filename : str
        Filename of saved model
    directory : str or pathlib.Path, optional
        Directory containing saved model (default: settings.MODEL_DIR)
        
    Returns:
    --------
    statsmodels.genmod.generalized_linear_model.GLMResultsWrapper
        Loaded model object
    """
    if directory is None:
        directory = settings.MODEL_DIR
    
    # Create full file path
    file_path = os.path.join(directory, filename)
    
    try:
        # Load model using pickle
        with open(file_path, 'rb') as f:
            model_result = pickle.load(f)
        
        if logger:
            logger.info(f"Model loaded from {file_path}")
            print(f"Model loaded from {file_path}")
        else:
            print(f"Model loaded from {file_path}")
        return model_result
    except FileNotFoundError:
        message = f"Error: Model file not found at {file_path}"
        if logger:
            print(message)
            logger.info(message)
        else:
            print(message)
        return None

    except Exception as e:
        message = f"Error loading model: {str(e)}"
        if logger:
            print(message)
            logger.info(message)
        else:
            print(message)
        return None