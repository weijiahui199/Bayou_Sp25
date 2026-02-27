"""
Coefficient plotting functions for the Bayou_Sp25 project.

This module contains functions to visualize model coefficients,
including coefficient plots with confidence intervals and significance indicators.
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

def plot_coefficients(model_result, plot_title='Poisson Model Coefficients', 
                     filename='poisson_coefficients.png', custom_names=None):
    """
    Plot model coefficients with confidence intervals.
    
    Parameters:
    -----------
    model_result : statsmodels.genmod.generalized_linear_model.GLMResultsWrapper
        Fitted model object
    plot_title : str
        Title for the plot (default: 'Poisson Model Coefficients')
    filename : str
        Filename for saving the plot (default: 'poisson_coefficients.png')
    custom_names : dict, optional
        Dictionary mapping variable names to display names
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with coefficient information used for plotting
    """
    if model_result is None:
        print("Error: Cannot plot coefficients - model result is missing")
        return None
    
    # Define variable categories
    rainfall_vars = ['rainfall', 'rainfall_lag1', 'rainfall_lag2', 'rainfall_lag3', 
                     'rainfall_cum3', 'rainfall_cum7']
    demographic_vars = ['income_scaled', 'population_scaled', 'percent_white', 
                        'percent_black', 'percent_asian', 'percent_other']
    plot_vars = rainfall_vars + demographic_vars
    
    # Extract model parameters
    params = model_result.params
    conf_int = model_result.conf_int(alpha=0.05)
    pvalues = model_result.pvalues
    
    # Filter variables
    filtered_vars = [var for var in plot_vars if var in params.index]
    
    # Create a DataFrame for plotting
    plot_df = pd.DataFrame({
        'Variable': filtered_vars,
        'Coefficient': [params[var] for var in filtered_vars],
        'CI_Lower': [conf_int.loc[var, 0] for var in filtered_vars],
        'CI_Upper': [conf_int.loc[var, 1] for var in filtered_vars],
        'P_Value': [pvalues[var] for var in filtered_vars],
        'Significant': [pvalues[var] < 0.05 for var in filtered_vars],
        'Category': ['Rainfall' if var in rainfall_vars else 'Demographics' for var in filtered_vars]
    })
    
    # Add display names if provided
    if custom_names is None:
        custom_names = settings.CUSTOM_VARIABLE_NAMES
        
    plot_df['Display_Name'] = plot_df['Variable'].map(lambda x: custom_names.get(x, x))
    
    # Sort by category and coefficient value
    plot_df = plot_df.sort_values('Coefficient', ascending=True)
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define colors for categories
    category_colors = settings.CATEGORY_COLORS
    
    # Plot each bar
    for i, (_, row) in enumerate(plot_df.iterrows()):
        # Determine color based on category and significance
        color = category_colors[row['Category']]
        alpha = 1.0 if row['Significant'] else 0.5
        
        # Plot the bar
        ax.barh(
            i, 
            row['Coefficient'],
            color=color,
            alpha=alpha,
            height=0.6
        )
        
        # Add error bars
        ax.plot(
            [row['CI_Lower'], row['CI_Upper']], 
            [i, i], 
            color='black',
            linestyle='-',
            linewidth=1.5
        )
        
        # Add caps to error bars
        ax.plot([row['CI_Lower']], [i], color='black', marker='|', markersize=8)
        ax.plot([row['CI_Upper']], [i], color='black', marker='|', markersize=8)
        
        # Add coefficient values as text
        ax.text(
            row['Coefficient'] + (0.005 if row['Coefficient'] >= 0 else -0.005),
            i + 0.3,  # Move text above the bar
            f"{row['Coefficient']:.4f} (p={row['P_Value']:.3f})",
            va='center',
            ha='left' if row['Coefficient'] >= 0 else 'right',
            fontsize=10,  
            fontweight='bold' if row['Significant'] else 'normal'  # Make significant values bold
        )
    
    # Add vertical line at zero
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    
    # Set labels
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df['Display_Name'])
    ax.set_xlabel('Coefficient Value')
    ax.set_title(plot_title)
    
    # Add grid
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Create legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', alpha=1.0, label='Rainfall (Significant)'),
        Patch(facecolor='blue', alpha=0.5, label='Rainfall (Non-significant)'),
        Patch(facecolor='green', alpha=1.0, label='Demographics (Significant)'),
        Patch(facecolor='green', alpha=0.5, label='Demographics (Non-significant)')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    # Save figure
    fig_path = os.path.join(settings.FIGURE_DIR, filename)
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=settings.DPI, bbox_inches='tight')
    print(f"Coefficient plot saved to {fig_path}")
    
    # Display the plot
    plt.show()
    
    return plot_df

def plot_coefficient_comparison(models_dict, coefficient, 
                               plot_title='Coefficient Comparison',
                               filename='coefficient_comparison.png'):
    """
    Compare a specific coefficient across multiple models.
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary of model results with model names as keys
    coefficient : str
        Name of the coefficient to compare
    plot_title : str
        Title for the plot (default: 'Coefficient Comparison')
    filename : str
        Filename for saving the plot (default: 'coefficient_comparison.png')
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with coefficient information used for plotting
    """
    if not models_dict:
        print("Error: Cannot plot coefficient comparison - models dictionary is empty")
        return None
    
    # Initialize lists for data
    model_names = []
    coefficients = []
    ci_lowers = []
    ci_uppers = []
    pvalues = []
    significants = []
    
    # Extract coefficient information from each model
    for model_name, model_result in models_dict.items():
        if coefficient in model_result.params.index:
            model_names.append(model_name)
            coefficients.append(model_result.params[coefficient])
            ci_lowers.append(model_result.conf_int().loc[coefficient, 0])
            ci_uppers.append(model_result.conf_int().loc[coefficient, 1])
            pvalues.append(model_result.pvalues[coefficient])
            significants.append(model_result.pvalues[coefficient] < 0.05)
    
    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'Model': model_names,
        'Coefficient': coefficients,
        'CI_Lower': ci_lowers,
        'CI_Upper': ci_uppers,
        'P_Value': pvalues,
        'Significant': significants
    })
    
    # Sort by coefficient value
    plot_df = plot_df.sort_values('Coefficient', ascending=True)
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each bar
    for i, (_, row) in enumerate(plot_df.iterrows()):
        # Determine color based on significance
        color = 'blue' if row['Significant'] else 'lightblue'
        
        # Plot the bar
        ax.barh(
            i, 
            row['Coefficient'],
            color=color,
            height=0.6
        )
        
        # Add error bars
        ax.plot(
            [row['CI_Lower'], row['CI_Upper']], 
            [i, i], 
            color='black',
            linestyle='-',
            linewidth=1.5
        )
        
        # Add caps to error bars
        ax.plot([row['CI_Lower']], [i], color='black', marker='|', markersize=8)
        ax.plot([row['CI_Upper']], [i], color='black', marker='|', markersize=8)
        
        # Add coefficient values as text
        ax.text(
            row['Coefficient'] + (0.005 if row['Coefficient'] >= 0 else -0.005),
            i,
            f"{row['Coefficient']:.4f} (p={row['P_Value']:.3f})",
            va='center',
            ha='left' if row['Coefficient'] >= 0 else 'right',
            fontsize=10,
            fontweight='bold' if row['Significant'] else 'normal'
        )
    
    # Add vertical line at zero
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    
    # Set labels
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df['Model'])
    ax.set_xlabel('Coefficient Value')
    ax.set_title(f"{plot_title}: {settings.CUSTOM_VARIABLE_NAMES.get(coefficient, coefficient)}")
    
    # Add grid
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Create legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', label='Significant (p < 0.05)'),
        Patch(facecolor='lightblue', label='Non-significant')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    # Save figure
    fig_path = os.path.join(settings.FIGURE_DIR, filename)
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=settings.DPI, bbox_inches='tight')
    print(f"Coefficient comparison plot saved to {fig_path}")
    
    # Display the plot
    plt.show()
    
    return plot_df