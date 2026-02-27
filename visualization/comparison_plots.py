"""
Comparison plotting functions for the Bayou_Sp25 project.

This module contains functions to visualize comparisons between different models,
time resolutions, and data types.
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

def plot_time_resolution_comparison(results_dict, metric='MSE', 
                                  title='Model Performance by Time Resolution',
                                  filename='time_resolution_comparison.png'):
    """
    Plot performance comparison across time resolutions.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary with time resolutions as keys and model results as values
        Each value should contain 'mse' and 'mae' keys
    metric : str
        Metric to plot: 'MSE', 'MAE', etc. (default: 'MSE')
    title : str
        Title for the plot (default: 'Model Performance by Time Resolution')
    filename : str
        Filename for saving the plot (default: 'time_resolution_comparison.png')
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with metric values used for plotting
    """
    if not results_dict:
        print("Error: Cannot plot time resolution comparison - results dictionary is empty")
        return None
    
    # Initialize lists for data
    resolutions = []
    metric_values = []
    metric_key = metric.lower()
    
    # Extract metric values for each resolution
    for resolution, result in results_dict.items():
        if metric_key in result:
            resolutions.append(str(resolution))
            metric_values.append(result[metric_key])
    
    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'Resolution': resolutions,
        metric: metric_values
    })
    
    # Set up figure
    plt.figure(figsize=settings.FIGURE_SIZE)
    
    # Create bar plot
    ax = sns.barplot(x='Resolution', y=metric, data=plot_df)
    
    # Add value labels on top of each bar
    for i, val in enumerate(metric_values):
        ax.text(i, val + 0.02 * max(metric_values), f'{val:.4f}', 
                ha='center', va='bottom', fontsize=9)
    
    # Customize plot
    plt.xlabel('Time Resolution (days)')
    plt.ylabel(metric)
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save figure
    fig_path = os.path.join(settings.FIGURE_DIR, filename)
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=settings.DPI, bbox_inches='tight')
    print(f"Time resolution comparison plot saved to {fig_path}")
    
    # Display the plot
    plt.show()
    
    return plot_df

def plot_data_type_comparison(results_dict, resolution, 
                            title='Model Performance Comparison',
                            filename='data_type_comparison.png'):
    """
    Plot performance comparison between overflow and 311 models.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary with data types as keys and model results as values
        Each value should be a dict with 'mse' and 'mae' keys
    resolution : str or int
        Time resolution for the comparison
    title : str
        Title for the plot (default: 'Model Performance Comparison')
    filename : str
        Filename for saving the plot (default: 'data_type_comparison.png')
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with metric values used for plotting
    """
    if not results_dict:
        print("Error: Cannot plot data type comparison - results dictionary is empty")
        return None
    
    # Extract metrics for each data type
    data_types = []
    mse_values = []
    mae_values = []
    
    for data_type, result in results_dict.items():
        if 'mse' in result and 'mae' in result:
            data_types.append(data_type)
            mse_values.append(result['mse'])
            mae_values.append(result['mae'])
    
    # Set up plot
    plt.figure(figsize=(10, 6))
    barWidth = 0.35
    r1 = [0, 1]  # Positions for MSE and MAE
    r2 = [x + barWidth for x in r1]
    
    # Create bars
    plt.bar(r1, [mse_values[0], mae_values[0]], width=barWidth, 
            label=data_types[0], color='blue', alpha=0.7)
    plt.bar(r2, [mse_values[1], mae_values[1]], width=barWidth, 
            label=data_types[1], color='red', alpha=0.7)
    
    # Add values on bars
    for i, val in enumerate([mse_values[0], mae_values[0]]):
        plt.text(r1[i], val + 0.01, f'{val:.4f}', ha='center')
    for i, val in enumerate([mse_values[1], mae_values[1]]):
        plt.text(r2[i], val + 0.01, f'{val:.4f}', ha='center')
    
    # Customize plot
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.title(f"{title} ({resolution}-day resolution)")
    plt.xticks([r + barWidth/2 for r in r1], ['MSE', 'MAE'])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save and show
    fig_path = os.path.join(settings.FIGURE_DIR, f"{filename.replace('.png', '')}_{resolution}day.png")
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=settings.DPI, bbox_inches='tight')
    print(f"Data type comparison plot saved to {fig_path}")
    
    plt.show()
    
    # Create output DataFrame
    plot_df = pd.DataFrame({
        'Metric': ['MSE', 'MAE'],
        data_types[0]: [mse_values[0], mae_values[0]],
        data_types[1]: [mse_values[1], mae_values[1]]
    })
    
    return plot_df

def plot_comprehensive_comparison(all_results, 
                                 title='Performance Comparison Across Resolutions',
                                 filename='comprehensive_comparison.png'):
    """
    Plot comprehensive comparison across all time resolutions.
    
    Parameters:
    -----------
    all_results : dict
        Dictionary with resolutions as keys and sub-dictionaries as values
        Each sub-dictionary should have data types as keys and metrics as values
    title : str
        Title for the plot (default: 'Performance Comparison Across Resolutions')
    filename : str
        Filename for saving the plot (default: 'comprehensive_comparison.png')
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with metric values used for plotting
    """
    if not all_results:
        print("Error: Cannot plot comprehensive comparison - results dictionary is empty")
        return None
    
    # Extract data for plotting
    resolutions = list(all_results.keys())
    data_types = list(all_results[resolutions[0]].keys())
    
    # Create DataFrames for MSE and MAE
    mse_data = {
        resolution: [all_results[resolution][dt]['mse'] for dt in data_types]
        for resolution in resolutions
    }
    mae_data = {
        resolution: [all_results[resolution][dt]['mae'] for dt in data_types]
        for resolution in resolutions
    }
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Plot MSE comparison
    plt.subplot(1, 2, 1)
    for i, dt in enumerate(data_types):
        plt.plot(resolutions, [all_results[r][dt]['mse'] for r in resolutions], 
                 marker='o', label=dt)
    
    plt.xlabel('Resolution (days)')
    plt.ylabel('MSE')
    plt.title('MSE Comparison')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot MAE comparison
    plt.subplot(1, 2, 2)
    for i, dt in enumerate(data_types):
        plt.plot(resolutions, [all_results[r][dt]['mae'] for r in resolutions], 
                 marker='s', label=dt)
    
    plt.xlabel('Resolution (days)')
    plt.ylabel('MAE')
    plt.title('MAE Comparison')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save and show
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    
    fig_path = os.path.join(settings.FIGURE_DIR, filename)
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=settings.DPI, bbox_inches='tight')
    print(f"Comprehensive comparison plot saved to {fig_path}")
    
    plt.show()
    
    # Create output DataFrame
    mse_df = pd.DataFrame(mse_data, index=data_types)
    mae_df = pd.DataFrame(mae_data, index=data_types)
    
    result_df = pd.concat([
        mse_df.stack().reset_index().rename(columns={'level_0': 'Data_Type', 'level_1': 'Resolution', 0: 'MSE'}),
        mae_df.stack().reset_index().rename(columns={'level_0': 'Data_Type', 'level_1': 'Resolution', 0: 'MAE'})['MAE']
    ], axis=1)
    
    return result_df
def plot_zipcode_error_map(zipcode_performance, 
                          title='Error by Zipcode',
                          filename='zipcode_error_map.png'):
    """
    Plot a map of prediction errors by zipcode.
    
    Note: This function requires geopandas and a shapefile for the zipcodes.
    If these are not available, it will display an error message.
    
    Parameters:
    -----------
    zipcode_performance : pandas.DataFrame
        DataFrame with performance metrics by zipcode
    title : str
        Title for the plot (default: 'Error by Zipcode')
    filename : str
        Filename for saving the plot (default: 'zipcode_error_map.png')
        
    Returns:
    --------
    None
    """
    if zipcode_performance is None:
        print("Error: Cannot plot zipcode error map - data is missing")
        return
    
    try:
        import geopandas as gpd
        
        # Path to zipcode shapefile (would need to be adjusted based on actual file location)
        # This is a placeholder - you would need to provide the actual shapefile
        shapefile_path = os.path.join(settings.DATA_DIR, 'Zipcode_Shapefiles/zipcodes.shp')
        
        # Check if shapefile exists
        if not os.path.exists(shapefile_path):
            print(f"Warning: Zipcode shapefile not found at {shapefile_path}")
            print("Falling back to alternative visualization method")
            plot_zipcode_error_scatter(zipcode_performance, title, filename)
            return
        
        # Load zipcode shapefile
        zipcode_gdf = gpd.read_file(shapefile_path)
        
        # Ensure zipcode column is string type for joining
        zipcode_gdf['ZIPCODE'] = zipcode_gdf['ZIPCODE'].astype(str)
        
        # Merge with performance data
        zipcode_performance_reset = zipcode_performance.reset_index()
        merged_gdf = zipcode_gdf.merge(zipcode_performance_reset, 
                                      left_on='ZIPCODE', 
                                      right_on='Zipcode', 
                                      how='inner')
        
        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Plot the error map
        merged_gdf.plot(column='abs_error', 
                       cmap='YlOrRd', 
                       linewidth=0.8, 
                       ax=ax, 
                       edgecolor='black',
                       legend=True,
                       legend_kwds={'label': "Absolute Error",
                                    'orientation': "horizontal"})
        
        # Add zipcode labels
        for idx, row in merged_gdf.iterrows():
            plt.annotate(text=row['Zipcode'], 
                        xy=(row.geometry.centroid.x, row.geometry.centroid.y),
                        ha='center',
                        fontsize=8)
        
        # Customize plot
        ax.set_title(title)
        ax.set_axis_off()
        
        # Save and show
        fig_path = os.path.join(settings.FIGURE_DIR, filename)
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=settings.DPI, bbox_inches='tight')
        print(f"Zipcode error map saved to {fig_path}")
        
        plt.show()
        
    except ImportError:
        print("Warning: geopandas not installed. Cannot create zipcode map.")
        print("Falling back to alternative visualization method")
        plot_zipcode_error_scatter(zipcode_performance, title, filename)

def plot_zipcode_error_scatter(zipcode_performance, 
                              title='Error by Zipcode (Scatter)',
                              filename='zipcode_error_scatter.png'):
    """
    Alternative visualization for zipcode errors using scatter plots.
    
    Parameters:
    -----------
    zipcode_performance : pandas.DataFrame
        DataFrame with performance metrics by zipcode
    title : str
        Title for the plot (default: 'Error by Zipcode (Scatter)')
    filename : str
        Filename for saving the plot (default: 'zipcode_error_scatter.png')
        
    Returns:
    --------
    None
    """
    if zipcode_performance is None:
        print("Error: Cannot plot zipcode error scatter - data is missing")
        return
    
    # Set up figure with multiple panels
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Error vs Income
    axes[0, 0].scatter(zipcode_performance['median_income'], 
                      zipcode_performance['abs_error'], 
                      alpha=0.6)
    axes[0, 0].set_xlabel('Median Income')
    axes[0, 0].set_ylabel('Absolute Prediction Error')
    axes[0, 0].set_title('Income vs. Prediction Error')
    axes[0, 0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2: Error vs White Population
    axes[0, 1].scatter(zipcode_performance['percent_white'], 
                      zipcode_performance['abs_error'], 
                      alpha=0.6)
    axes[0, 1].set_xlabel('White Population (%)')
    axes[0, 1].set_ylabel('Absolute Prediction Error')
    axes[0, 1].set_title('White Population vs. Prediction Error')
    axes[0, 1].grid(True, linestyle='--', alpha=0.7)
    
    # Plot 3: Error vs Black Population
    axes[1, 0].scatter(zipcode_performance['percent_black'], 
                      zipcode_performance['abs_error'], 
                      alpha=0.6)
    axes[1, 0].set_xlabel('Black Population (%)')
    axes[1, 0].set_ylabel('Absolute Prediction Error')
    axes[1, 0].set_title('Black Population vs. Prediction Error')
    axes[1, 0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot 4: Error vs Population
    axes[1, 1].scatter(zipcode_performance['population'], 
                      zipcode_performance['abs_error'], 
                      alpha=0.6)
    axes[1, 1].set_xlabel('Population')
    axes[1, 1].set_ylabel('Absolute Prediction Error')
    axes[1, 1].set_title('Population vs. Prediction Error')
    axes[1, 1].grid(True, linestyle='--', alpha=0.7)
    
    # Add overall title
    fig.suptitle(title, fontsize=16)
    
    # Save and show
    fig_path = os.path.join(settings.FIGURE_DIR, filename)
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
    plt.savefig(fig_path, dpi=settings.DPI, bbox_inches='tight')
    print(f"Zipcode error scatter plot saved to {fig_path}")
    
    plt.show()

def plot_demographic_impact(zipcode_performance, 
                           title='Demographic Factors Impact on Prediction Error',
                           filename='demographic_impact.png'):
    """
    Create a regression plot showing the impact of demographic factors on prediction errors.
    
    Parameters:
    -----------
    zipcode_performance : pandas.DataFrame
        DataFrame with performance metrics by zipcode
    title : str
        Title for the plot (default: 'Demographic Factors Impact on Prediction Error')
    filename : str
        Filename for saving the plot (default: 'demographic_impact.png')
        
    Returns:
    --------
    None
    """
    if zipcode_performance is None:
        print("Error: Cannot plot demographic impact - data is missing")
        return
    
    # Set up a multi-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Demographic variables to analyze
    demographic_vars = [
        ('median_income', 'Median Income'),
        ('percent_white', 'White Population (%)'),
        ('percent_black', 'Black Population (%)'),
        ('population', 'Population')
    ]
    
    # Create regression plots for each demographic variable
    for i, (var, label) in enumerate(demographic_vars):
        if var in zipcode_performance.columns:
            sns.regplot(x=var, y='abs_error', data=zipcode_performance, 
                      ax=axes[i], scatter_kws={'alpha': 0.6})
            
            # Add correlation coefficient
            corr = zipcode_performance[var].corr(zipcode_performance['abs_error'])
            axes[i].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                       transform=axes[i].transAxes,
                       fontsize=10, 
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            axes[i].set_xlabel(label)
            axes[i].set_ylabel('Absolute Prediction Error')
            axes[i].set_title(f'{label} vs. Prediction Error')
            axes[i].grid(True, linestyle='--', alpha=0.7)
    
    # Add overall title
    fig.suptitle(title, fontsize=16)
    
    # Save and show
    fig_path = os.path.join(settings.FIGURE_DIR, filename)
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
    plt.savefig(fig_path, dpi=settings.DPI, bbox_inches='tight')
    print(f"Demographic impact plot saved to {fig_path}")
    
    plt.show()