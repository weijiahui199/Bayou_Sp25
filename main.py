"""
Main script for the Bayou_Sp25 project.

This script demonstrates how to use the various modules to
perform the complete analysis pipeline.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import project modules
from config import settings
from data_process.loaders import load_all_data
from data_process.preprocessors import preprocess_data
from data_process.feature_engineering import create_features, prepare_model_data
from utils.time_series import get_fifth_fold_data
from models.poisson_model import build_poisson_model, get_significant_variables, save_model
from models.model_evaluation import evaluate_model, compare_models
from utils.zipcode_analysis import identify_best_worst_zipcodes, analyze_demographic_effect_on_errors
from visualization.coefficient_plots import plot_coefficients
from visualization.performance_plots import plot_actual_vs_predicted, plot_error_distribution
from visualization.comparison_plots import plot_time_resolution_comparison, plot_comprehensive_comparison
from utils.logging import get_results_logger, save_model_results, save_comparison_results

def run_single_resolution_analysis(resolution=1, data_type='overflow'):
    """
    Run complete analysis for a single resolution and data type.
    
    Parameters:
    -----------
    resolution : int
        Time resolution in days (default: 1)
    data_type : str
        Type of event data to analyze ('overflow' or '311')
        
    Returns:
    --------
    dict
        Dictionary with model results and performance metrics
    """
    # Set up logger
    logger = get_results_logger(data_type, resolution)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Running analysis for {data_type} data with {resolution}-day resolution")
    logger.info(f"{'='*80}")
    
    # Prepare model data
    model_data = prepare_model_data(resolution, data_type)
    
    if model_data is None:
        logger.info("Error: Failed to prepare model data")
        return None
    
    # Split into train and test sets using time series CV
    cv_data = get_fifth_fold_data(model_data)
    
    if cv_data is None:
        logger.info("Error: Failed to create CV folds")
        return None
    
    train_data = cv_data['train_data']
    test_data = cv_data['test_data']
    
    # Determine response variable
    response_var = 'overflow' if data_type == 'overflow' else '311_calls'
    
    # Build and fit Poisson model
    poisson_result = build_poisson_model(train_data, response_var)
    
    if poisson_result is None:
        logger.info("Error: Failed to build Poisson model")
        return None
    
    # Get significant variables
    sig_vars = get_significant_variables(poisson_result)
    
    # Plot coefficients
    plot_df = plot_coefficients(
        poisson_result,
        plot_title=f'Fixed Effects Poisson Model: {data_type.capitalize()} ({resolution}-day)',
        filename=f'poisson_coefficient_plot_{data_type}_{resolution}day.png'
    )
    
    # Evaluate on training data
    train_predictions, train_error_metrics, train_accuracy_metrics = evaluate_model(
        poisson_result, train_data, response_var, 'Training'
    )
    
    # Evaluate on test data
    test_predictions, test_error_metrics, test_accuracy_metrics = evaluate_model(
        poisson_result, test_data, response_var, 'Test'
    )
    
    # Plot actual vs predicted for test data
    plot_actual_vs_predicted(
        test_predictions[response_var],
        test_predictions['predicted'],
        title=f'Test Set: Actual vs Predicted {data_type.capitalize()} Events ({resolution}-day)',
        filename=f'{data_type}_test_actual_vs_predicted_{resolution}day.png'
    )
    
    # Plot error distribution
    plot_error_distribution(
        test_predictions[response_var] - test_predictions['predicted'],
        title=f'Test Set: Error Distribution ({resolution}-day)',
        filename=f'{data_type}_test_error_dist_{resolution}day.png'
    )
    
    # Save the model
    model_filename = f'poisson_{data_type}_{resolution}day.pkl'
    save_model(poisson_result, model_filename)
    
    # Return results
    results = {
        'model_result': poisson_result,
        'train_error_metrics': train_error_metrics,
        'test_error_metrics': test_error_metrics,
        'train_accuracy_metrics': train_accuracy_metrics,
        'test_accuracy_metrics': test_accuracy_metrics,
        'significant_variables': sig_vars,
        'mse': test_error_metrics['MSE'],
        'mae': test_error_metrics['MAE']
    }
    
    # Save results to files
    save_model_results(results, data_type, resolution)
    
    return results

def run_comparison_analysis():
    """
    Run comparison analysis across all resolutions and data types.
    
    Returns:
    --------
    dict
        Dictionary with results for all models
    """
    # Set up logger
    logger = get_results_logger('comparison')
    
    logger.info("\n" + "="*80)
    logger.info("Running Comparison Analysis Across Resolutions and Data Types")
    logger.info("="*80)
    
    # Initialize results dictionary
    all_results = {}
    
    # Run analysis for each resolution
    for resolution in settings.TIME_RESOLUTIONS:
        resolution_results = {}
        
        # Analyze overflow data
        overflow_results = run_single_resolution_analysis(resolution, 'overflow')
        if overflow_results:
            resolution_results['overflow'] = overflow_results
        
        # Analyze 311 call data
        calls311_results = run_single_resolution_analysis(resolution, '311')
        if calls311_results:
            resolution_results['311_calls'] = calls311_results
        
        # Store resolution results
        if resolution_results:
            all_results[str(resolution)] = resolution_results
    
    # Generate comparison plots
    
    # Compare overflow models across resolutions
    overflow_results_by_resolution = {
        res: results['overflow']
        for res, results in all_results.items()
        if 'overflow' in results
    }
    
    if overflow_results_by_resolution:
        plot_time_resolution_comparison(
            {res: {'mse': results['mse']} for res, results in overflow_results_by_resolution.items()},
            metric='MSE',
            title='Overflow Model Performance by Time Resolution',
            filename='overflow_resolution_comparison_mse.png'
        )
        
        plot_time_resolution_comparison(
            {res: {'mae': results['mae']} for res, results in overflow_results_by_resolution.items()},
            metric='MAE',
            title='Overflow Model Performance by Time Resolution',
            filename='overflow_resolution_comparison_mae.png'
        )
    else:
        logger.info("Error: Cannot plot time resolution comparison for overflow - insufficient results")
    
    # Compare 311 models across resolutions
    calls311_results_by_resolution = {
        res: results['311_calls']
        for res, results in all_results.items()
        if '311_calls' in results
    }
    
    if calls311_results_by_resolution:
        plot_time_resolution_comparison(
            {res: {'mse': results['mse']} for res, results in calls311_results_by_resolution.items()},
            metric='MSE',
            title='311 Calls Model Performance by Time Resolution',
            filename='311_resolution_comparison_mse.png'
        )
    else:
        logger.info("Error: Cannot plot time resolution comparison for 311 calls - insufficient results")
    
    # Comprehensive comparison
    if all_results:
        plot_comprehensive_comparison(
            all_results,
            title='Performance Comparison Across Resolutions and Data Types',
            filename='comprehensive_comparison.png'
        )
    else:
        logger.info("Error: Cannot plot comprehensive comparison - insufficient results")
    
    # Save comparison results
    save_comparison_results(all_results)
    
    return all_results

def analyze_zipcode_performance(data_type='overflow', resolution=1):
    """
    Analyze model performance by zipcode.
    
    Parameters:
    -----------
    data_type : str
        Type of event data to analyze ('overflow' or '311')
    resolution : int
        Time resolution in days
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with performance metrics by zipcode
    """
    # Set up logger
    logger = get_results_logger(f"{data_type}_zipcode", resolution)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Zipcode Performance Analysis for {data_type} data ({resolution}-day)")
    logger.info(f"{'='*80}")
    
    # Load model
    model_filename = f'poisson_{data_type}_{resolution}day.pkl'
    model_path = os.path.join(settings.MODEL_DIR, model_filename)
    
    try:
        from models.poisson_model import load_model
        model_result = load_model(model_filename)
        
        if model_result is None:
            logger.info(f"Error: Could not load model from {model_path}")
            return None
    except Exception as e:
        logger.info(f"Error loading model: {str(e)}")
        return None
    
    # Prepare model data
    model_data = prepare_model_data(resolution, data_type)
    
    if model_data is None:
        logger.info("Error: Failed to prepare model data")
        return None
    
    # Get test data
    cv_data = get_fifth_fold_data(model_data)
    
    if cv_data is None:
        logger.info("Error: Failed to create CV folds")
        return None
    
    test_data = cv_data['test_data']
    
    # Make predictions
    from models.poisson_model import predict
    predictions = predict(model_result, test_data)
    
    # Analyze zipcode performance
    from utils.zipcode_analysis import create_zipcode_summary
    zipcode_summary = create_zipcode_summary(test_data, predictions)
    
    if zipcode_summary is None:
        logger.info("Error: Failed to create zipcode summary")
        return None
    
    # Identify best and worst performing zipcodes
    best_zipcodes, worst_zipcodes = identify_best_worst_zipcodes(zipcode_summary)
    
    # Analyze demographic effects
    demo_results = analyze_demographic_effect_on_errors(zipcode_summary)
    
    # Save results
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    summary_file = os.path.join(
        settings.RESULTS_DIR, 
        f'{data_type}_zipcode_summary_{resolution}day_{timestamp}.csv'
    )
    zipcode_summary.to_csv(summary_file)
    logger.info(f"Zipcode performance summary saved to {summary_file}")
    
    # Visualize zipcode performance
    from visualization.comparison_plots import plot_zipcode_error_scatter, plot_demographic_impact
    
    plot_zipcode_error_scatter(
        zipcode_summary,
        title=f'Zipcode Error Analysis: {data_type.capitalize()} ({resolution}-day)',
        filename=f'{data_type}_zipcode_error_scatter_{resolution}day.png'
    )
    
    plot_demographic_impact(
        zipcode_summary,
        title=f'Demographic Impact on Errors: {data_type.capitalize()} ({resolution}-day)',
        filename=f'{data_type}_demographic_impact_{resolution}day.png'
    )
    
    return zipcode_summary


def main():
    """
    Main function to execute the analysis pipeline.
    """
    # Set up output directories
    os.makedirs(settings.FIGURE_DIR, exist_ok=True)
    os.makedirs(settings.MODEL_DIR, exist_ok=True)
    os.makedirs(settings.RESULTS_DIR, exist_ok=True)
    
    # Run a single resolution analysis as a simple test
    # Uncomment this to run a single analysis
    results = run_single_resolution_analysis(resolution=5, data_type='overflow')
    
    # Run the full comparison analysis
    #all_results = run_comparison_analysis()
    
    # Analyze zipcode performance for the best model
    # Uncomment this to run zipcode analysis
    # zipcode_summary = analyze_zipcode_performance(data_type='overflow', resolution=7)
    
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()