import matplotlib.pyplot as plt
import os
import pandas as pd
from utils.load_item_mapping import load_item_mapping

def align_dataframes(dataframes):
    """
    Aligns a list of DataFrames to have the same indices.

    Parameters:
    - dataframes (list of pd.DataFrame): List of DataFrames to be aligned.

    Returns:
    - aligned_dataframes (list of pd.DataFrame): List of DataFrames with aligned indices.
    """
    if not dataframes:
        return []

    # Find the intersection of all DataFrame indices
    common_indices = dataframes[0].index
    for df in dataframes[1:]:
        common_indices = common_indices.intersection(df.index)

    # Reindex each DataFrame to the common indices
    aligned_dataframes = [df.loc[common_indices] for df in dataframes]
    
    return aligned_dataframes

def aggregate_time_series(time_series, aggregation='D'):
    """
    Aggregates the time series data to a specified frequency.

    Parameters:
    - time_series (dict): Dictionary with model names as keys and data frames as values.
    - aggregation (str): Aggregation frequency. Defaults to 'D' for daily.

    Returns:
    - aggregated_time_series (dict): Dictionary with aggregated time series data.
    """
    aggregated_time_series = {}
    for model_name, df in time_series.items():
        aggregated_time_series[model_name] = df.resample(aggregation).sum()
    return aggregated_time_series

def smooth_time_series(time_series, window=7):
    """
    Applies a moving average smoothing to the time series data.

    Parameters:
    - time_series (dict): Dictionary with model names as keys and data frames as values.
    - window (int): The window size for the moving average. Defaults to 7.

    Returns:
    - smoothed_time_series (dict): Dictionary with smoothed time series data.
    """
    smoothed_time_series = {}
    for model_name, df in time_series.items():
        smoothed_time_series[model_name] = df.rolling(window=window, min_periods=1).mean()
    return smoothed_time_series

def plot_time_series(time_series, title, save_path, n_values=96, max_variates=8):
    """
    Plots the actual vs predicted values for multivariate time series data.

    Parameters:
    - time_series (dict): Dictionary with model names as keys and data frames as values.
                          Each data frame must have variate names as column names.
    - title (str): Title of the plot.
    - save_path (str): Path to save the plot.
    - n_values (int): Number of values to plot on the x-axis. Defaults to 96.
    - max_variates (int): Number of variate series to plot. Defaults to 8.

    Returns:
    - None
    """
    # Use only the first n_values for plotting
    y_true = time_series['Actual'].iloc[:n_values]
    y_preds = {model_name: df.iloc[:n_values] for model_name, df in time_series.items() if model_name != 'Actual'}

    # Load item mapping
    item_mapping = load_item_mapping()
    
    num_variates = min(y_true.shape[1], max_variates)
    
    num_columns = 2
    num_rows = (num_variates + 1) // num_columns
    
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(14, 6 * num_variates // 3), sharex=True)
    axes = axes.flatten()
    
    if num_variates == 1:
        axes = [axes]  # Make axes iterable if there's only one subplot
    
    variate_names = y_true.columns
    
    for i in range(num_variates):
        axes[i].plot(y_true.iloc[:, i], label='Actual', linestyle='-', marker='o', color='black')
        for model_name, y_pred in y_preds.items():
            axes[i].plot(y_pred.iloc[:, i], label=f'Predicted ({model_name})', linestyle='--', marker='x')
        
        variate_title = variate_names[i]
        item_number = variate_title #.split('__')[-1] if '__' in variate_title else None
        item_name = item_mapping.get(item_number, "Unknown")
        axes[i].set_title(f'{variate_title} ({item_name})', fontsize=14)
        axes[i].legend(fontsize=12)
        axes[i].grid(True, linestyle='--', alpha=0.6)
        axes[i].tick_params(axis='both', which='major', labelsize=12)
        #plt.setp(axes[i].xaxis.get_majorticklabels(), rotation=45)
        
    for j in range(num_variates, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    fig.suptitle(title, fontsize=16)

    # Ensure save_path directory exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Save the plot
    plot_file_path = os.path.join(save_path, f'{title.replace(" ", "_").lower()}.png')
    plt.savefig(plot_file_path, dpi=300)

def plot_multivariate_time_series_predictions(time_series, max_variates=8, title='Time Series Predictions', save_path="experiment/plots", n_values=96, smoothing_window=14):
    """
    Plots the actual vs predicted values for multivariate time series data and its aggregated and smoothed versions.

    Parameters:
    - time_series (dict): Dictionary with model names as keys and data frames as values. 
                          Each data frame must have variate names as column names.
    - max_variates (int): Number of variate series to plot. Defaults to 8.
    - title (str): Title of the plot.
    - save_path (str): Path to save the plot. Defaults to "experiment/plots".
    - n_values (int): Number of values to plot on the x-axis. Defaults to 96.
    - smoothing_window (int): Window size for the moving average smoothing. Defaults to 7.

    Returns:
    - None
    """
    print(time_series)
    # Align all DataFrames to have the same indices
    aligned_dataframes = align_dataframes(list(time_series.values()))
    time_series = dict(zip(time_series.keys(), aligned_dataframes))
    print(time_series)

    # Plot the original time series
    plot_time_series(time_series, title=title, save_path=save_path, n_values=n_values, max_variates=max_variates)

    # Aggregate the time series to daily frequency
    aggregated_time_series = aggregate_time_series(time_series, aggregation='D')

    # Plot the aggregated time series
    plot_time_series(aggregated_time_series, title='Aggregated ' + title, save_path=save_path, n_values=n_values, max_variates=max_variates)

    # Smooth the aggregated time series
    smoothed_aggregated_time_series = smooth_time_series(aggregated_time_series, window=smoothing_window)

    # Plot the smoothed aggregated time series
    plot_time_series(smoothed_aggregated_time_series, title='Smoothed Aggregated ' + title, save_path=save_path, n_values=n_values, max_variates=max_variates)

