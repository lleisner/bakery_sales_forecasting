import matplotlib.pyplot as plt
import os
import pandas as pd
from utils.load_item_mapping import load_item_mapping

def plot_multivariate_time_series_predictions(time_series, max_variates=8, title='Time Series Predictions', save_path="experiment/plots", n_values=96):
    """
    Plots the actual vs predicted values for multivariate time series data.
    
    Parameters:
    - time_series (dict): Dictionary with model names as keys and data frames as values. 
                          Each data frame must have variate names as column names.
    - max_variates (int): Number of variate series to plot. Defaults to 2.
    - title (str): Title of the plot.
    - save_path (str): Path to save the plot. Defaults to "experiment/plots".
    - n_values (int): Number of values to plot on the x-axis. Defaults to 448.

    Returns:
    - None
    """
    # Ensure the time_series dict contains at least two entries (one for actual values and at least one for predictions)
    assert len(time_series) > 1, "The time_series dictionary must contain at least two entries: one for actual values and at least one for predictions."

    # Find the intersection of all DataFrame indices
    common_indices = None
    for model_name, df in time_series.items():
        if common_indices is None:
            common_indices = df.index
        else:
            common_indices = common_indices.intersection(df.index)

    # Use only the common indices for plotting
    y_true = time_series['Actual'].loc[common_indices].iloc[:n_values]
    y_preds = {model_name: df.loc[common_indices].iloc[:n_values] for model_name, df in time_series.items() if model_name != 'Actual'}

    # Load item mapping
    item_mapping = load_item_mapping()
    
    timesteps = y_true.shape[0]
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
    plot_file_path = os.path.join(save_path, 'time_series_predictions.png')
    plt.savefig(plot_file_path, dpi=300)
