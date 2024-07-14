import matplotlib.pyplot as plt
import os
from utils.load_item_mapping import load_item_mapping

def plot_multivariate_time_series_predictions(y_true, y_preds, model_names, variate_names=None, max_variates=2, title='Time Series Predictions', save_path="experiment/plots", n_values=448):
    """
    Plots the actual vs predicted values for multivariate time series data.
    
    Parameters:
    - y_true (array-like): True values of the time series (shape: [timesteps, variates]).
    - y_preds (list of array-like): List of predicted values from different models (each with shape: [timesteps, variates]).
    - model_names (list of str): List of model names corresponding to each set of predictions.
    - variate_names (list of str, optional): List of variate names to use as subplot titles. Defaults to None.
    - max_variates (int): Number of variate series to plot. Defaults to 2.
    - title (str): Title of the plot.
    - save_path (str): Path to save the plot. Defaults to "experiment/plots".
    - n_values (int): Number of values to plot on the x-axis. Defaults to 448.

    Returns:
    - None
    """
    # Ensure the number of predictions matches the number of model names
    assert len(y_preds) == len(model_names), "The number of y_preds must match the number of model_names."
    
    # Ensure y_preds have the correct shape
    for y_pred in y_preds:
        assert y_pred.shape == y_true.shape, "Each y_pred must have the same shape as y_true."
        
    # Load item mapping
    item_mapping = load_item_mapping()
    
    # Use only the first n_values for plotting
    y_true = y_true[:n_values]
    y_preds = [y_pred[:n_values] for y_pred in y_preds]
    
    timesteps = y_true.shape[0]
    num_variates = min(y_true.shape[1], max_variates)
    
    num_columns = 2
    num_rows = (num_variates + 1) // num_columns
    
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(14, 6 * num_variates//3), sharex=True)
    axes = axes.flatten()
    
    if num_variates == 1:
        axes = [axes]  # Make axes iterable if there's only one subplot
    
    for i in range(num_variates):
        axes[i].plot(y_true[:, i], label='Actual', linestyle='-', marker='o', color='black')
        for y_pred, model_name in zip(y_preds, model_names):
            axes[i].plot(y_pred[:, i], label=f'Predicted ({model_name})', linestyle='--', marker='x')
        
        variate_title = variate_names[i] if variate_names and i < len(variate_names) else f'Variate {i + 1}'
        item_number = variate_title.split('__')[-1] if '__' in variate_title else None
 
        item_name = item_mapping.get(item_number, "Unkown")
        axes[i].set_title(f'{variate_title} ({item_name})', fontsize=14)
        axes[i].legend(fontsize=12)
        axes[i].grid(True, linestyle='--', alpha=0.6)
        axes[i].tick_params(axis='both', which='major', labelsize=12)
        
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

