import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np



def plot_time_series(*arrays):
    """
    Plots actuals and predictions for time series data.

    :param arrays: A variable number of numpy arrays, all with the same shape (sequence_length, number of variates).
                   The first array is treated as actual values, and subsequent arrays as predictions.
    """
    actuals = arrays[0]
    preds = arrays[1:]
    num_variates = actuals.shape[1]
    sequence_length = actuals.shape[0]

    # Determine the plot grid (try to make it as square as possible)
    num_plots_side = int(np.ceil(np.sqrt(num_variates)))
    fig, axes = plt.subplots(num_plots_side, num_plots_side, figsize=(15, 15))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Plot each variate in a separate subplot
    for i in range(num_variates):
        ax = axes[i]
        ax.plot(range(sequence_length), actuals[:, i], label='Actual', color='blue', marker='o')

        # Plot each set of predictions
        for idx, pred in enumerate(preds):
            ax.plot(range(sequence_length), pred[:, i], label=f'Pred_{idx}', linestyle='--', marker='x')

        ax.set_title(f'Variate {i+1}')
        ax.legend()
        ax.grid(True)

    # Turn off unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def plot_preds_actuals(predictions, actuals):
    """
    Plots predictions vs actual values for multiple variables over time in a single figure,
    arranging subplots so that the number of rows and columns are as equal as possible.

    Parameters:
    - predictions: A TensorFlow tensor or NumPy array with shape (n_variables, n_timepoints),
                   containing the predicted values.
    - actuals: A TensorFlow tensor or NumPy array with the same shape as predictions,
               containing the actual values.
    """
    # Convert TensorFlow tensors to NumPy arrays if necessary
    predictions = predictions.numpy()
    actuals = actuals.numpy()

    n_variables = predictions.shape[0]

    # Calculate the number of rows and columns for the subplots
    n_cols = int(np.ceil(np.sqrt(n_variables)))
    n_rows = int(np.ceil(n_variables / n_cols))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5), 
                            constrained_layout=True)
    fig.suptitle('Model Predictions vs Actual Values for All Variables')

    # Hide unused subplots if any
    for i in range(n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        if i < n_variables:
            ax = axs[row, col] if n_variables > 1 else axs[i]
            ax.plot(predictions[i], label='Predictions', marker='o')
            ax.plot(actuals[i], label='Actuals', marker='x')
            ax.set_title(f'Variable {i+1}')
            ax.set_xlabel('Timepoint')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True)
        else:
            # Hide unused subplot
            axs[row, col].axis('off')

    plt.show()

def plot_df_per_column(df):

    df_normalized = df.copy()
    df_normalized.index = df.index.map(lambda x: x.replace(year=2020))
    n_days, n_items = df.shape

    plt_cols = int(np.ceil(np.sqrt(n_items)))
    plt_rows = int(np.ceil(n_items / plt_cols))

    fig, axes = plt.subplots(
            plt_rows, 
            plt_cols, 
            figsize=(plt_cols * 5, plt_rows * 5), 
            sharex=True,
            constrained_layout=True
        )
    for i, item in enumerate(df.columns):
        ax = axes.flatten()[i]
        for year, group in df.groupby(df.index.year)[item]:
            normalized_dates = group.index.map(lambda x: x.replace(year=2020))
            group.index = normalized_dates
            group.plot(ax=ax, label=str(year))


            #group.plot(ax=ax, label=str(year))
        ax.set_title(item)
        ax.legend(title='Year')
    
    plt.tight_layout()
    plt.show()
