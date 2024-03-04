import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

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
