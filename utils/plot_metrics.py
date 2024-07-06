import matplotlib.pyplot as plt
import os

def plot_metrics(history, save_dir='experiment/plots/metrics'):
    """
    Plots the training and validation metrics and saves the plots to a directory.

    Parameters:
    - history: A History object returned by model.fit(), which contains the training and validation metrics.
    - save_dir: Directory to save the plots.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    metrics = [m for m in history.history.keys() if not m.startswith('val_')]
    
    for metric in metrics:
        if f'val_{metric}' in history.history:
            plt.figure(figsize=(10, 6))
            plt.plot(history.history[metric], label=f'Training {metric}')
            plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
            plt.title(f'Training and Validation {metric}')
            plt.xlabel('Epochs')
            plt.ylabel(metric)
            plt.legend()
            plt.grid(True)
            plt.savefig(f'{save_dir}/{metric}.png')
            plt.close()
        else:
            print(f'Skipping metric "{metric}" as it does not have a validation counterpart.')

