import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def visualize_overall_data_distribution(data):
    """
    Visualizes the overall distribution of the dataset.

    Parameters:
    data (array-like): The input data.

    Returns:
    None
    """
    # Flatten the data to treat it as a single distribution
    data_flat = np.asarray(data).flatten()
    
    # Set up the matplotlib figure
    plt.figure(figsize=(10, 6))
    
    # Plot histogram and KDE
    sns.histplot(data_flat, kde=True)
    plt.title('Overall Distribution of the Data')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    
    # Calculate overall statistics
    mean = data_flat.mean()
    median = np.median(data_flat)
    std = data_flat.std()
    
    # Display overall statistics
    textstr = '\n'.join((
        f'Mean: {mean:.2f}',
        f'Median: {median:.2f}',
        f'Standard Deviation: {std:.2f}'))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gca().text(0.95, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
                   verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.show()
