
import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_single_dataframe_subplot(ax, df, title, column='10', num_points=96):
    """
    Creates a subplot for a given dataframe on the provided axes.

    Parameters:
    - ax (matplotlib.axes.Axes): The axes on which to plot the dataframe.
    - df (pd.DataFrame): The dataframe to plot.
    - title (str): Title for the subplot, typically the name of the data file.
    - column (str): The column of the dataframe to plot. Default is '10'.
    - num_points (int): Number of data points to plot. Default is 192.

    Returns:
    - None: Modifies the provided axes with the plot.
    """
    if column not in df.columns:
        ax.text(0.5, 0.5, 'Column not found', fontsize=12, ha='center')
        return

    # Selecting the data to plot based on the number of points specified
    data_to_plot = df.iloc[-num_points:][column] if num_points < len(df) else df[column]
    ax.plot(data_to_plot.index, data_to_plot, label=column, marker='x', linestyle='-')  # Using line plot with 'x' markers
    ax.set_title(title)  # Setting the title to the file name
    ax.legend()
    ax.grid(True)




def plot_multiple_dataframes(folder_path, timestamp_col='date', column='10', num_points=96):
    """
    Reads all dataframes from a specified directory, creates subplots for each, and displays them in a single figure.

    Parameters:
    - folder_path (str): Path to the directory containing the CSV files.
    - timestamp_col (str): Name of the column to parse as the datetime index.
    - column (str): Column to plot in each dataframe. Default is '10'.
    - num_points (int): Number of data points to plot in each subplot. Default is 192.

    Returns:
    - None: Displays the combined plot.
    """
    
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    num_files = len(files)
    if num_files == 0:
        print("No CSV files found.")
        return

    # Calculate the number of rows and columns for subplots to approximate a 16:9 aspect ratio
    cols = int((16 / 9) * (num_files ** 0.5))
    rows = (num_files + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, 9))
    axes = axes.flatten()  # Flatten to 1D array for easier iteration

    for i, file in enumerate(files):
        if i < len(axes):
            df = pd.read_csv(os.path.join(folder_path, file), index_col=timestamp_col, parse_dates=[timestamp_col])
            plot_single_dataframe_subplot(axes[i], df, file, column, num_points)
        else:
            break

    # Turn off unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


