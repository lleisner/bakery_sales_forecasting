import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

def visualize_seasonality(df, item, start_date=None, end_date=None, period=24):
    # Select data for the specified week
    if (start_date is not None) & (end_date is not None):
        print('fuck')
        df = df[(df.index >= start_date) & (df.index <= end_date)]

    """
    Visualize the seasonality of a specific item using STL decomposition.

    Parameters:
    - df: DataFrame with datetime index and columns representing items.
    - item: Name of the item to visualize.

    Returns:
    None (displays the plot).
    """
    # Apply STL decomposition to the selected item's time series
    stl_result = STL(df[item], period=period)
    res = stl_result.fit()

    # Plot the original time series
    plt.figure(figsize=(12, 6))
    
    plt.subplot(6, 1, 1)
    plt.plot(df[item], label='Original')
    plt.legend()

    # Plot the trend component
    plt.subplot(6, 1, 2)
    plt.plot(res.trend, label='Trend')
    plt.legend()
    
    # Plot the trend component
    plt.subplot(6, 2, 2)
    plt.plot(df.index, res.trend, label='Trend', linewidth=2, color='orange')
    plt.title('Trend Component')
    plt.legend()

    # Plot the seasonal component
    plt.subplot(6, 1, 3)
    plt.plot(res.seasonal, label='Seasonal')
    plt.legend()

    # Plot the seasonal component
    plt.subplot(6, 2, 3)
    plt.plot(df.index, res.seasonal, label='Seasonal', linewidth=2, color='green')
    plt.title('Seasonal Component')
    plt.legend()


    # Plot the residual component
    plt.subplot(6, 2, 1)
    plt.plot(res.resid, label='Residual')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Example usage:
# visualize_seasonality(df, 'item_10')
