import numpy as np
import pandas as pd

def calculate_windows(entries_per_day, lookback_days, forecast_days):
    """
    Calculates the window size and forecast size as number of timestamps based on the number of entries per day, 
    and the lookback and forecast period specified in days. 

    Parameters:
    - entries_per_day: Number of entries per day in the dataset. (Can be < 1, e.g. for a weekly freq)
    - lookback_days: Number of days to include in the look back window.
    - forecast_days: Number of days to include in the forecast horizon.

    Returns:
    - window_size: Calculated window size.
    - forecast_size: Calculated forecast size.
    """
    window_size = int(entries_per_day * lookback_days)
    forecast_size = max(1, int(entries_per_day * forecast_days))
    return window_size, forecast_size



    


def calculate_entries_per_day(time_col):
    """
    Calculates the number of entries per day based on the time differences between timestamps in the given series.

    Parameters:
    - time_col: A pandas Series containing datetime values.

    Returns:
    - entries_per_day: The calculated number of entries (timestamps) per day.

    Raises:
    - ValueError: If more than two distinct time differences are found. 
    
    Note:
    More than two distinct time differences between timestamps raise an error, since this indicates non-uniformity in the timestamps.
    A single time difference indicates a continous series (e.g. full hourly frequency), 
    two time differences indicate a discontinous series with a constant suspended interval (e.g. business hours frequency).
    
    """    


    # Calculate time differences
    time_diffs = pd.to_datetime(time_col).diff().dropna()    
    time_diff_counts = time_diffs.value_counts()
    
    # Raise error if series displays excessive discontinuity
    if len(time_diff_counts) > 2:
        print(f"Distinct time differences found: {time_diff_counts.index.tolist()}")
        raise ValueError("More than two time differences found")

    # Determine most common differences
    most_common_diff = time_diff_counts.index[0]
    second_most_common_diff = time_diff_counts.index[1] if len(time_diff_counts) > 1 else None

    # Calculate entries per day
    if second_most_common_diff:
        gap_hours = second_most_common_diff.components.hours - 1
        entries_per_day = 24 - gap_hours        
    else:
        entries_per_day = 1 / (most_common_diff.components.days + most_common_diff.components.hours / 24)
    
    return entries_per_day

def infer_ts_cols_by_int_name(data):
    """
    Infers time series columns by numeric names and separates covariate columns.
    
    Parameters:
    - data: Input DataFrame.
    
    Returns:
    - ts_cols: List of inferred time series columns (numeric column names).
    - cov_cols: List of covariate columns (other columns).
    
    Note: 
    Specifically designed for the sales forecasting datasets 
    to be used as sep_func in get_var_covar_specs().
    """
    ts_cols = [col for col in data.columns if str(col).isnumeric()]
    cov_cols = [x for x in data.columns if x not in set(ts_cols)]
    return ts_cols, cov_cols

def get_var_covar_specs(data, sep_func=lambda data: (data.columns, [])):
    """
    Separates time series (ts_cols) and covariate columns (cov_cols) in the dataset.
    
    Parameters:
    - data: Input DataFrame.
    - sep_func: Function to separate columns. Default considers all columns as time series.
    
    Returns:
    - ts_cols: List of time series columns.
    - cov_cols: List of covariate columns.
    """
    ts_cols, cov_cols = sep_func(data)
    return ts_cols, cov_cols, None, None

def calculate_data_ranges(train_size, val_size, test_size):
    """
    Calculates the data ranges for training, validation, and testing sets.

    Parameters:
    - train_size (int): The size of the training set.
    - val_size (int): The size of the validation set.
    - test_size (int): The size of the test set.

    Returns:
    - tuple: Ranges for the training, validation, and test sets.
    """
    return ((0, train_size), 
            (train_size, train_size + val_size), 
            (train_size + val_size, train_size + val_size + test_size))
    
    
def calculate_splits(total_entries, window_size, train_split, val_split, test_split):
    """
    Calculates the train, validation, and test split sizes based on total entries, window size,
    forecast size, and the specified split ratios.

    Parameters:
    - total_entries: Total number of entries in the dataset.
    - window_size: Size of each window.
    - forecast_size: Size of the forecast period.
    - train_split: Proportion of data for training.
    - val_split: Proportion of data for validation.
    - test_split: Proportion of data for testing.

    Returns:
    - train_size: Calculated training set size.
    - val_size: Calculated validation set size.
    - test_size: Calculated test set size.
    """

    
    # Calculate number of windows that can be created from the data
    max_full_windows = total_entries  - window_size 
    
    adjust_split_size = lambda split: (int(max_full_windows * split) // window_size) * window_size
    
    max_full_windows = ((total_entries) // window_size) * window_size - window_size
    
    adjust_size = lambda split: (int(max_full_windows * split) // window_size) * window_size
    
    train_size, val_size, test_size = map(adjust_size, [train_split, val_split, test_split])
    
    print("split sizes used for this dataset", (total_entries, train_size, val_size, test_size))
    return train_size, val_size, test_size


def calculate_splits(num_windows, train_ratio, val_ratio, test_ratio, window_size):
    # Calculate initial split sizes
    train_size = int(train_ratio * num_windows)
    val_size = int(val_ratio * num_windows)
    test_size = int(test_ratio * num_windows)
    
    print((train_size, val_size, test_size))
    
    # Adjust split sizes to be multiples of the window_size
    train_size = (train_size // window_size) * window_size
    val_size = (val_size // window_size) * window_size
    test_size = (test_size // window_size) * window_size
    
    print("WINDOW SIZE USED: ", window_size)
    
    print("split sizes used for this dataset", (num_windows, train_size, val_size, test_size))

    return train_size, val_size, test_size


def check_data_for_size_condition(data, forecast_size, window_size):
    """
    Checks if the dataset fullfills the condition (length_of_data == forecast_size + window_size * k

    Args:
        data (pd.DataFrame): Dataset to be checked
        forecast_size (int): size of the forecast 
        window_size (int): size of the lookback

    Returns:
        pd.DataFrame: Dataset with the required size adjustments
    """
    data_size = len(data)
    
    if (data_size - forecast_size) % window_size == 0:
        return data
    
    nearest_size = ((data_size - forecast_size) // window_size + 1) * window_size + forecast_size
    
    if nearest_size > data_size:
        nearest_size = ((data_size - forecast_size) // window_size) * window_size + forecast_size
    
    print(f"Current size of the dataset does not fit the desired lookback and forecast settings, adjusting size from {data_size} to {nearest_size}")
    return data[-int(nearest_size):]


LARGEST_NUMBER_OF_LOOKBACK_DAYS = 28

def analyze_data(file_path, train_split, val_split, test_split, lookback_days, forecast_days, numeric_ts_cols=True, datetime_col='date'):
    """
    Analyzes a dataset and returns relevant configuration details for time series analysis.

    Parameters:
    - file_path (str): Path to the CSV file containing the dataset.
    - train_split (float): Proportion of the data to be used for training.
    - val_split (float): Proportion of the data to be used for validation.
    - test_split (float): Proportion of the data to be used for testing.
    - lookback_days (int): Number of past days to consider for each data point (lookback window size).
    - forecast_days (int): Number of future days to predict (forecast horizon).
    - numeric_ts_cols (bool): Whether to infer time series columns by their numeric names. Defaults to True.
    - datetime_col (str): The column name containing the datetime information. Defaults to 'date'.

    Returns:
    - dict: A dictionary containing analyzed dataset information, including file path, datetime column, 
            numerical and categorical covariate columns, timeseries columns, train/val/test ranges, 
            history length (lookback window size), prediction length (forecast horizon), and other settings.
    """
    
    
    data = pd.read_csv(file_path, header=0)
    
    entries_per_day = calculate_entries_per_day(data[datetime_col])
    
    ## TASK: Crop the data to dates that should actually be used (multiple of lookback_days +)
    
    total_days = len(data) / entries_per_day
    
    
    window_size, forecast_size = calculate_windows(entries_per_day=entries_per_day, 
                                                   lookback_days=lookback_days, 
                                                   forecast_days=forecast_days)
    
    print("entries per day", entries_per_day, "window_size: ", window_size, "forecast", forecast_size)
    

    
    # Use a split that can accomodate full windows for the largest window size used
    largest_window = LARGEST_NUMBER_OF_LOOKBACK_DAYS * entries_per_day
    
    # Crop the data to a size that fits all windows and the forecast
    data = check_data_for_size_condition(data, forecast_size=forecast_size, window_size=largest_window)
    
    # Account for the dropped timestamps after the lag is applied
    total_entries = len(data) - forecast_size
    num_windows = total_entries - largest_window
    

    
    train_size, val_size, test_size = calculate_splits(num_windows=num_windows, 
                                                       train_ratio=train_split, 
                                                       val_ratio=val_split, 
                                                       test_ratio=test_split, 
                                                       window_size=window_size)
    
    train_range, val_range, test_range = calculate_data_ranges(train_size, val_size, test_size)
    ts_cols, cov_cols, cat_cov_cols, cyc_cov_cols = get_var_covar_specs(data.drop(columns=[datetime_col]), sep_func=infer_ts_cols_by_int_name if numeric_ts_cols else lambda x: (x.columns, []))

    return {
        "data_path": file_path,
        "datetime_col": datetime_col,
        "numerical_cov_cols": cov_cols,
        "categorical_cov_cols": cat_cov_cols,
        "cyclic_cov_cols": cyc_cov_cols,
        "timeseries_cols": ts_cols,
        "train_range": train_range,
        "val_range": val_range,
        "test_range": test_range,
        "hist_len": window_size,
        "pred_len": forecast_size,
        "stride": 1,
        "sample_rate": 1,   
        }  


if __name__ == "__main__":
    # Example usage:
    file_path_8h = 'data/sales_forecasting/sales_forecasting_8h/sales_forecasting_8h_top2.csv'
    file_path_16h = 'data/sales_forecasting/sales_forecasting_16h/sales_forecasting_16h_top2.csv'

    file_path_1w= 'data/sales_forecasting/sales_forecasting_1w/sales_forecasting_1w_top2.csv'
    file_path_1d= 'data/sales_forecasting/sales_forecasting_1d/sales_forecasting_1d_top2.csv'


    # Testing the function on both datasets to ensure it works for both daily and sub-daily frequencies
    results = {}
    for file_path in [file_path_8h, file_path_1w, file_path_1d, file_path_16h]:
        dataset = file_path.split("/")[-1].split('.')[0]
        results[dataset] = analyze_data(file_path, 0.5, 0.1667, 0.1667, 28, 7)
        print(results[dataset])

