import pandas as pd
import yaml
import os

def analyze_dataset(file_path, timestamp_col='date', train_split=0.496, val_split=0.166, test_split=0.166, lookback_days=28, forecast_days=7, number_of_days_included=1680, window_size=None, forecast_size=None, infer_ts_cols=False):
    """
    Analyzes a time series dataset from a CSV file, providing detailed dataset information and parameters for potential data analysis. Outputs the data in YAML format.

    Parameters:
    - file_path (str): Path to the CSV file containing the dataset.
    - timestamp_col (str): Column name in the CSV file that contains timestamp data. Default is 'date'.
    - train_split (float): Proportion suggested for training data analysis. Default is 0.496.
    - val_split (float): Proportion suggested for validation data analysis. Default is 0.166.
    - test_split (float): Proportion suggested for testing data analysis. Default is 0.166.
    - lookback_days (int): Number of past days to consider for each data point, used to calculate window size if not provided.
    - number_of_days_included (int): The total number of days the data ranges over.
    - window_size (int or None): Custom window size for data processing, calculated if not provided.
    - infer_ts_cols (bool): defaults to False. If true, set numeric column names as time series columns. Tailored to the custom sales_forecasting datasets, where columns with numeric column names are the time series columns.
    Returns:
    - str: A YAML-formatted string containing metadata about the dataset, such as file name, number of dimensions, suggested window size, and suggested sizes for data analysis, structured for potential future data analysis.

    Raises:
    - Exception: If there is an error in reading the file, parsing dates, or inferring the time series frequency.
    """
    # Load the dataset
    data = pd.read_csv(file_path, header=0)
    # Adjust this parameter based on the size of the daily dataset
    file_name = file_path.split("/")[-1]
    dataset_name = file_name.split('.')[0] 
    # Try to parse the timestamp column and determine frequency
    try:
        data[timestamp_col] = pd.to_datetime(data[timestamp_col])
        data.set_index(timestamp_col, inplace=True)
        freq = pd.infer_freq(data.index)
        print(f"Time series frequency inferred as: {freq}")
    except Exception as e:
        print(f"Could not infer frequency from column '{timestamp_col}': {e}")
    
    
    # Display basic info about the dataset
    print(f"Total number of entries: {len(data)}")   
    # Calculate total entries that are multiples of the window size
    total_entries = len(data)
    time_points_per_day = total_entries / number_of_days_included
    
    if not window_size:
        window_size = int(time_points_per_day * lookback_days)
    if not forecast_size:
        forecast_size = max(1, int(time_points_per_day * forecast_days))
    
    max_full_windows = total_entries // window_size * window_size
    
    # Calculate raw split sizes
    raw_train_size = int(max_full_windows * train_split)
    raw_val_size = int(max_full_windows * val_split)
    raw_test_size = int(max_full_windows * test_split)

    # Adjust sizes to be multiples of window_size
    train_size = raw_train_size // window_size * window_size 
    val_size = raw_val_size // window_size * window_size
    test_size = raw_test_size // window_size * window_size

    # Ensure total used entries does not exceed max_full_windows
    total_used = train_size + val_size + test_size
    if total_used > max_full_windows:
        # Reduce test_size to fit the window size multiple constraint
        test_size -= (total_used - max_full_windows)
        
    train_size, val_size, test_size = train_size + 1, val_size + 1, test_size + 1
    if infer_ts_cols:
        ts_cols = [col for col in data.columns if str(col).isnumeric()]
        cov_cols = [x for x in  data.columns if x not in set(ts_cols)]
    else:
        ts_cols = data.columns
        cov_cols = []

    # Prepare output data
    output = {
        dataset_name: {
            "file_name": file_name,
            "ts_dim": len(ts_cols),
            "cov_dim": len(cov_cols), 
            "suggested_window": window_size,
            "suggested_forecast": forecast_size,
            "train_size": train_size,
            "val_size": val_size,
            "test_size": test_size,
            "ts_cols": ts_cols,
            "cov_cols": cov_cols,
        }
    }
    return output

    


def analyze_all_datasets(folder_path, timestamp_col='date', train_split=0.496, val_split=0.166, test_split=0.166, infer_ts_cols=False, yaml_output_path='experiment/dataset_analysis.yaml'):
    """
    Analyzes all CSV files in a specified directory for time series data, leveraging the `analyze_dataset` function to extract and print dataset parameters and metadata for each file in YAML format.

    Parameters:
    - folder_path (str): Path to the directory containing CSV files.
    - timestamp_col (str): Default column name for dates in the CSV files, passed to `analyze_dataset`.
    - train_split (float): Default proportion suggested for the training data analysis, passed to `analyze_dataset`.
    - val_split (float): Default proportion suggested for the validation data analysis, passed to `analyze_dataset`.
    - test_split (float): Default proportion suggested for the testing data analysis, passed to `analyze_dataset`.
    - infer_ts_cols (bool): Defaults to False. If true, set numeric column names as time series columns. Tailored to the custom sales_forecasting datasets, where columns with numeric column names are the time series columns.
    - yaml_output_path (str): Path to save the output YAML file. Default is 'experiment/dataset_analysis.yaml'.

    Returns:
    - None: Outputs dataset analysis results directly to the console. No return value. Errors during file processing are caught and printed.

    Raises:
    - Exception: Outputs an error message if no CSV files are found in the directory or if an error occurs during the analysis of any file.
    """
    # List all files in the folder
    all_files = os.listdir(folder_path)
    # Filter to include only CSV files
    csv_files = [file for file in all_files if file.endswith('.csv')]
    
    # Check if no CSV files found
    if not csv_files:
        print("No CSV files found in the directory.")
        return

    # List to store all YAML outputs
    yaml_outputs = {}

    # Loop through all found CSV files and analyze them
    for file_name in csv_files:
        file_path = os.path.join(folder_path, file_name)
        print(f"\nAnalyzing file: {file_path}")
        try:
            result = analyze_dataset(
                file_path, 
                timestamp_col=timestamp_col, 
                train_split=train_split, 
                val_split=val_split, 
                test_split=test_split,
                infer_ts_cols=infer_ts_cols,
            )
            print(yaml.dump(result, sort_keys=False, default_flow_style=False, allow_unicode=True))
            yaml_outputs.update(result)
        except Exception as e:
            print(f"Failed to analyze {file_path}: {e}")

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(yaml_output_path), exist_ok=True)

    # Write all YAML outputs to a file
    with open(yaml_output_path, 'w') as yaml_file:
        yaml.dump(yaml_outputs, yaml_file, sort_keys=False, default_flow_style=False, allow_unicode=True)

    print(f"All entries have been written to '{yaml_output_path}'.")

if __name__ == "__main__":
    analyze_all_datasets("data/sales_forecasting", infer_ts_cols=True)

