import pandas as pd
import yaml

def analyze_dataset(file_path, timestamp_col='date', train_split=0.496, val_split=0.166, test_split=0.166, window_size=96):
    # Load the dataset
    data = pd.read_csv(file_path)
    
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

    # Prepare output data
    output = {
        dataset_name: {
            "file_name": file_name,
            "train_size": train_size,
            "val_size": val_size,
            "test_size": test_size,
            "dim": len(data.columns)
        }
    }
    
    # Print output data in YAML format
    print(yaml.dump(output, sort_keys=False, default_flow_style=False, allow_unicode=True))
    
    return output

import os

def analyze_all_datasets(folder_path, timestamp_col='date', train_split=0.496, val_split=0.166, test_split=0.166):
    # List all files in the folder
    all_files = os.listdir(folder_path)
    # Filter to include only CSV files
    csv_files = [file for file in all_files if file.endswith('.csv')]
    
    # Check if no CSV files found
    if not csv_files:
        print("No CSV files found in the directory.")
        return

    # Loop through all found CSV files and analyze them
    for file_name in csv_files:
        file_path = os.path.join(folder_path, file_name)
        print(f"\nAnalyzing file: {file_path}")
        try:
            yaml_output = analyze_dataset(
                file_path, 
                timestamp_col=timestamp_col, 
                train_split=train_split, 
                val_split=val_split, 
                test_split=test_split
            )
        except Exception as e:
            print(f"Failed to analyze {file_path}: {e}")
