
def calculate_windows(entries_per_day, lookback_days, forecast_days):
    window_size = int(entries_per_day * lookback_days)
    forecast_size = max(1, int(entries_per_day * forecast_days))
    return window_size, forecast_size

def calculate_splits(total_entries, window_size, forecast_size, train_split, val_split, test_split):
    # Calculate the maximum number of full windows that can fit into the dataset
    max_full_windows = ((total_entries - forecast_size) // window_size) * window_size
    
    # Lambda function to calculate and adjust split sizes based on window size
    adjust_size = lambda split: (int(max_full_windows * split) // window_size) * window_size
    
    # Calculate and adjust split sizes
    train_size, val_size, test_size = map(adjust_size, [train_split, val_split, test_split])
    return train_size, val_size, test_size

def print_results(total_days, frequencies, ranges, lookback_days, forecast_days, train_split, val_split, test_split):
    for freq, range_hours in zip(frequencies, ranges):
        entries_per_day = freq
        total_entries = total_days * entries_per_day

        print(f"\nFrequency: {freq}h, Range: {range_hours}h, Total Entries: {total_entries}")
        
        for lookback in lookback_days:
            window_size, forecast_size = calculate_windows(entries_per_day, lookback, forecast_days)
            train_size, val_size, test_size = calculate_splits(total_entries, window_size, forecast_size, train_split, val_split, test_split)

            print(f"Window Size: {window_size}, Forecast Size: {forecast_size}, Dataset Size (Train, Val, Test): ({train_size}, {val_size}, {test_size})")

def get_test_splits(total_entries, window_size, forecast_size, train_split, val_split, test_split):
    for entries in total_entries:
        train_size, val_size, test_size = calculate_splits(entries, window_size, forecast_size, train_split, val_split, test_split)
        print(f"Dataset size (Train, Val, Test): {(train_size, val_size, test_size)}")

# Constants
total_days = 1687
lookback_days = [28, 14, 7]
forecast_days = 7
train_split = 0.50
val_split = 0.1667
test_split = 0.1667

# Define frequencies and ranges
frequencies = [8, 16, 24, 1, 1/7]
ranges = [8, 16, 24, 24, 24]

# Run and print the results
print_results(total_days, frequencies, ranges, lookback_days, forecast_days, train_split, val_split, test_split)

total_entries = [17420, 69680, 7589, 52698]
window_size = 96
forecast_size = 96

get_test_splits(total_entries, window_size, forecast_size, train_split, val_split, test_split)