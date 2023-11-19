import pandas as pd

def convert_to_hourly(df, hours: tuple=None, fill_method: str=None) -> pd.DataFrame:
    """
    Convert a DataFrame from daily intervals to hourly intervals.

    Args:
        df (pd.DataFrame): The original DataFrame with daily intervals.
        hours (tuple, optional): Tuple specifying the start and end hours for the hourly intervals.
                                 Defaults to (6, 21).
        fill_method (str, optional): Method to fill missing values ('ffill', 'bfill', or None).
                                     Defaults to None.

    Returns:
        pd.DataFrame: A new DataFrame with hourly intervals.
    """
    # Define the expected date format
    date_format = "%d.%m.%Y"

    # Convert the index to datetime with the specified date format
    df.index = pd.to_datetime(df.index, format=date_format, errors='coerce')

    # Create an hourly date range for the entire period
    start_date = df.index.min().strftime('%Y-%m-%d')
    end_date = df.index.max().strftime('%Y-%m-%d')
    hourly_range = pd.date_range(start=start_date + ' 00:00:00', end=end_date + ' 23:00:00', freq='H')

    # Create a DataFrame with hourly intervals
    hourly_df = pd.DataFrame(index=hourly_range)

    # Fill the hourly DataFrame using forward filling
    hourly_df = hourly_df.join(df, how='left')
    if fill_method:
        hourly_df.ffill(inplace=True)
    else:
        hourly_df.fillna(0, inplace=True)

    if hours:
        # Filter the hourly DataFrame to the specified hours    
        hourly_df = hourly_df.between_time(f"{hours[0]}:00", f"{hours[1]}:00")
    return hourly_df
