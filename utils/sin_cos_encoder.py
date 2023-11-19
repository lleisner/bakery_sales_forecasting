import numpy as np
import pandas as pd

def create_sine_cosine_encoding(data: pd.DataFrame, name: str, parameter_range: int) -> pd.DataFrame:
    """
    Create sine and cosine encodings for a column of cyclic data within the range of -0.5 to 0.5.
    Args:
        data (pd.Series): The DataFrame column containing the cyclic data.
        parameter_range (int): The range of the cyclic data (e.g. 12 for months).
    Returns:
        pd.DataFrame: A DataFrame containing two columns: 'sine_encoding' and 'cosine_encoding' scaled to the range [-0.5, 0,5].
    """
    # Convert the column to radians
    radians = (data / parameter_range) * 2 * np.pi

    # Calculate sine and cosine encodings
    sine_encoding = 0.5 * (np.sin(radians) + 1) - 0.5
    cosine_encoding = 0.5 * (np.cos(radians) + 1) - 0.5

    # Create a DataFrame with the encodings
    encoding_df = pd.DataFrame({f'{name}_sine_encoding': sine_encoding, f'{name}_cosine_encoding': cosine_encoding})

    return encoding_df