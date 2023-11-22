import pandas as pd
import os
from data_provider.sub_providers.base_provider import DataProvider

class WeatherDataProvider(DataProvider):
    def __init__(self, source_directory='data/weather'):
        """
        Initialize WeatherDataProvider instance.

        Args:
        - source_directory (str): Directory path where weather data is located.
        """
        super().__init__(source_directory)
        
    def _read_file(self, file_path):
        """
        Read weather data from a CSV file and perform data preprocessing.

        Args:
        - file_path (str): Path to the CSV file containing weather data.

        Returns:
        - pd.DataFrame: Processed DataFrame containing weather information.
        """
        df = pd.read_csv(file_path, sep=';')

        # Rename columns for clarity
        df.rename(columns={
            "validdate": "Datetime",
            "t_2m:C": "temperature",
            "precip_1h:mm": "precipitation",
            "effective_cloud_cover:octas": "cloud_cover",
            "wind_speed_10m:ms": "wind_speed",
            "wind_dir_10m:d": "wind_direction"
        }, inplace=True)

        # Convert the "date" column to datetime and set the timezone to Berlin
        df['Datetime'] = pd.to_datetime(df['Datetime']).dt.tz_convert('Europe/Berlin')

        # Remove the timezone information to work with Berlin time without timezone
        df['Datetime'] = df['Datetime'].dt.tz_localize(None)

        # Set the "date" column as the index
        df = df.set_index("Datetime")

        # Remove duplicate index entries, keeping the first occurrence
        df = df[~df.index.duplicated(keep='first')]
        
        # Drop Cloud Cover
        df.drop(columns=['cloud_cover'], axis=1, inplace=True)

        return df
        
    def _process_data(self, df):
        """
        Process the given DataFrame. In this case, no more processing is needed.

        Args:
        - df (pd.DataFrame): Input DataFrame to be processed.

        Returns:
        - pd.DataFrame: Processed DataFrame.
        """
        return df


if __name__=="__main__":
    processor = WeatherDataProvider()
    df = processor.get_data()
    print(df)


