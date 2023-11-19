import pandas as pd
import os
from utils.multi_processor import DataProvider


class WeatherDataProvider(DataProvider):
    def __init__(self, source_directory='/Users/lorenzleisner/Desktop/CLOUD/GFBD/lleisner/weather'):
        super().__init__(source_directory)
        
    def read_file(self, file_path):
    # Read the data from the Excel file using a semicolon (;) separator
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
        
    def process_data(self, df):
        return df


if __name__=="__main__":
    processor = WeatherDataProvider()
    df = processor.get_data()
    print(df)


