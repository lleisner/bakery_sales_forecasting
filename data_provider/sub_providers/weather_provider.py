import requests
from datetime import datetime
import pytz
import pandas as pd
from data_provider.sub_providers.base_provider import BaseProvider
import os

class WeatherDataProvider(BaseProvider):
    def __init__(self, source_directory="data/new_weather"):
        super().__init__(source_directory)

    def _process_data(self, df):
        df = df[['temp', 'rain_1h', 'clouds_all', 'wind_speed', 'wind_deg']]

        df.index = pd.to_datetime(df.index.str.replace(' UTC', ''), format='%Y-%m-%d %H:%M:%S %z')

        df.index = df.index.tz_convert('Europe/Berlin').tz_localize(None)
        df.index.name = 'datetime'
        df = df.rename(columns={
            'temp': 'temperature',
            'rain_1h': 'precipitation',
            'clouds_all': 'cloud_cover',
            'wind_speed': 'wind_speed',
            'wind_deg': 'wind_direction'
        })
        return df.fillna(0)

    def _read_file(self, file_path):
        return pd.read_csv(file_path, index_col='dt_iso')

        

if __name__ == "__main__":
    provider = NewWeatherDataProvider()
    df = provider.get_data()
    print(df)

