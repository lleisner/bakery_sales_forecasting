import requests
from datetime import datetime
import pytz
import pandas as pd
from data_fetcher.sub_fetchers.base_fetcher import BaseFetcher


class WeatherFetcher(BaseFetcher):
    def __init__(self):
        lat = 53.77
        lon = 7.69
        key = "d8485d5f5221ad77dae7328a7c8781bd"
        self.url_1h2d = url = f"https://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={lon}&units=metric&appid={key}"
        self.url_3h5d = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&units=metric&appid={key}"
        
    def cloud_cover_to_okta(self, cover):
        return min(round(cover / 12.5), 8)

    def process_1h2d_data(self):
        data = requests.get(self.url_1h2d).json()
        processed_data = [{
            'datetime': datetime.utcfromtimestamp(hour['dt']),
            'temperature': hour['temp'],
            'precipitation': hour['rain']['1h'] if 'rain' in hour and '1h' in hour['rain'] else 0,
            'cloud_cover': self.cloud_cover_to_okta(hour['clouds']),
            'wind_speed': hour['wind_speed'],
            'wind_direction': hour['wind_deg']
        } for hour in data['hourly']]
        return self.post_process_data(processed_data)
        

    def process_3h5d_data(self):
        data = requests.get(self.url_3h5d).json()
        processed_data = [{
            'datetime': datetime.utcfromtimestamp(item['dt']),
            'temperature': item['main']['temp'],
            'precipitation': item['rain']['3h']/3 if 'rain' in item and '3h' in item['rain'] else 0,
            'cloud_cover': self.cloud_cover_to_okta(item['clouds']['all']),
            'wind_speed': item['wind']['speed'],
            'wind_direction': item['wind']['deg']
        } for item in data['list']]
        return self.post_process_data(processed_data)
    
    def post_process_data(self, processed_data):
        df = pd.DataFrame(processed_data)
        berlin_timezone = pytz.timezone('Europe/Berlin')
        df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize(pytz.utc).dt.tz_convert(berlin_timezone)
        df.set_index('datetime', inplace=True)
        df.index = df.index.tz_localize(None) 
        return df
    
    def get_data(self):
        return self.process_1h2d_data().combine_first(self.process_3h5d_data().resample('H').bfill()).round(2)
        
        

if __name__ == "__main__":
    fetcher = WeatherFetcher()
    df = fetcher.get_data()
    print(df)
