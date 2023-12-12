import pandas as pd
import os

from data_provider.time_configs import TimeConfigs
from data_provider.sub_providers import (
    SalesDataProvider,
    FahrtenDataProvider,
    WeatherDataProvider,
    FerienDataProvider
)

class DataProvider:
    def __init__(self, configs, data_directory='data/database', file_name='dataset.csv'):
        self.providers = {
            'sales': SalesDataProvider(),
            'fahrten': FahrtenDataProvider(),
            'weather': WeatherDataProvider(),
            'ferien': FerienDataProvider(),
        }
        self.data_directory = data_directory
        self.file_name = file_name
        self.configs = configs
        
    def merge(self):
        data = []
        for key, provider in self.providers.items():
            data.append(provider.get_data())
        data = pd.concat(data, axis=1, join='outer').asfreq(freq='H').fillna(0)
        data = self.filter_time_and_date(data)
        return data
    
    def save_to_files(self):
        directory = 'data/processed_data'
        for key, provider in self.providers.items():
            df = provider.get_data()
            df = self.filter_time_and_date(df)    
            file_path = os.path.join(directory, f'{key}_data.csv')
            df.to_csv(file_path)
        
    def create_new_database(self):
        data = []
        for key, provider in self.providers.items():
            try:
                data.append(provider.save_to_csv(data_directory=self.data_directory, filename=key))
                print(f"Created new {key} database")
            except Exception as e:
                print(f"Failed to create new {key} database: {str(e)}")

        data = self.filter_time_and_date(pd.concat(data, axis=1, join='outer').asfreq(freq='H').fillna(0))
        data.to_csv(os.path.join(self.data_directory, 'main_database'))
        
    
    def filter_time_and_date(self, df) -> pd.DataFrame:
        df = df[(df.index >= pd.Timestamp(self.configs.start_date + ' ' + self.configs.start_time)) & (df.index <= pd.Timestamp(self.configs.end_date + ' ' + self.configs.end_time))]
        df = df.between_time(self.configs.start_time, self.configs.end_time)
        return df
    
    def get_data(self):
        file_path = os.path.join(self.data_directory, self.file_name)
        print(file_path)
        if not os.path.exists(file_path):
            # Create data if the file doesn't exist yet
            df = self.merge()
            # Create directory if it doesn't exist
            if not os.path.exists(self.data_directory):
                os.makedirs(self.data_directory)
            # Save the dataset to file
            df.to_csv(file_path, index=False)
            print(f"Saved dataset to {file_path}")
            
        # Load existing data
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        #df.set_index(create_datetime_index(), inplace=True)
        print(f"Loaded dataset from {file_path}")
            
        return df
            
if __name__ == "__main__":
    configs = TimeConfigs()
    provider = DataProvider(configs=configs)
    provider.create_new_database()
