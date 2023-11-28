import pandas as pd
import os

from data_provider.sub_providers import (
    SalesDataProvider,
    FahrtenDataProvider,
    WeatherDataProvider,
    FerienDataProvider
)

class DataMerger:
    def __init__(self, data_directory='data/test', file_name='dataset_with_index.csv'):
        self.providers = [
            SalesDataProvider(),
            FahrtenDataProvider(),
            WeatherDataProvider(),
            FerienDataProvider()
            ]
        self.data_directory = data_directory
        self.file_name = file_name
        
        
    def merge(self):
        data = []
        for provider in self.providers:
            data.append(provider.get_data())
        data = pd.concat(data, axis=1, join='outer').asfreq(freq='H').fillna(0)
        data = self.filter_time_and_date(data)
        return data
    
    def filter_time_and_date(self, df, start_date: str='2019-01-01', end_date: str='2023-08-31', start_time: str = '06:00:00', end_time: str = '21:00:00') -> pd.DataFrame:
        df = df[(df.index >= pd.Timestamp(start_date + ' ' + start_time)) & (df.index <= pd.Timestamp(end_date + ' ' + end_time))]
        df = df.between_time(start_time, end_time)
        return df
    
    def get_data(self):
        file_path = os.path.join(self.data_directory, self.file_name)

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
    merger = DataMerger()
    data = merger.merge()
    print(data)