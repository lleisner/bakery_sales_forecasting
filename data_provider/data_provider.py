import pandas as pd
import os

from utils.configs import ProviderConfigs
from utils.analyze_data import analyze_all_datasets
from data_provider.sub_providers import (
    SalesDataProvider,
    FahrtenDataProvider,
    GaestezahlenProvider,
    WeatherDataProvider,
    FerienDataProvider
)

class DataProvider:
    def __init__(self, 
                 length_of_day='8h',
                 start_date='2019-02-01',
                 end_date='2023-08-01',
                 source_directory='data/database',
                 item_selection = ["broetchen", "plunder"]):
        
        self.providers = {
            'sales': SalesDataProvider(item_selection=item_selection),
            'ferien': FerienDataProvider(),
            'gaeste': GaestezahlenProvider(),
            'fahrten': FahrtenDataProvider(),
            'weather': WeatherDataProvider(),
        }
        self.source_directory = source_directory
        self.start_date = start_date
        self.end_date = end_date
        self.length_of_day = length_of_day
        

    def create_new_sub_databases(self, provider_list=None):
        provider_list = list(self.providers.keys()) if provider_list is None else provider_list
        for key, provider in self.providers.items():
            if key in provider_list:
                try:
                    # Save data to their respective files and append to main data
                    provider.save_to_csv(source_directory=self.source_directory, filename=key)
                    print(f"Created new {key} database")
                except Exception as e:
                    print(f"Failed to create new {key} database: {str(e)}")

    
    def load_and_concat_sub_databases(self):
        data = []
        for key in self.providers.keys():
            file_path = os.path.join(self.source_directory, key)
            sub_database = pd.read_csv(f'{file_path}.csv', index_col=0, parse_dates=True)
            data.append(sub_database)

        result = pd.concat(data, axis=1, join='outer').fillna(0)
        result.index.name = 'date'
        filtered_result = self.filter_time_and_date(result)
        
        return filtered_result
    
    def save_combined_data(self, directory='ts_datasets'):
        combined_data = self.load_and_concat_sub_databases()
        filename = f"sales_forecasting_{self.length_of_day}.csv"
        file_path = os.path.join(directory, filename)
        combined_data.to_csv(file_path)
        print(f"Combined data saved to {file_path}")
        
    
    def filter_time_and_date(self, df):
        time_mapping = {
            '8h': ("08:00:00", "15:00:00"),
            '16h': ("06:00:00", "21:00:00"),
            '24h': ("00:00:00", "23:00:00"),
            '1d':("00:00:00", "23:00:00"),
        }
        
        if self.length_of_day not in time_mapping:
            raise ValueError(f"Invalid length of day {self.length_of_day}. Please choose among 8, 16, or 24.")
        elif self.length_of_day == '1d':
            df = df.resample('D').sum()
        start_time, end_time = time_mapping[self.length_of_day]

        df = df[(df.index >= pd.Timestamp(f"{self.start_date} {start_time}")) & 
                (df.index <= pd.Timestamp(f"{self.end_date} {end_time}"))]
        
        df = df.between_time(start_time, end_time)
        
        return df

    """    
    DEPRECEATED
    
    def load_database(self):
        data = []
        for key in self.providers.keys():
            file_path = os.path.join(self.source_directory, key)
            database = pd.read_csv(f'{file_path}.csv', index_col=0, parse_dates=True)
            
            try:
                # try integration of new data (unused so far)
                new_data = pd.read_csv(f'data/new_data/{key}.csv', index_col=0, parse_dates=True)
                result = new_data.combine_first(database)
            except:
                result = database
                
            data.append(result)

        # Join all dataframes
        result = data[0]
        for df in data[1:]:
            result = result.join(df, how='outer')
            
        result = pd.concat(data, axis=1, join='outer')
        result = self.filter_time_and_date(result)
        
        return result.fillna(0)
    """



            
if __name__ == "__main__":
    for period in ['8h', '16h', '24h', '1d']:
        provider = DataProvider(length_of_day=period)
        provider.save_combined_data()
    
    analyze_all_datasets("ts_datasets")
    
 
