import pandas as pd
import os

from utils.plot_dataframe import plot_multiple_dataframes
from data_provider.sub_providers import (
    SalesDataProvider,
    FahrtenDataProvider,
    GaestezahlenProvider,
    WeatherDataProvider,
    FerienDataProvider
)

class DataProvider:
    def __init__(self, 
                 period='8h',
                 start_date='2019-02-20',
                 end_date='2023-10-31',
                 source_directory='data/sub_datasets',
                 item_selection=["pastry"],
                 top_k=16,):
        
        self.providers = {
            'sales': SalesDataProvider(item_selection=item_selection, top_k=top_k),
            'ferien': FerienDataProvider(),
            'gaeste': GaestezahlenProvider(),
            'fahrten': FahrtenDataProvider(),
            'weather': WeatherDataProvider(),
        }
        self.source_directory = source_directory
        self.start_date = start_date
        self.end_date = end_date
        self.period = period
        self.top_k = top_k
        

    def create_new_sub_databases(self, provider_list=None):
        provider_list = list(self.providers.keys()) if provider_list is None else provider_list
        for key, provider in self.providers.items():
            if key in provider_list:
                try:
                    # Save data to their respective files and append to main data
                    provider.save_to_csv(data_directory=self.source_directory, filename=key)
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
    
    def save_combined_data(self, directory='data/sales_forecasting'):
        os.makedirs(directory, exist_ok=True)
        combined_data = self.load_and_concat_sub_databases()
        filename = f"sales_forecasting_{self.period}_top{self.top_k}.csv"
        file_path = os.path.join(directory, filename)
        combined_data.to_csv(file_path)
        print(f"Combined data saved to {file_path}")
        
    
    def filter_time_and_date(self, df):
        time_mapping = {
            '8h': ("08:00:00", "15:00:00"),
            '16h': ("06:00:00", "21:00:00"),
            '24h': ("00:00:00", "23:00:00"),
        }
        
        if self.period not in time_mapping:
            if self.period == '1d':
                df = df.resample('D').sum()
            elif self.period == '1w':
                df = df.resample('W-MON').sum()
            else:
                raise ValueError(f"Invalid period {self.period}. Please choose among 8h, 16h, 24h, 1d, 1w.")
            start_time, end_time = time_mapping['24h']
        else:
            start_time, end_time = time_mapping[self.period]
            df = df.between_time(start_time, end_time)

        df = df[(df.index >= pd.Timestamp(f"{self.start_date} {start_time}")) & 
                (df.index <= pd.Timestamp(f"{self.end_date} {end_time}"))]
        
        return df


            
if __name__ == "__main__":
    if True:
        for k in [2, 16, 32, 64]:
            provider = DataProvider(top_k=k)
            provider.create_new_sub_databases(provider_list="sales")
            for period in ['8h', '16h', '24h', '1d', '1w']:
                provider = DataProvider(period=period, top_k=k)
                provider.save_combined_data(f"data/sales_forecasting/sales_forecasting_{period}")
                
            
    plot_multiple_dataframes("data/sales_forecasting")
