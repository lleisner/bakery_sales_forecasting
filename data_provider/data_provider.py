import pandas as pd
import os

from utils.configs import ProviderConfigs
from data_provider.sub_providers import (
    SalesDataProvider,
    FahrtenDataProvider,
    GaestezahlenProvider,
    WeatherDataProvider,
    FerienDataProvider
)

class DataProvider:
    def __init__(self, configs, data_directory='data/database'):
        self.providers = {
            'gaeste': GaestezahlenProvider(),
            'fahrten': FahrtenDataProvider(),
            'weather': WeatherDataProvider(),
            'ferien': FerienDataProvider(),
            'sales': SalesDataProvider(item_selection=configs.item_selection),
        }
        self.data_directory = data_directory
        self.configs = configs
        

    def create_new_database(self, provider_list=None):
        provider_list = list(self.providers.keys()) if provider_list is None else provider_list
        for key, provider in self.providers.items():
            if key in provider_list:
                try:
                    # Save data to their respective files and append to main data
                    provider.save_to_csv(data_directory=self.data_directory, filename=key)
                    print(f"Created new {key} database")
                except Exception as e:
                    print(f"Failed to create new {key} database: {str(e)}")

    def load_database(self):
        data = []
        for key in self.providers.keys():
            file_path = os.path.join(self.data_directory, key)
            database = pd.read_csv(f'{file_path}.csv', index_col=0, parse_dates=True)
            try:
                new_data = pd.read_csv(f'data/new_data/{key}.csv', index_col=0, parse_dates=True)
                result = new_data.combine_first(database)
            except:
                result = database
            data.append(result)

        result = data[0]
        for df in data[1:]:
            result = result.join(df, how='outer')
        result = self.filter_time_and_date(result)
        
        return result.fillna(0)
        
    
    def filter_time_and_date(self, df) -> pd.DataFrame:
        df = df[(df.index >= pd.Timestamp(self.configs.start_date + ' ' + self.configs.start_time)) & (df.index <= pd.Timestamp(self.configs.end_date + ' ' + self.configs.end_time))]
        df = df.between_time(self.configs.start_time, self.configs.end_time)
        return df

    def categorize_columns(self, df):
        numeric_col_names = []
        dtype_categories = {}

        for col in df.columns:
            # Check if column name is numeric
            try:
                float(col)  # Attempt to convert the column name to float
                numeric_col_names.append(col)
            except ValueError:
                # Categorize remaining columns by their dtype
                col_dtype = str(df[col].dtype)
                if col_dtype not in dtype_categories:
                    dtype_categories[col_dtype] = [col]
                else:
                    dtype_categories[col_dtype].append(col)

        return numeric_col_names, dtype_categories



            
if __name__ == "__main__":
    configs = ProviderConfigs()
    provider = DataProvider(configs=configs)
    #provider.create_new_database(provider_list=['gaeste'])
    df = provider.load_database()
    print(df)
    labels, categories_dict = provider.categorize_columns(df)
    print(labels)
    print(categories_dict)
