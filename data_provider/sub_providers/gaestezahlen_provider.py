import pandas as pd
import numpy as np
from data_provider.sub_providers.base_provider import BaseProvider

class GaestezahlenProvider(BaseProvider):
    def __init__(self, source_directory='data/raw_sources/gaestezahlen'):
        super().__init__(source_directory)
        
    def _read_file(self, filepath):
        def parse_date(date_str):
            return pd.to_datetime(date_str, format='%d.%m.%y')
        df = pd.read_csv(filepath, header=None, names=['dayofweek', 'date', 'gaestezahlen'], index_col=1, parse_dates=True, encoding='latin1', sep=';', engine='python', date_format='%d.%m.%y')
        return df
    
    def _process_data(self, df):
        df = df.resample('H').ffill()
        return df.drop(columns=['dayofweek'])
        
if __name__ == "__main__":
    provider = GaestezahlenProvider()
    df = provider.get_data()
    print(df)