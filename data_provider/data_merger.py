import pandas as pd

from data_provider.sub_providers import (
    SalesDataProvider,
    FahrtenDataProvider,
    WeatherDataProvider,
    FerienDataProvider
)

class DataMerger:
    def __init__(self):
        self.providers = [
            SalesDataProvider(),
            FahrtenDataProvider(),
            WeatherDataProvider(),
            FerienDataProvider()
            ]
        
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
        
if __name__ == "__main__":
    merger = DataMerger()
    data = merger.merge()
    print(data)