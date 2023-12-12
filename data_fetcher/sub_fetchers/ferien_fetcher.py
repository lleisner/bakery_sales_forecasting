import requests
from datetime import datetime
import pandas as pd
from urllib.parse import urlencode, urljoin
from data_fetcher.sub_fetchers.base_fetcher import BaseFetcher

class FerienFetcher(BaseFetcher):
    def __init__(self, api_key='test2021'):
        self.base_url = f'https://api.schulferien.org/deutschland'
        self.api_token = f'?api_token={api_key}'
        
    def get_json(self, url_extension=''):
        return requests.get(f'{self.base_url}{url_extension}{self.api_token}').json()
        
    def get_laender(self):
        return [entry['slug'] for entry in self.get_json()['states']]
        
    def get_vacation_times(self, laender, year='2015'):
        return [self.get_json(f'/{land}/school-holidays/{year}') for land in laender]    
    
    def create_combined_vacation_dataframe(self, list_of_vacation_data, laender, year='2015'):
        year_range = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='H')
        combined_dfs = []
        
        for index, location_data in enumerate(list_of_vacation_data):
            location_name = laender[index]
            location_df = pd.DataFrame(index=year_range, columns=[location_name], dtype=bool)
            location_df[location_name] = False
            
            for period in location_data:
                start = pd.to_datetime(period['periods'][0]['start'])
                end = pd.to_datetime(period['periods'][0]['end'])
                location_df.loc[start:end, location_name] = True
            
            combined_dfs.append(location_df)
        
        combined_df = pd.concat(combined_dfs, axis=1)
        combined_df.index = pd.to_datetime(combined_df.index)
        return combined_df.astype(int)
    
    def get_data(self):
        laender = self.get_laender()
        vacation_times = self.get_vacation_times(laender)
        data = self.create_combined_vacation_dataframe(vacation_times, laender)
        return data





        
if __name__ == "__main__":
    fetcher = FerienFetcher()
    df = fetcher.get_data()

    print(df)