import pandas as pd
import os
from data_fetcher.sub_fetchers.base_fetcher import BaseFetcher

class SalesFetcher(BaseFetcher):
    def __init__(self, file_path='/Users/lorenzleisner/Desktop/CLOUD/GFBD/lleisner/vcom.VComExp', time_mapping_path='data/raw/time_mapping.txt'):
        self.file_path = file_path
        self.file_path = '/mnt/c/cloud/gfbd/lleisner/vcom.VComExp'
        self.time_mapping = self.get_time_mapping(time_mapping_path)
        
        
    def get_data(self):
        data = {}  # Initialize a dictionary to hold the data

        with open(self.file_path, 'r') as file:
            current_date = None  # Initialize the current date
            for line in file:
                line_info = line.strip()
                if line_info.startswith('0'):
                    parts = line_info.split(';')
                    current_date = parts[3].split(",")[1]
                elif line_info.startswith('8'):
                    parts = line_info.split(',')
                    time_info = self.time_mapping.get(parts[2])  # Extracting the time information
                    datetime_index = pd.to_datetime(f'{current_date} {time_info}', format='%d.%m.%y %H:%M')
                    item_info = parts[3].split("/")[0]  # Extracting the item information
                    if ":" in item_info:
                        quantity = float(item_info.split(":")[1])  # Extracting number of items sold
                        item_number = parts[1]
                        if item_number not in data:
                            data[item_number] = {}  # Create a new inner dictionary for the current item number if it doesn't exist
                        if datetime_index not in data[item_number]:
                            data[item_number][datetime_index] = quantity  # Add a new time info entry if it doesn't exist
                        else:
                            data[item_number][datetime_index] += quantity  # Accumulate the quantity if the time info already exists
        # Creating a DataFrame from the dictionary
        df = pd.DataFrame(data).fillna(0) 
        df.sort_index(inplace=True)
        df = df.reindex(sorted(df.columns, key=lambda x: int(x)), axis=1)
        return df
      
    def get_time_mapping(self, time_mapping_path) -> pd.DataFrame:
        time_mapping = {}
        with open(time_mapping_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) >= 3:
                    time_mapping[parts[0]] = parts[2][:2] + ':00'
        return time_mapping

    def remove_vcom_file(self):
        try:
            os.remove(self.file_path)
        except Exception as e:
            print(f"Could not remove file: {e}")

if __name__ == "__main__":
    fetcher = SalesFetcher()
    df = fetcher.get_data()
    print(df)
    
