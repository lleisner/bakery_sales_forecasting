import pandas as pd
import re
import os
from datetime import datetime
from data_fetcher.sub_fetchers.base_fetcher import BaseFetcher

class SalesFetcher(BaseFetcher):
    def __init__(self):
        #self.file_path = '/Users/lorenzleisner/Desktop/CLOUD/GFBD/lleisner/vcom.VComExp'
        self.file_path = '/mnt/c/cloud/gfbd/lleisner/vcom.VComExp'
        self.time_mapping_path = 'data/time_mapping.txt'
        
        
    def extract_data(self):
        """
        Extract relevant data from the given vcom.VComExp file

        Args:
            file_path (str): Path to the input text file containing data.

        Returns:
            date: the date of the retrieved data
            relevant_sales: DataFrame containing extracted relevant data.
        """
        date = None
        relevant_data = []

        with open(self.file_path, 'r') as file:
            for line in file:
                line = line.strip()
                
                date_match = re.search(r'22,(\d{2}\.\d{2}\.\d{2})', line)
                if date_match:
                    date = datetime.strptime(date_match.group(1), '%d.%m.%y').date()
                    

                elif line.startswith("8,"):
                    relevant_info = {}
                    parts = line.split(",")
                    relevant_info['time_idx'] = int(parts[2])
                    relevant_info["item"] = int(parts[1])
                    relevant_info["amount"] = float(parts[-1].split("/")[0].replace("CA:", ""))
                    relevant_data.append(relevant_info)
                    
        relevant_sales = pd.DataFrame(relevant_data)
        return date, relevant_sales
    
    
    def get_time_mapping(self, date) -> pd.DataFrame:
        """
        Load time zone mapping data from the given text file.

        Args:
            date (str): the date corresponding to the data

        Returns:
            pd.DataFrame: DataFrame containing time_idx -> datetime mapping data.
        """
        time_mapping = []
        with open(self.time_mapping_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) >= 3:
                    time_obj = datetime.strptime(parts[2], '%H:%M').time()
                    combined_datetime = datetime.combine(date, time_obj)
                    time_mapping.append({'time_idx': int(parts[0]), 'datetime': combined_datetime})

        result_df = pd.DataFrame(time_mapping)
        return result_df
   
    
    def merge_time_info_to_data(self, data, time_info) -> pd.DataFrame:
        """
        Merge the relevant data with time zone mapping data and perform data aggregation.

        Args:
            relevant_data_df (pd.DataFrame): DataFrame containing relevant data.
            time_zone_mapping_df (pd.DataFrame): DataFrame containing time zone mapping data.

        Returns:
            pd.DataFrame: Aggregated DataFrame with relevant data.
        """
        merged_df =  data.merge(time_info, on='time_idx', how='left').drop(columns='time_idx')
        merged_df['datetime'] = merged_df['datetime'].dt.floor('H')

        return merged_df
    
    
    def aggregate_on_hour(self, data) -> pd.DataFrame:
        """
        Aggregate data by summing up values with the same Artikelnummer and datetime.

        Args:
            data (pd.DataFrame): DataFrame with relevant data and time info.
            date (string): Date of the collected data

        Returns:
            pd.DataFrame: Aggregated DataFrame with summed values.
        """
        grouped_df = data.groupby(['item', 'datetime'])['amount'].sum().reset_index()
        pivot_df = grouped_df.pivot(index='datetime', columns='item', values='amount').fillna(0)
        return pivot_df

    def get_data(self):
        date, data = self.extract_data()
        time_info = self.get_time_mapping(date)
        data_with_time_info = self.merge_time_info_to_data(data=data, time_info=time_info)
        data = self.aggregate_on_hour(data=data_with_time_info)
        
        os.remove(self.file_path)
        return data
        
if __name__ == "__main__":
    fetcher = SalesFetcher()
    df = fetcher.get_data()
    print(df)
    
