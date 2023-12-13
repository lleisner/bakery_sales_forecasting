import pandas as pd
import os
import numpy as np
from datetime import datetime
from data_provider.sub_providers.base_provider import BaseProvider
    
 
class FahrtenDataProvider(BaseProvider):
    def __init__(self, source_directory='data/nsb_fahrzeiten'):
        """
        Initialize FahrtenDataProvider instance.

        Args:
        - source_directory (str): Directory path where Fahrten data is located.
        """
        super().__init__(source_directory)

    def _read_file(self, filepath):
        """
        Process a CSV file and return a DataFrame with selected columns.

        Args:
        - filepath (str): The path to the CSV file.

        Returns:
        pd.DataFrame: Processed DataFrame with selected columns.

        This function reads a CSV file from the specified filepath, selects the first
        six columns (the rest of the data is irrelevant), renames them, and drops rows where all values are missing. 
        """
        df = pd.read_csv(filepath, encoding='latin1', sep=';', engine='python')
        df = df.iloc[:, :6]
        df.columns = ['Datum', 'Start', 'Abfahrt', 'Schiff', 'Ziel', 'Ankunft']
        df.dropna(how='all', inplace=True)
        return df
    
    def _process_data(self, df):
        """
        Process the given DataFrame.

        Args:
        - df (pd.DataFrame): Input DataFrame to be processed.

        Returns:
        - pd.DataFrame: Processed DataFrame.
        """
        df = self._process_schedule_data(df)
        df = self._post_process_schedule_data(df)
        return df
        
    def _process_schedule_data(self, df):
        """
        Process the schedule data in the DataFrame.

        This function processes schedule data by:
        1. converting the 'Datum' column to datetime format (date only)
        2. generating a 'Zeit' (time) column using arrival and departure times
        3. adding _an, _ab info to the 'Schiff' (ship) columns 

        Args:
            df (pd.DataFrame): Input DataFrame containing schedule information.

        Returns:
            pd.DataFrame: Processed DataFrame with modified 'Datum', 'Zeit', and 'Schiff' columns.
        """
        df['Datum'] = pd.to_datetime(df['Datum'], format='%d.%m.%Y')
        
        conditions = [df['Start'].isin(['SP', 'SP+']), df['Ziel'].isin(['SP', 'SP+'])]
        
        df['Zeit'] = np.select(conditions, [df['Abfahrt'], df['Ankunft']], default=np.nan)
        df['Schiff'] = np.select(conditions, [df['Schiff'] + '_ab', df['Schiff'] + '_an'], default=np.nan)
        
        return df
    
    def _post_process_schedule_data(self, df):
        """
        Perform post-processing on the schedule data DataFrame.
        
        1. Drop columns that are no longer needed
        2. Create a Datetimeobject with hourly intervals to be used used as Index
        3. Group by Datetime and expand the resulting Series to a binary Dataframe
    
        Args:
            df (pd.DataFrame): Input DataFrame containing schedule information.

        Returns:
            pd.DataFrame: One-Hot encoded Schiff information with Datetime Index
            
        """
        df.drop(columns=['Start', 'Abfahrt', 'Ziel', 'Ankunft'], inplace=True)
        
        df['datetime'] = (df['Datum'] + pd.to_timedelta(df['Zeit'].astype(str))).dt.floor('H')

        grouped_series = df.groupby('datetime')['Schiff'].agg(list)
        
        one_hot_encoding = pd.get_dummies(grouped_series.apply(pd.Series).stack()).groupby(level=0).max().astype(int)
        
        return one_hot_encoding
    



if __name__ == "__main__":
    processor = FahrtenDataProvider()
    df = processor.get_data()
    print(df)