import pandas as pd
from datetime import datetime
import os
from utils.multi_processor import DataProvider
from utils.custom_dataframe import CustomDataFrame


class SalesDataProvider(DataProvider):
    def __init__(self, source_directory = 'data/sales', item_intervals = [(10, 300)]):
        super().__init__(source_directory)
        self.item_intervals = item_intervals
        
    def read_file(self, file_path):
        xls = pd.ExcelFile(file_path)
        dataframes = []
        columns = []

        for sheet_name in xls.sheet_names:
            
            if sheet_name == xls.sheet_names[0]: # Parse and save column names from first sheet
                df = pd.read_excel(xls, sheet_name, header=0, index_col=1)
                columns = df.columns.insert(0, '')
            else: # Assign column names to rest of the sheets
                df = pd.read_excel(xls, sheet_name, header=None, names=columns, index_col=1)

            date = datetime.strptime(df.iloc[0,0], '%d.%m.%Y').date()   # Get date for each sheet
            df = df.drop(df.columns[[0, 1, 2]], axis=1).T       # Drop unnecessary columns and transpose Dataframe 
            df.index = [datetime.combine(date, index) for index in df.index]    # Set DatetimeIndex
            dataframes.append(df)
               
        return pd.concat(dataframes).fillna(0)
    
    def process_data(self, df):
        df = df.sort_index()
        df = df.reindex(sorted(df.columns, key=int), axis=1)
        df = df.clip(lower=0)
        df = df[sorted(list(self._filter_range() & set(df.columns)))]
        return df

    def _filter_range(self):
        selected_cols = set()
        for start, end in self.item_intervals:
            selected_cols.update(int(col) for col in range(start, end + 1) if start <= col <= end)
        return selected_cols



if __name__ == "__main__":
    processor = SalesDataProvider()
    df = processor.get_data()
    print(df)