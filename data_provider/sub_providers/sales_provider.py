import pandas as pd
from datetime import datetime
import os
from data_provider.sub_providers.base_provider import BaseProvider


class SalesDataProvider(BaseProvider):
    def __init__(self, source_directory = 'data/raw_sources/sales', item_selection = ["broetchen", "plunder"]):
        """
        Initialize SalesDataProvider instance.

        Args:
        - source_directory (str): Directory path where sales data is located.
        - item_intervals (list of tuples): List containing intervals of items to consider.
                                           Each tuple represents a range (start, end).
        """
        super().__init__(source_directory)
        
        item_selection_dict = {
            "broetchen": [(10, 13), (16), (20, 32), (34,35), (38, 39)],
            "plunder": [(80), (82, 86), (97, 99), (105, 107), (111, 112)],
            "suppe": [(250, 271)],
            "test": [(10, 11), (83, 84)],
            "normale": [(10)]
        }
        self.item_intervals = [item_selection_dict[category] for category in item_selection]
        
    def _read_file(self, file_path):
        """
        Read data from an Excel file and process it.

        Args:
        - file_path (str): Path to the Excel file.

        Returns:
        - pd.DataFrame: Concatenated and processed DataFrame containing sales data.
        """
        xls = pd.ExcelFile(file_path)
        dataframes = []
        columns = []

        for sheet_name in xls.sheet_names:
            
            if sheet_name == xls.sheet_names[0]: # Parse and save column names from first sheet
                df = pd.read_excel(xls, sheet_name, header=0, index_col=1, engine='openpyxl')
                columns = df.columns.insert(0, '')
            else: # Assign column names to rest of the sheets
                df = pd.read_excel(xls, sheet_name, header=None, names=columns, index_col=1, engine='openpyxl')

            date = datetime.strptime(df.iloc[0,0], '%d.%m.%Y').date()   # Get date for each sheet
            df = df.drop(df.columns[[0, 1, 2]], axis=1).T       # Drop unnecessary columns and transpose Dataframe 
            df.index = [datetime.combine(date, index) for index in df.index]    # Set DatetimeIndex
            df.index.name = 'datetime'
            dataframes.append(df)    

        return pd.concat(dataframes).fillna(0)
    
    def _process_data(self, df):
        """
        Process the given DataFrame.

        Args:
        - df (pd.DataFrame): Input DataFrame to be processed.

        Returns:
        - pd.DataFrame: Processed DataFrame.
        """
        df = df.sort_index()
        df = df.reindex(sorted(df.columns, key=int), axis=1)
        df = df.clip(lower=0)
        df = df[sorted(list(self._filter_range(self.item_intervals) & set(df.columns)))]
        df = self._clean_invalid_sales(df)
        df = self.remove_outlier_days(df)
        df = self._add_opening_times(df)
        return df

    
    def _filter_range(self, item_intervals):
        """
        Helper method to generate a set of selected column indices based on item intervals.

        Returns:
        - set: Set containing selected column indices.
        """
        selected_cols = set()
        
        if isinstance(item_intervals, int):
            selected_cols.add(item_intervals)
        elif isinstance(item_intervals, tuple):
            selected_cols.update(set(int(col) for col in range(item_intervals[0], item_intervals[1] + 1)))
        elif isinstance(item_intervals, list):
            for sub_range in item_intervals:
                selected_cols.update(self._filter_range(sub_range))
        else:
            raise ValueError(f'item_range {item_intervals} not of valid type: {type(item_intervals)}')
        
        return selected_cols

    
    def _clean_invalid_sales(self, df, threshold=0.1):
        """
        Cleans out unwanted noise from outliers:
            - sales outside regular opening hours (8:00-17:00)
            - sales during irregular closing hours (caused by e.g. a midday break)
        
        Args:
        - df (pd.DataFrame): Input DataFrame with outliers
        - threshold (float): min percentage compared to previous/following hour sales to not be considered an outlier.
        
        Returns:
        - df (pd.DataFrame): Cleaned up DataFrame
        """

        valid_time_range = pd.to_datetime(df.index).to_series().between_time('8:00', '16:00')
        invalid_rows_mask = ~df.index.isin(valid_time_range.index)
        df.loc[invalid_rows_mask] = 0
        
        row_sums = df.sum(axis=1)
        prev_row_sums = row_sums.shift(1)
        next_row_sums = row_sums.shift(-1)
        outlier_rows = (row_sums < prev_row_sums * threshold) | (row_sums < next_row_sums * threshold)
        df.loc[outlier_rows] = 0
        return df
    
    
    def _add_opening_times(self, df):
        df['is_open'] = df.any(axis=1).astype(int)
        return df
    

    def aggregate_days(self, df):
        return df.resample('D').sum()
    
    
    def remove_outlier_days(self, df, positive_multiplier=2, negative_multiplier=3):
        daily_totals = df.resample('D').sum()

        # Calculate the sum for each row (across all columns)
        total_sum = daily_totals.sum(axis=1)

        # Calculate rolling mean and standard deviation for the total sum
        rolling_mean = total_sum.rolling(window=8, min_periods=3).mean()
        rolling_std = total_sum.rolling(window=8, min_periods=3).std()

        # Calculate threshold for potential outliers
        threshold_positive = rolling_mean + positive_multiplier * rolling_std
        threshold_negative = rolling_mean - negative_multiplier * rolling_std

        # Identify potential outliers
        potential_outliers = (total_sum > threshold_positive) | (total_sum < threshold_negative)
        
        outlier_bools = potential_outliers.resample('H').ffill()
        
        inverse_bools = ~outlier_bools

        return df.loc[inverse_bools.index.intersection(df.index)][inverse_bools]

if __name__ == "__main__":
    processor = SalesDataProvider()
    df = processor.get_data()
    print(df)
 
