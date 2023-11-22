import pandas as pd
import os
from abc import ABC, abstractmethod
    

class BaseProvider(ABC):
    @abstractmethod
    def get_data(self):
        """
        Abstract method to retrieve and process data.

        Returns:
        - Any: Processed data.
        """
        pass

    
class DataProvider(BaseProvider):
    def __init__(self, source_directory):
        """
        Initialize DataProvider instance.

        Args:
        - source_directory (str): Directory path where data is located.
        """
        self.source_directory = source_directory
        
    @abstractmethod
    def _read_file(self):
        """
        Abstract method to read data from a file.

        Returns:
        - Any: Data read from the file.
        """
        pass
    
    @abstractmethod
    def _process_data(self, df):
        """
        Abstract method to process data in a DataFrame.

        Args:
        - df (pd.DataFrame): Input DataFrame containing data.

        Returns:
        - pd.DataFrame: Processed DataFrame.
        """
        pass
    
    def _read_dir(self):
        """
        Read all files in the source directory and concatenate the data into a DataFrame.

        Returns:
        - pd.DataFrame: Concatenated DataFrame containing data from all files.
        """
        dataframes = []
        for filename in os.listdir(self.source_directory):
            file_path = os.path.join(self.source_directory, filename)
            dataframes.append(self.read_file(file_path))
        return pd.concat(dataframes).fillna(0)
    
    def get_data(self):
        """
        Retrieve and process data.

        Returns:
        - Any: Processed data.
        """
        df = self._read_dir()
        df = self._process_data(df)
        return df