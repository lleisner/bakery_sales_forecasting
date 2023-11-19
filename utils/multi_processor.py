import pandas as pd
import os
from abc import ABC, abstractmethod
    
class BaseProvider(ABC):
    @abstractmethod
    def get_data(self):
        pass

    
class DataProvider(BaseProvider):
    def __init__(self, source_directory):
        self.source_directory = source_directory
        
    @abstractmethod
    def read_file(self):
        pass
    
    @abstractmethod
    def process_data(self, df):
        pass
    
    def read_dir(self):
        dataframes = []
        for filename in os.listdir(self.source_directory):
            file_path = os.path.join(self.source_directory, filename)
            dataframes.append(self.read_file(file_path))
        return pd.concat(dataframes).fillna(0)
    
    def get_data(self):
        df = self.read_dir()
        df = self.process_data(df)
        return df