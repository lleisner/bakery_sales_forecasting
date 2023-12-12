from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import socket
import os
from abc import ABC, abstractmethod
import pandas as pd

class BaseFetcher(ABC):
    def update_csv(self, data_directory, filename):
        file_path = os.path.join(data_directory, filename)
        new_data = self.get_data()
        old_data = pd.read_csv(file_path, index_col=0, parse_dates=True)
        updated_data = new_data.combine_first(old_data)
        updated_data.to_csv(file_path)

    @abstractmethod
    def get_data(self):
        pass