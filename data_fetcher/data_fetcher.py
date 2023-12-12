import pandas as pd
import os
import logging

from data_fetcher.sub_fetchers import (
    SalesFetcher,
    FahrtenFetcher,
    WeatherFetcher,
    FerienFetcher
)


class DataFetcher:
    def __init__(self, data_directory='data/new_data/'):
        self.fetchers = {
            'sales': SalesFetcher(data_directory),
            'fahrten': FahrtenFetcher(data_directory),
            'weather': WeatherFetcher(data_directory),
            #'ferien': FerienFetcher(data_directory)
        }
        self.data_directory = data_directory

        # Set up logging configurations
        logging.basicConfig(filename='logs/update/update.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def update_database(self):
        for key, fetcher in self.fetchers.items():
            try:
                fetcher.update_csv(key)
                logging.info(f"Update successful for fetcher '{key}'")
            except Exception as e:
                logging.error(f"Update failed for fetcher '{key}': {str(e)}")


if __name__ == "__main__":
    fetcher = DataFetcher()
    fetcher.update_database()