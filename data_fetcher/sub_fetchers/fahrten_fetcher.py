
from typing import Optional
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
import pandas as pd
from collections import defaultdict
from datetime import datetime, timedelta
import time
import os



class FahrtenFetcher():
    def __init__(self):
        self.driver = self.setup_driver()
        self.driver.get('https://www.spiekeroog.de/buchung/')

    def setup_driver(self):
        """
        Set up a headless Chrome WebDriver instance.
        
        Returns:
            driver (WebDriver): A configured WebDriver instance, or None if an error occurs.
        """
        try:
            options = webdriver.ChromeOptions()
            options.add_argument('--headless')
            driver_path = Service('/home/lleisner/chromedriver-linux64/chromedriver')
          #  driver_path = '/Users/lorenzleisner/Downloads/chromedriver-mac-x64/chromedriver'
            driver = webdriver.Chrome(options=options, service=driver_path)
            return driver
        except Exception as e:
            print(f"An error occurred while setting up the driver: {e}")
            return None
        
    def _find_element(self):
        # Helper function to refind stale input element and return input
        element = self.driver.find_element(By.CSS_SELECTOR, 'input[placeholder="dd.mm.YYYY"]')
        element.send_keys(Keys.RETURN)
        return element

    def _set_date(self, new_date: str) -> Optional[str]:
        """
        Get the HTML content of the page after updating the date.
        
        Args:
            new_date (str): The new date to input.
            
        Returns:
            html_content (str): The HTML content of the updated page, or None if an error occurs.
        """
        input_element = self._find_element()
        self.driver.execute_script("arguments[0].value = '';", input_element)
        input_element.send_keys(new_date)
        time.sleep(1)
        input_element = self._find_element()

        return self.driver.page_source
        

    def get_hin_ruck_fahrten(self, html_content: str) -> tuple:
        """
        Extract hinfahrten and ruckfahrten fahrt elements from HTML content.

        Args:
            html_content (str): HTML content to parse.

        Returns:
            tuple: A tuple containing lists of hinfahrten and ruckfahrten fahrt elements, or a tuple of empty lists if an error occurs
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        fahrten = soup.find_all(class_='col-lg-6 col-md-6 col-sm-6 col-xs-12')
        hinfahrten = fahrten[0].find_all(class_='fahrt')
        ruckfahrten = fahrten[1].find_all(class_='fahrt')
        return hinfahrten, ruckfahrten
 
 
    def _process_fahrt(self, fahrt_element: BeautifulSoup) -> tuple:
        """
        Process a single fahrt element and extract relevant information.

        Args:
            fahrt_element (BeautifulSoup): A BeautifulSoup Tag representing a fahrt element.

        Returns:
            tuple: A tuple containing departure time, ship, and booking information, or a tuple of Nones if an error occurs
        """
        departure_time_element = fahrt_element.find(class_='col-lg-3')
        departure_time = departure_time_element.get_text(strip=True).replace('Uhr', '')
        schiff_element = fahrt_element.find('div', class_='col-lg-1 col-md-1 col-sm-1 xol-xs-1')
        schiff = schiff_element.get_text(strip=True)
        booking_element = fahrt_element.find('div', class_='pull-right')
        booking = booking_element.img.get('src').split('_')[-1].replace('.png', '')
        return departure_time, schiff, booking


    def merge_fahrten_to_df(self, fahrt_elements: list) -> pd.DataFrame:
        """
        Process a list of fahrt elements and create a DataFrame.

        Args:
            fahrt_elements (list): List of BeautifulSoup Tags representing fahrt elements.

        Returns:
            pd.DataFrame: A DataFrame containing processed fahrt data, or an empty DataFrame if an error occurs
        """
        fahrt_data = []
        for fahrt_element in fahrt_elements:
            departure_time, schiff, booking = self._process_fahrt(fahrt_element)    

            data = {
                'Zeit': departure_time,
                'Schiff': schiff,
                'Auslastung': booking
            }
            fahrt_data.append(data)
    
        return pd.DataFrame(fahrt_data)
        
        
    def _count_fahrten_per_hour(self, date: str, fahrten: pd.DataFrame, direction: str) -> pd.DataFrame:
        hourly_counts = defaultdict(lambda: defaultdict(int))
        prev = datetime.strptime(f'{date} 00:00', "%d.%m.%Y %H:%M")
        for index, row in fahrten.iterrows():
            zeit = datetime.strptime(f'{date} {row["Zeit"]}', "%d.%m.%Y %H:%M")
            schiff = row["Schiff"]
            duration = 0
            if direction=="an":
                if schiff == "WEX":
                    duration = 25
                elif zeit < prev:
                    hourly_counts[(zeit + timedelta(minutes=45)).replace(minute=0)][schiff] = 0
                    duration = 120
                else:
                    duration = 45

            prev = zeit
            zeit = zeit + timedelta(minutes=duration)
            hourly_counts[zeit.replace(minute=0)][schiff] = 1
    
        df = pd.DataFrame.from_dict(hourly_counts, orient="index")
        desired_column_order = [f"SP1_{direction}", f"SP2_{direction}", f"SP4_{direction}", f"WEX_{direction}"]
        df.columns = [f"{col}_{direction}" for col in df.columns]
        df = df.reindex(columns=desired_column_order)
        df.fillna(0, inplace=True)
        return df


    def get_new_fahrzeiten(self, num_days: int, drop_WEX: bool = True) -> pd.DataFrame:
        all_days = []
        for day in range(num_days):
            attempts = 0
            while attempts < 5:  # Maximum number of attempts per day
                try:
                    new_date = datetime.now() + timedelta(days=day+1)
                    new_date = new_date.strftime('%d.%m.%Y')

                    html_content = self._set_date(new_date)
                    hinfahrten, ruckfahrten = self.get_hin_ruck_fahrten(html_content)

                    hin = self.merge_fahrten_to_df(hinfahrten)
                    ruck = self.merge_fahrten_to_df(ruckfahrten)

                    an = self._count_fahrten_per_hour(new_date, hin, "an")
                    ab = self._count_fahrten_per_hour(new_date, ruck, "ab")

                    df = pd.concat([an, ab], axis=1)
                    df.index = pd.to_datetime(df.index, format="%d.%m.%Y", errors='coerce')
                    
                    break  # Exit the retry loop if successful
                    
                except Exception as e:
                    attempts += 1
                    print(f"An error occurred in iteration {day}, attempt {attempts}: {e}")
                    time.sleep(1)  # Wait for 1 second before retrying
                    self.driver = self.setup_driver()
                    self.driver.get('https://www.spiekeroog.de/buchung/')
            all_days.append(df)

            if attempts == 5:  # If maximum attempts reached without success
                print(f"Max attempts reached for iteration {day}. No data retrieved.")

        if all_days:
            all_data = pd.concat(all_days)
            all_data.sort_index(inplace=True)
            all_data = all_data.resample('H').asfreq().fillna(0)
            
            current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
            end_time = current_hour + timedelta(days=num_days-1)

            time_range = pd.date_range(start=current_hour, end=end_time, freq='H')
            all_data = all_data.reindex(time_range, fill_value=0)

            if drop_WEX:
                all_data.drop(columns=['WEX_an', 'WEX_ab'], inplace=True)
            return all_data
        else:
            print("No data was retrieved.")
            return pd.DataFrame() 



    def main(self):
        df = self.get_new_fahrzeiten(num_days=3, drop_WEX=False)
        print(df)
        
        


## Run selenium and chrome driver to scrape data from cloudbytes.dev
import time
import os.path

from bs4 import BeautifulSoup
import requests
import pandas as pd
from collections import defaultdict
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time
from datetime import datetime, timedelta
from typing import Optional
from bs4.element import Tag

from data_fetcher.sub_fetchers.base_fetcher import BaseFetcher

from functools import partial


    
def get_driver() -> Optional[webdriver.Chrome]:
    """Return a configured Chrome web driver."""
    base_fetcher = BaseFetcher()
    driver = base_fetcher.driver
    driver.get('https://www.spiekeroog.de/buchung/')
    return driver


def get_html_for_date(driver: Optional[webdriver.Chrome], new_date: str) -> Optional[str]:
    """
    Get the HTML content of the page after updating the date.
    
    Args:
        driver (WebDriver): The WebDriver instance.
        new_date (str): The new date to input.
        
    Returns:
        html_content (str): The HTML content of the updated page, or None if an error occurs.
    """
    def find_element():
        # Helper function to refind stale input element and return input
        element = driver.find_element(By.CSS_SELECTOR, 'input[placeholder="dd.mm.YYYY"]')
        element.send_keys(Keys.RETURN)
        return element
    
    try:
        # Find the input element with placeholder "dd.mm.YYYY" and data-type attribute
        input_element = find_element()

        # Clear current value
        driver.execute_script("arguments[0].value = '';", input_element)
        # Send new date
        input_element.send_keys(new_date)
        # Re-find the input element after the page action
        input_element = find_element()
        time.sleep(1)
        # Re-find the input element and confirm new date
        input_element = find_element()

        return driver.page_source
    
    except Exception as e:
        print(f"An error occurred while getting HTML for date: {e}")
        return None
    
def get_html_content(driver: Optional[webdriver.Chrome], day_offset: int=1) -> tuple:
    """
    Retrieve html content for the fahrzeiten from a specific url
    
    Args:
        day_offset (int): Int representing the offset in days (0=today) for the information to be retrieved
    
    Return:
        tuple: A tuple containing the date and the html_content of the retrieved data, or a tuple of Nones if an error occurs
    """
    try:
        new_date = datetime.now() + timedelta(days=day_offset)
        new_date = new_date.strftime('%d.%m.%Y')

        html_content = get_html_for_date(driver, new_date)

        return new_date, html_content
    except Exception as e:
        print(f"An error occured while retrieving html content: {e}")
        return None, None
    
def extract_fahrt_elements(html_content: str) -> tuple:
    """
    Extract hinfahrten and ruckfahrten fahrt elements from HTML content.

    Args:
        html_content (str): HTML content to parse.

    Returns:
        tuple: A tuple containing lists of hinfahrten and ruckfahrten fahrt elements, or a tuple of empty lists if an error occurs
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        fahrten = soup.find_all(class_='col-lg-6 col-md-6 col-sm-6 col-xs-12')
        hinfahrten = fahrten[0].find_all(class_='fahrt')
        ruckfahrten = fahrten[1].find_all(class_='fahrt')
        return hinfahrten, ruckfahrten
    except Exception as e:
        print(f"An error occured while extracting fahrt elements: {e}")
        return [], []
    
def process_fahrt_element(fahrt_element: Tag) -> tuple:
    """
    Process a single fahrt element and extract relevant information.

    Args:
        fahrt_element (Tag): A BeautifulSoup Tag representing a fahrt element.

    Returns:
        tuple: A tuple containing departure time, ship, and booking information, or a tuple of Nones if an error occurs
    """
    try:
        departure_time_element = fahrt_element.find(class_='col-lg-3')
        departure_time = departure_time_element.get_text(strip=True).replace('Uhr', '')
        schiff_element = fahrt_element.find('div', class_='col-lg-1 col-md-1 col-sm-1 xol-xs-1')
        schiff = schiff_element.get_text(strip=True)
        booking_element = fahrt_element.find('div', class_='pull-right')
        booking = booking_element.img.get('src').split('_')[-1].replace('.png', '')
        return departure_time, schiff, booking
    except Exception as e:
        print(f"An error occured while proccessing a fahrt element: {e}")
        return None, None, None


def process_fahrt_elements(fahrt_elements: list) -> pd.DataFrame:
    """
    Process a list of fahrt elements and create a DataFrame.

    Args:
        fahrt_elements (list): List of BeautifulSoup Tags representing fahrt elements.

    Returns:
        pd.DataFrame: A DataFrame containing processed fahrt data, or an empty DataFrame if an error occurs
    """
    try:
        fahrt_data = []
        for fahrt_element in fahrt_elements:
            departure_time, schiff, booking = process_fahrt_element(fahrt_element)    

            data = {
                'Zeit': departure_time,
                'Schiff': schiff,
                'Auslastung': booking
            }
            fahrt_data.append(data)
    
        return pd.DataFrame(fahrt_data)
    except Exception as e:
        print(f"An error occurred while processing a fahrt elements: {e}")
        return pd.DataFrame()
    


    

def generate_hourly_counts(date: str, fahrten: pd.DataFrame, direction: str) -> pd.DataFrame:
    """
    Generate a DataFrame with hourly counts for a given date and direction (an/ab) from a DataFrame of trips (fahrten).

    Args:
        date (str): The date for which counts are generated in the format "%d.%m.%Y".
        fahrten (pd.DataFrame): DataFrame containing trip data.
        direction (str): The direction of the trips ('anreise' or 'abreise').

    Returns:
        pd.DataFrame: A DataFrame with hourly counts as index and the specified direction as column.
    """
    # Initialize an empty dictionary to store counts
    hourly_count = {}
    for index, row in fahrten.iterrows():
        zeit = datetime.strptime(f'{date} {row["Zeit"]}', "%d.%m.%Y %H:%M").replace(minute=0, second=0)
        
        # Check if the timestamp is already in the dictionary, and if not, initialize it to 0
        if zeit not in hourly_count:
            hourly_count[zeit] = 0

        # Increment the count for the timestamp
        hourly_count[zeit] += 1
    df = pd.DataFrame.from_dict(hourly_count, orient='index', columns=[direction])
    return df


def count_fahrten_per_hour(date: str, fahrten: pd.DataFrame, direction: str) -> pd.DataFrame:
    """
    Count ship arrivals or departures per hour.

    Args:
        date (str): The date in the format "%d.%m.%Y".
        fahrten (pd.DataFrame): DataFrame containing ship data.
        direction (str, optional): "an" for arrivals (default) or "ab" for departures.

    Returns:
        pd.DataFrame: DataFrame with ship counts per hour.
    """
    hourly_counts = defaultdict(lambda: defaultdict(int))  # Use a nested defaultdict
    prev = datetime.strptime(f'{date} 00:00', "%d.%m.%Y %H:%M")
    for index, row in fahrten.iterrows():
        zeit = datetime.strptime(f'{date} {row["Zeit"]}', "%d.%m.%Y %H:%M")
        schiff = row["Schiff"]
        duration = 0
        # Take into account the time the ship needs to arrive
        if direction=="an":
            if schiff == "WEX": # WEX is an express ship that will only take 25 minutes (does not do sightseeing tours)
                duration = 25

            elif zeit < prev: # this trip is a sightseeing tour, it will take 2h to arrive
                hourly_counts[(zeit + timedelta(minutes=45)).replace(minute=0)][schiff] = 0
                duration = 120

            else: # a regular tour will take 45 minutes to arrive
                duration = 45

        prev = zeit
        zeit = zeit + timedelta(minutes = duration)
        hourly_counts[zeit.replace(minute=0)][schiff] = 1  # Increment the count
    
    # Convert the nested defaultdict to a DataFrame
    df = pd.DataFrame.from_dict(hourly_counts, orient="index")
    
    # Rename columns with the specified direction
    desired_column_order = [f"SP1_{direction}", f"SP2_{direction}", f"SP4_{direction}", f"WEX_{direction}"]

    df.columns = [f"{col}_{direction}" for col in df.columns]
    df = df.reindex(columns=desired_column_order)
    df.fillna(0, inplace=True)
    
    return df

    


def get_new_fahrzeiten(num_days: int, drop_WEX: bool=True) -> pd.DataFrame:
    """
    Retrieve and process fahrzeiten data for a specified number of days.

    Args:
        num_days (int): The number of days to retrieve data for (starting from today).

    Returns:
        pd.DataFrame: A DataFrame containing processed fahrzeiten data for all specified days.

    Note:
        This function uses a web driver to access and retrieve data from a website. It iterates through each
        day, extracts relevant data, processes it, and combines it into a final DataFrame. If any errors
        occur during the process, they are logged, and the function attempts to continue with the next day.
    """
    all_days = []
    driver = get_driver()
    for day in range(num_days):
        success = False
        attempts = 0

        while attempts < 10 and not success:  # Maximum number of attempts
            try:
                date, html_content = get_html_content(driver=driver, day_offset=day)
                hinfahrten, ruckfahrten = extract_fahrt_elements(html_content)

                hin = process_fahrt_elements(hinfahrten)
                ruck = process_fahrt_elements(ruckfahrten)

                an = count_fahrten_per_hour(date, hin, "an")
                ab = count_fahrten_per_hour(date, ruck, "ab")

                df = pd.concat([an, ab], axis=1)
                df.index = pd.to_datetime(df.index, format="%d.%m.%Y", errors='coerce')
                
                all_days.append(df)
                success = True  # Set success flag to True upon successful completion
                
            except Exception as e:
                attempts += 1
                print(f"An error occurred in iteration {day}, attempt {attempts}: {e}")
                time.sleep(1)  # Wait for 1 second before retrying

        if not success:
            print(f"Max attempts reached for iteration {day}. No data retrieved.")
    driver.quit()


    if all_days:
        all_data = pd.concat(all_days)
        all_data.sort_index(inplace=True)
        all_data = all_data.resample('H').asfreq().fillna(0)
        
        # Set the current hour and the end time 5 days from now
        current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
        end_time = current_hour + timedelta(days=num_days-1)

        # Resample the DataFrame to hourly intervals, forward fill, and fill missing values with zeros


        # Align data to the specified time range
        time_range = pd.date_range(start=current_hour, end=end_time, freq='H')
        all_data = all_data.reindex(time_range, fill_value=0)

        if drop_WEX:
            all_data.drop(columns=['WEX_an', 'WEX_ab'], inplace=True)
        return all_data
    else:
        print("No data was retrieved.")
        return pd.DataFrame() 
    
    
        
            
            
            
            
            
            
        
        


        
    
    

    
    


if __name__ == "__main__":
    fetcher = FahrtenFetcher()
    fetcher.main()
    
    ## CAREFUL !!! THE CLASS FAHRTENFETCHER DOES NOT WORK AS INTENDED YET !!!
    ## THE DATA IS NOT BEING PROCESSED CORRECTLY. 
    ## THE FUNCTIONS IMPLEMENTED SEPERATELY DO WORK AS INTENDED BUT NO CORRECT ERROR HANDLING
    
    df = get_new_fahrzeiten(num_days=3, drop_WEX=False)
    print(df)
    


    

