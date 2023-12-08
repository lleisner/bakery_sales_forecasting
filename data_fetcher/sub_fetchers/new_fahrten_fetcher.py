import time
import os.path

from bs4 import BeautifulSoup
import pandas as pd
from collections import defaultdict
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time
from datetime import datetime, timedelta
from functools import partial

    
class NewFahrtenFetcher:
    def __init__(self):
        self.driver = self.setup_driver()
        self.driver.get('https://www.spiekeroog.de/buchung/')

    def setup_driver(self):
        try:
            options = webdriver.ChromeOptions()
            options.add_argument('--headless')
           # driver_path = '/home/lleisner/chromedriver/stable/chromedriver-linux64/chromedriver'
            driver_path = '/Users/lorenzleisner/Downloads/chromedriver-mac-x64/chromedriver'
            
            driver = webdriver.Chrome(driver_path, options=options)
            return driver
        except Exception as e:
            print(f"An error occurred while setting up the driver: {e}")
            return None
        
    def _set_date(self, date):
        def find_input(driver):
            element = driver.find_element(By.CSS_SELECTOR, 'input[placeholder="dd.mm.YYYY"]')
            element.send_keys(Keys.RETURN)
            return element
        
        date = date.strftime('%d.%m.%Y')
        input = find_input(self.driver)
        self.driver.execute_script("arguments[0].value = '';", input)
        input.send_keys(date)
        
        time.sleep(1)
        input = find_input(self.driver)
        
        return date, self.driver.page_source

    def get_hin_ruck_fahrten(self, html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        fahrten = soup.find_all(class_='col-lg-6 col-md-6 col-sm-6 col-xs-12')
        hinfahrten = fahrten[0].find_all(class_='fahrt')
        ruckfahrten = fahrten[1].find_all(class_='fahrt')
        
        return hinfahrten, ruckfahrten    
    
    def process_single_fahrt(self, fahrt):
        departure_time_element = fahrt.find(class_='col-lg-3')
        departure_time = departure_time_element.get_text(strip=True).replace('Uhr', '')
        schiff_element = fahrt.find('div', class_='col-lg-1 col-md-1 col-sm-1 xol-xs-1')
        schiff = schiff_element.get_text(strip=True)
        
        booking_element = fahrt.find('div', class_='pull-right')
        booking = booking_element.img.get('src').split('_')[-1].replace('.png', '')
        
        return departure_time, schiff, booking
    
    def process_fahrten_list(self, fahrten_list):
        fahrten_data = []
        for fahrt in fahrten_list:
            departure_time, schiff, booking = self.process_single_fahrt(fahrt)    

            data = {
                'Zeit': departure_time,
                'Schiff': schiff,
                'Auslastung': booking
            }
            fahrten_data.append(data)
        return pd.DataFrame(fahrten_data)
    
    
    def count_fahrten_per_hour(self, date, fahrten, direction):
        def calculate_duration(row, date, direction, hourly_counts, prev):
            zeit = datetime.strptime(f'{date} {row["Zeit"]}', "%d.%m.%Y %H:%M")
            schiff = row["Schiff"]
            duration = 0
            
            if direction == "an":
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
            
            return prev  
        
        hourly_counts = defaultdict(lambda: defaultdict(int)) 
        prev = datetime.strptime(f'{date} 00:00', "%d.%m.%Y %H:%M")
        
        calculate_duration_partial = partial(calculate_duration, date=date, direction=direction, hourly_counts=hourly_counts, prev=prev)

        fahrten.apply(calculate_duration_partial, axis=1)
        
        df = pd.DataFrame.from_dict(hourly_counts, orient='index')
        
        desired_column_order = [f"SP1_{direction}", f"SP2_{direction}", f"SP4_{direction}", f"WEX_{direction}"]

        df.columns = [f"{col}_{direction}" for col in df.columns]
        df = df.reindex(columns=desired_column_order)
        df.fillna(0, inplace=True)
        
        return df
    

    def get_data(self, num_days=3, drop_WEX=True):
        all_days = []
        start = datetime.combine(datetime.now() + timedelta(days=1), datetime.min.time())
        
        for day in range(num_days):
            date, html_content = self._set_date(date=start + timedelta(days=day))
            
            hin, ruck = self.get_hin_ruck_fahrten(html_content=html_content)
            hin, ruck = self.process_fahrten_list(hin), self.process_fahrten_list(ruck)
            
            an, ab = self.count_fahrten_per_hour(date, hin, 'an'), self.count_fahrten_per_hour(date, ruck, 'ab')
            
            fahrten_per_day = pd.concat([an, ab], axis=1)
            fahrten_per_day.index = pd.to_datetime(fahrten_per_day.index, format="%d.%m.%Y", errors='coerce')
            
            all_days.append(fahrten_per_day)
            
        self.driver.quit()
        
        all_days = pd.concat(all_days).sort_index()
        all_days = all_days.resample('H', origin=start)
   
        all_days = all_days.asfreq().fillna(0)
        
        current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
        end_time = current_hour + timedelta(days=num_days)

        time_range = pd.date_range(start=current_hour, end=end_time, freq='H')
        #all_days = all_days.reindex(time_range, fill_value=0)
        
        
        if drop_WEX:
            all_days.drop(columns=['WEX_an', 'WEX_ab'], inplace=True)
            
        return all_days
        
if __name__ == "__main__":
    fetcher = NewFahrtenFetcher()
    df = fetcher.get_data()
    print(df)