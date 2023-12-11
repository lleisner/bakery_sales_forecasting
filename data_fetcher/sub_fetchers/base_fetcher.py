from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import socket

class BaseFetcher:
    def __init__(self):
        self.driver = self.setup_driver()
    
    def setup_driver(self):
        """
        Set up a headless Chrome WebDriver instance.
        
        Returns:
            driver (WebDriver): A configured WebDriver instance, or None if an error occurs.
        """
        try:
            options = webdriver.ChromeOptions()
            options.add_argument('--headless')

            service = Service(self.get_driver_path())
            
            driver = webdriver.Chrome(options=options, service=service)

            return driver
        except Exception as e:
            print(f"An error occurred while setting up the driver: {e}")
            return None
        
    def get_driver_path(self):
        hostname = socket.gethostname()
        driver_path = None
        if hostname == "DESKTOP-4FMH331":
            driver_path = '/Users/lorenzleisner/Downloads/chromedriver-mac-x64/chromedriver'
        elif hostname == "Lorenzs-MBP.fritz.box":
            driver_path = '/home/lleisner/chromedriver/stable/chromedriver-linux64/chromedriver'
        return None