from selenium import webdriver

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
           # driver_path = '/home/lleisner/chromedriver/stable/chromedriver-linux64/chromedriver'
            driver_path = '/Users/lorenzleisner/Downloads/chromedriver-mac-x64/chromedriver'
            
            driver = webdriver.Chrome(driver_path, options=options)
            #driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)
            return driver
        except Exception as e:
            print(f"An error occurred while setting up the driver: {e}")
            return None
        