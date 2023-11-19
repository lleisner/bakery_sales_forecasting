import pandas as pd
import ferien
from data_provider.sub_providers.base_provider import BaseProvider
from utils.hourly_converter import convert_to_hourly

class FerienDataProvider(BaseProvider):
    
    def get_data(self):
        return self.process()
    
    def process(self, start: str='2017-01-01', end: str='2023-09-01') -> pd.DataFrame:
        """
        Retrieves and compiles vacation data for different states within a specified date range using the ferien-api

        Args:
            start (str, optional): The start date in 'YYYY-MM-DD' format (default is '2017-01-01').
            end (str, optional): The end date in 'YYYY-MM-DD' format (default is '2023-09-31').

        Returns:
            pd.DataFrame: A DataFrame with states as columns and binary values indicating vacation days (1) or non-vacation days (0).
        """
        state_codes = ferien.state_codes()
        all_days = pd.date_range(start, end)

        all_states = pd.DataFrame(index=all_days)

        for state in state_codes:
            vacations = ferien.state_vacations(state)
            vacation_periods = []

            for vacation in vacations:
                start_date = vacation.start.date()
                end_date = vacation.end.date()
                vacation_periods.append((start_date, end_date))

            is_vacation = all_states.index.to_series().apply(lambda x: any(start_date <= x.date() <= end_date for start_date, end_date in vacation_periods))

            all_states[state] = is_vacation.astype(int)
        
        all_states = convert_to_hourly(all_states, fill_method='ffill')
        return all_states
    
    
if __name__ == "__main__":
    processor = FerienDataProvider()
    df = processor.get_data()
    print(df)