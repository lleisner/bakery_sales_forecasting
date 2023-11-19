import pandas as pd

class CustomDataFrame(pd.DataFrame):
    _metadata = ["added_property"]
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    @property
    def _constructor(self):
        return CustomDataFrame

    def filter_range(self, *intervals):
        """
        Filter the DataFrame by selected columns based on defined intervals.

        Args:
            *intervals (tuple): Intervals as tuples of start and end values.

        Returns:
            CustomDataFrame: A new DataFrame with columns within the specified intervals.
        """
        selected_cols = set()
        for start, end in intervals:
            selected_cols.update(str(col) for col in range(start, end + 1) if start <= col <= end)
        return self[selected_cols]


    def sort_cols(self):
        """
        Sorts the columns of the DataFrame in ascending order.

        Returns:
            CustomDataFrame: A new DataFrame with columns sorted.
        """
        return self.reindex(sorted(self.columns, key=int), axis=1)


    def negatives_to_zero(self):
        """
        Replaces negative values with 0.

        Returns:
            CustomDataFrame: A new DataFrame with negative values replaced by 0.
        """
        return self.clip(lower=0)
    

    
    @classmethod
    def from_pd_dataframe(cls, dataframe):
        """
        Convert a regular DataFrame to a CustomDataFrame.

        Args:
            dataframe (pd.DataFrame): The DataFrame to be converted.

        Returns:
            CustomDataFrame: An instance of CustomDataFrame.
        """
        return cls(dataframe)