import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler, SplineTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from models.iTransformer.cyclic_encoder import CyclicEncoder

class DataLoader(object):
    """Data loader parent class"""
    
    def __init__(
        self,
        data_path,
        datetime_col,
        numerical_cov_cols,
        categorical_cov_cols,
        cyclic_cov_cols,
        timeseries_cols,
        train_range,
        val_range,
        test_range,
        hist_len,
        pred_len,
        batch_size,
        stride=1,
        sample_rate=1,
        steps_per_epoch=None,
        validation_steps=None,
        freq='H',
        normalize=True,
    ):
        """_summary_

        Args:
            data_path (str): path to source csv file
            datetime_col (list): column name of datetime col
            numerical_cov_cols (list): column names for numerical covariates
            categorical_cov_cols (list): column names for categorical covariates
            cyclic_cov_cols (list): column  names for cyclical covariates
            timeseries_cols (list): column names of timeseries variates to be predicted
            train_range (tuple): train ranges
            val_range (tuple): validation ranges
            test_range (tuple): test ranges
            hist_len (int): size of lookback window
            pred_len (int): size of forecast horizon
            batch_size (int): batch size
            stride (int, optional): stride, i.e. timesteps to skip between the creation of each window. Defaults to 1.
            sample_rate (int, optional): sample rate, i.e. period between individual timesteps within a window. Defaults to 1.
            normalize (bool, optional): normalize data or not. Defaults to True.
            freq (str, optional): _description_. Defaults to 'H'.
        """
        self.data_df = pd.read_csv(open(data_path, 'r'))
        self.data_df.fillna(0, inplace=True)
        self.data_df.set_index(pd.DatetimeIndex(self.data_df[datetime_col], inplace=True))
        
        self.num_cov_cols = numerical_cov_cols
        self.cat_cov_cols = categorical_cov_cols
        self.cyc_cov_cols = cyclic_cov_cols
        self.ts_cols = timeseries_cols if timeseries_cols else self.data_df.columns.tolist()
        
        self.train_range = train_range
        self.val_range = val_range
        self.test_range = test_range
        
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.batch_size = batch_size
        
        self.stride = stride
        self.sampling_rate = sample_rate
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps
        self.freq = freq
        
        self.normalize = normalize
        
        self._add_missing_cols()
        
        self._create_temporal_features()
        
        self._arrange_data()
        
        if self.normalize: 
            self._normalize_data()
            
        self._create_lagged_features()
        
    def _add_missing_cols(self):
        pass
        
    def _arrange_data(self):
        self.data_df = self.data_df[self.ts_cols + [col for col in self.data_df if col not in self.ts_cols]]
    
    def _create_temporal_features(self) -> None:
        index = self.data_df.index
        features = {
            'hour': index.hour,
            'dayofweek': index.dayofweek,
            'dayofmonth': index.day,
            'dayofyear': index.dayofyear,
            'month': index.month,
            'quarter': index.quarter,
        }
        time_df = pd.DataFrame(features, index=index)
        self.cyc_cov_cols.extend(time_df.columns.tolist())
        self.data_df = pd.concat([time_df, self.data_df], axis=1)
        
    def _normalize_data(self, remainder="passthrough"):
        """Normalize and encode the data
        
        Args:
            remainder (str, optional): "drop" or "passthrough" unspecified columns. Defaults to "passthrough"
            
        Returns:
            None
            
        Apply specific transformations to the different subsets of the data:
        - MinMax scaling to target and numerical columns.
        - One-Hot encoding to categorical columns.
        - Custom cyclic spline encoding to cyclic columns.
        
        Columns that appear in the data but are not specified in the __init__() 
        are either dropped or left as is, depending on the remainder parameter.
        After transformation, update the self.data_df and other affected class attributes.
        """
        
        # List to store transformer configurations
        transformers_list = []
        
        # Dictionary mapping column types to their transformers and prefixes
        cols_dict = {
            'ts_cols': (StandardScaler(), "target"), 
            'num_cov_cols': (StandardScaler(), "numerical"),
            'cat_cov_cols': (OneHotEncoder(), "categorical"),
            'cyc_cov_cols': (StandardScaler(), "cyclic"),
        }
        
        # Append appropriate transformers to the list if columns are available
        for attr, (transformer, prefix) in cols_dict.items():
            cols = getattr(self, attr)
            if cols:
                transformers_list.append((prefix, transformer, cols))

        # Create a ColumnTransformer with the configured transformers. Not specified columns are left as is.
        transformer = ColumnTransformer(
            transformers=transformers_list,
            remainder=remainder,
        )

        # Apply the ColumnTransformer to the dataframe
        transformed_df = transformer.fit_transform(self.data_df)
        # Retrieve the new column names after transformation
        column_names = transformer.get_feature_names_out()
        # Update the dataframe with transformed data and new column names
        self.data_df = pd.DataFrame(transformed_df, columns=column_names, index=self.data_df.index)
        
        # Update class attributes with new column names
        for attr, prefix in cols_dict.items():
            setattr(self, attr, [col for col in column_names if col.startswith(prefix[1])])
            
    def _create_lagged_features(self, include_mean_max_min=False):
        """Creates lagged and rolling statistical features (mean, max, min) for target variables

        Args:
            targets (pd.DataFrame): DataFrame with target variables

        Returns:
            pd.DataFrame: DataFrame with original targets shifted and their rolling mean, max, and min.
        
        Assumes 'self.pred_len' is set for lag and rolling window size.
        Adds the lagged features at the end of the dataframe, which is important for 
        the batch_x, batch_x_mark separation for the iTransformer later on    
        """
        targets = self.data_df[self.ts_cols]
        feature_frames = []   
        lag = self.pred_len * self.sampling_rate
        if include_mean_max_min:
            operations = {
                'rolling_mean': 'mean',
                'rolling_max': 'max',
                'rolling_min': 'min',
            }
            
            for op_name, op in operations.items():
                result = getattr(targets.shift(lag).rolling(window=self.hist_len), op)()
                result.columns = [f"{col}_{op_name}_{self.hist_len}h" for col in result.columns]
                feature_frames.append(result)
            
        lagged = targets.shift(lag).rename(columns=lambda x: f"lagged_{lag}h_{x}")
        feature_frames.append(lagged)
        
        lagged_features =  pd.concat(feature_frames, axis=1)
        self.data_df = pd.concat([self.data_df, lagged_features], axis=1).dropna()
            
            
    def get_train_test_splits(self):
        pass
    
    def get_features_names_out(self):
        return self.data_df.columns
    
    def get_data(self):
        return self.data_df