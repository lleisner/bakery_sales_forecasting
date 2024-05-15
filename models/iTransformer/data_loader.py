import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler, SplineTransformer, OneHotEncoder, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from utils.cyclic_encoder import CyclicEncoder
import warnings

# Ignore FutureWarnings in sklearn.utils.validation
warnings.filterwarnings(action='ignore', category=FutureWarning, module='sklearn.utils.validation')
    
class ITransformerData(object):
    """Data Loader class"""
    
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
        stride,
        sample_rate,
        batch_size,
        epoch_len,
        val_len,
        freq='H',
        normalize=True,     
        drop_remainder=False,
    ):
        """Initialize objects.

        Args:
            data_path (str): path to the csv file
            datetime_col (list): column name for datetime col 
            covariate_cols (list): column names for numerical covariates
            categorical_cov_cols (list): column names for categorical covariates
            circ_cov_cols (list): column names for circular covariates
            timeseries_cols (list): columns of timeseries to be predicted
            train_range (tuple): train ranges
            val_range (tuple): validation ranges
            test_range (tuple): test ranges
            hist_len (int): historical context length
            pred_len (int): prediction length
            batch_size (int): batch size
            epoch_len (int): steps per epoch
            val_len (int): steps per validation
            freq (str, optional): frequency of original data. Defaults to 'H'.
            normalize (bool, optional): normalize data or not. Defaults to True.
            drop_remainder (bool, optional): drop or pass through columns that are not specified in cols lists in the normalization process.
            
        Returns:
            None
        """
        self.data_df = pd.read_csv(open(data_path, 'r'))
        self.data_df.fillna(0, inplace=True)
        self.data_df.set_index(pd.DatetimeIndex(self.data_df[datetime_col]), inplace=True)
        self.data_df.drop(columns=[datetime_col], inplace=True)

        self.num_cov_cols = numerical_cov_cols
        self.cat_cov_cols = categorical_cov_cols
        self.cyc_cov_cols = cyclic_cov_cols if cyclic_cov_cols else []
        if timeseries_cols:
            self.ts_cols = timeseries_cols
        else:
            self.ts_cols = self.data_df.columns.tolist()
            #self.ts_cols = self.use_num_columns_as_ts_list()

        self.train_range = train_range
        self.val_range = val_range
        self.test_range = test_range
        
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.stride = stride
        self.sampling_rate = sample_rate
        self.batch_size = batch_size
        self.freq = freq
        self.normalize = normalize
        self.epoch_len = epoch_len
        self.val_len = val_len
        

        # Create temporal features from index
        self._create_temporal_features(self.data_df.index)

        # Get data in the right order for splitting into x, y, x_mark
        self.data_df = self.data_df[self.ts_cols + [col for col in self.data_df if col not in self.ts_cols]]

        if normalize:
            self._normalize() if drop_remainder else self._normalize("passthrough")
        

        self._create_lagged_features()
        
        
    def _create_temporal_features(self, data: pd.DatetimeIndex) -> pd.DataFrame:
        """Generate a DataFrame with temporal features extracted from a DatetimeIndex.

        Args:
            data (pd.DatetimeIndex): The input datetime index from which to extract the time-related features.

        Returns:
            pd.DataFrame: DataFrame with columns for hour, day of week, day of month, day of year, month, and quarter.
        """
        features = {
            'hour': data.hour,
            'dayofweek': data.dayofweek,
            'dayofmonth': data.day,
            'dayofyear': data.dayofyear,
            'month': data.month,
            'quarter': data.quarter,
        }
        time_df = pd.DataFrame(features, index=data)
        self.cyc_cov_cols.extend(time_df.columns.tolist())
        self.data_df = pd.concat([time_df, self.data_df], axis=1)
        
    
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
                result = getattr(targets.shift(1).rolling(window=lag), op)()
                result.columns = [f"{col}_{op_name}_{lag}h" for col in result.columns]
                feature_frames.append(result)
            
        lagged = targets.shift(lag).rename(columns=lambda x: f"lagged_{lag}h_{x}")
        feature_frames.append(lagged)
        
        lagged_features =  pd.concat(feature_frames, axis=1)
        self.data_df = pd.concat([self.data_df, lagged_features], axis=1).dropna()
        
    
    
    def _normalize(self, remainder="drop"):
        """Normalize and encode the data
        
        Args:
            remainder (str, optional): "drop" or "passthrough" unspecified columns. Defaults to "drop"
            
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
            'ts_cols': (RobustScaler(), "target"), 
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
        
    
    def make_dataset(self, start_index, end_index):
        dataset = tf.keras.utils.timeseries_dataset_from_array(
            data=self.data_df,
            targets=None,
            sequence_length=self.hist_len,
            sequence_stride=self.stride,
            sampling_rate=self.sampling_rate,
            batch_size=None,
            shuffle=False,
            start_index=start_index,
            end_index=end_index,
        )
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        dataset = dataset.map(self.split_batch)
        return dataset
        
        
    def split_batch(self, batch):
        num_targets = len(self.ts_cols)
        batch_y = batch[:, -self.pred_len:, :num_targets]
        batch_x = batch[:, :, -num_targets:]
        batch_x_mark = batch[:, :, num_targets:-num_targets]
        
        batch_y.set_shape([None, self.pred_len, None])
        batch_x.set_shape([None, self.hist_len, None])
        batch_x_mark.set_shape([None, self.hist_len, None])

        return batch_x, batch_y, batch_x_mark
    
        
    def get_train_test_splits(self):
        train_data = self.make_dataset(self.train_range[0], self.train_range[1])
        val_data = self.make_dataset(self.val_range[0], self.val_range[1])
        test_data = self.make_dataset(self.test_range[0], self.test_range[1])
        return train_data, val_data, test_data
    

    def get_data(self):
        return self.data_df
    
    def get_feature_names_out(self):
        return self.data_df.columns
    
    def use_num_columns_as_ts_list(self):
        return [col for col in self.data_df.columns if str(col).isnumeric()]

    def __repr__(self):
        return (f"ITransformerData(data_path={self.data_path!r}, datetime_col={self.datetime_col!r}, "
                f"num_cov_cols={len(self.numerical_cov_cols) if self.numerical_cov_cols else 0}, "
                f"cat_cov_cols={len(self.categorical_cov_cols) if self.categorical_cov_cols else 0}, "
                f"cyc_cov_cols={len(self.cyclic_cov_cols) if self.cyclic_cov_cols else 0}, "
                f"timeseries_cols={len(self.timeseries_cols) if self.timeseries_cols else 0}, "
                f"train_range={self.train_range}, val_range={self.val_range}, test_range={self.test_range}, "
                f"hist_len={self.hist_len}, pred_len={self.pred_len}, stride={self.stride}, "
                f"sample_rate={self.sample_rate}, batch_size={self.batch_size}, "
                f"epoch_len={self.epoch_len}, val_len={self.val_len}, normalize={self.normalize})")
    
    """DEPRECEATED"""
    
    """
    def feature_target_split(self):
        y = self.data_df[self.ts_cols]
        
        # Construct new names for the shifted and rolled columns
        lagged_names = {col: f"{col}_lagged_{self.pred_len}h" for col in y.columns}
        mean_names = {col: f"{col}_rolling_mean_{self.pred_len}h" for col in y.columns}
        max_names = {col: f"{col}_rolling_max_{self.pred_len}h" for col in y.columns}
        min_names = {col: f"{col}_rolling_min_{self.pred_len}h" for col in y.columns}
        
        # Apply renaming with dictionary mappings
        lagged_features = pd.concat(
            [
                self.data_df,
                y.shift(self.pred_len).rename(columns=lagged_names),
                y.shift(1).rolling(self.pred_len).mean().rename(columns=mean_names),
                y.shift(1).rolling(self.pred_len).max().rename(columns=max_names),
                y.shift(1).rolling(self.pred_len).min().rename(columns=min_names),
            ],
            axis="columns",
        )
        lagged_features.dropna(inplace=True)
        
        # Now, we don't drop `self.ts_cols` from `self.data_df` but from `lagged_features` for X
        X = lagged_features.drop(columns=self.ts_cols)
        
        # Ensure y is aligned with the processed features
        y= lagged_features[self.ts_cols]

        return X, y
    
    def prepare_windowed_datasets(self, train, val, test):
        window_generator = WindowGenerator(
            hist_len=self.hist_len,
            pred_len=self.pred_len,
            shift=self.stride, 
            train_df=train,
            val_df=val,
            test_df=test,
            label_columns=self.ts_cols,  
        )
        
        train_dataset = window_generator.train
        val_dataset = window_generator.val
        test_dataset = window_generator.test
        return train_dataset, val_dataset, test_dataset
    
    def get_time_series_split(self, X, y):
        ts_cv = TimeSeriesSplit(
            n_splits=5,
            gap=self.pred_len,
            max_train_size=None,
            test_size=None,
        )
        all_splits = list(ts_cv.split(X, y))
        return all_splits
    
    def prepare_data(self):
        X, y = self.feature_target_split()
        print(self.get_time_series_split(X, y))
        print(X)
        print(y)
        
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, shuffle=False)
        
        window_gen = WindowGenerator(
            hist_len=self.hist_len,
            pred_len=self.pred_len,
            shift=self.stride,
            batch_size=self.batch_size,
            train_df=(X_train, y_train),
            val_df=(X_val, y_val),
            test_df=(X_test, y_test),
        )
        train_data = window_gen.train
        val_data = window_gen.val
        test_data = window_gen.test
        
        return train_data, val_data, test_data
    
    def make_dataset(self, start_index, end_index):
        ts_dfa = tf.keras.utils.timeseries_dataset_from_array
        
        num_samples = end_index - start_index - self.hist_len + 1
        num_batches = num_samples // self.batch_size
        end_index = start_index + num_batches * self.batch_size + self.hist_len - 1 

        input_dataset = ts_dfa(
            data=self.data_df.drop(columns=self.ts_cols),
            targets=None,
            sequence_length=self.hist_len,
            sequence_stride=self.stride,
            sampling_rate=1,
            batch_size=self.batch_size,
            shuffle=False,
            start_index=start_index,
            end_index=end_index,
        )
        target_start_index = start_index + (self.hist_len - self.pred_len)
        target_dataset = ts_dfa(
            data=self.data_df[self.ts_cols],
            targets=None,
            sequence_length=self.pred_len,
            sequence_stride=self.stride,
            sampling_rate=1,
            batch_size=self.batch_size,
            shuffle=False,
            start_index=target_start_index,
            end_index=end_index,
        )    
        num_targets = len(self.ts_cols)
        num_features = len(self.data_df.columns) - num_targets
        print(num_features, num_targets)
        input_dataset = input_dataset.map(lambda x: tf.ensure_shape(x, (self.batch_size, self.hist_len, num_features)))
        target_dataset = target_dataset.map(lambda x: tf.ensure_shape(x, (self.batch_size, self.pred_len, num_targets)))
        
        variates, covariates = input_dataset.map(self.split_features)
        
        batch_x, batch_x_mark, batch_y = variates, covariates, target_dataset
        
        return batch_x, batch_y, batch_x_mark
        
        """