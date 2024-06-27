import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler, SplineTransformer, OneHotEncoder, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from utils.cyclic_encoder import CyclicEncoder
from models.data_loader import DataLoader
import warnings

# Ignore FutureWarnings in sklearn.utils.validation
warnings.filterwarnings(action='ignore', category=FutureWarning, module='sklearn.utils.validation')
    
class ITransformerData(DataLoader):
    """Data Loader class"""
    
    def __init__(self, drop_remainder=False, *args, **kwargs):
        """
        Initialize objects

        Args:
            drop_remainder (bool, optional): Whether to drop or to passthrough unspecified columns during normalization.
            *args: Additional positional arguments for the parent class DataLoader.
            **kwargs: Additional keyword arguments for the partent class DataLoader.
        """
        super().__init__(*args, **kwargs)
        self.drop_remainder = drop_remainder

    
    def make_dataset(self, start_index, end_index):
        print(f"creating a dataset with seq_len {self.hist_len}, stride {self.stride}, sampling rate {self.sampling_rate} and indicies {(start_index, end_index)}")
        #end_index = end_index + self.hist_len
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
        dataset = dataset.batch(self.batch_size)#, drop_remainder=True)
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
    
    def get_prediction_set(self):
        """
        work in progress:
        return the test split with self.stride = self.pred_len, 
        such that we can make predicitons on whole days without overlap.
        Note:
            messy, really only works when pred_len <= hist_len
        """
        self.stride = self.pred_len
        pred_set = self.make_dataset(self.test_range[0], self.test_range[1])
        print("using test range: ", self.test_range)
        self.stride = 1
        
        buffer_offset = self.hist_len - self.pred_len
        index = self.data_df.index[self.test_range[0] + buffer_offset : self.test_range[1]]

        return pred_set, index

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