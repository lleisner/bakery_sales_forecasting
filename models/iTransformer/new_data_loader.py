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
        dataset = dataset.batch(self.batch_size) #, drop_remainder=True)
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
    
    def get_dummy(self):
        """
        Generates dummy input for the model to create its variables.
        Needed to load a model with saved weights. 
        
        Returns:
            batch_x: Dummy input batch for time series data
            batch_x_mark: Dummy input batch for additional features
        """
        
        num_features = self.data_df.shape[1]
        
        dummy_data = tf.random.normal([1, self.hist_len, num_features])
        
        batch_x, batch_y, batch_x_mark = self.split_batch(dummy_data)
        
        return batch_x, batch_x_mark
    
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
    
    
