import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging

class TiDEData(object):
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
        permute=False,
    ):
        self.data_df = pd.read_csv(open(data_path, 'r'))
        self.data_df.fillna(0, inplace=True)
        self.data_df[datetime_col] = pd.to_datetime(self.data_df[datetime_col])
        self.data_df.set_index(datetime_col, inplace=True)
        
        self.num_cols = numerical_cov_cols if numerical_cov_cols else []
        self.cat_cols = categorical_cov_cols if categorical_cov_cols else []
        self.cyc_cols = cyclic_cov_cols if cyclic_cov_cols else []
        
        self.ts_cols = timeseries_cols if timeseries_cols else self.data_df.columns
        
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.batch_size = batch_size
        self.freq = freq
        self.normalize = normalize
        self.premute = permute
        
        
        