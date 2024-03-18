import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler, SplineTransformer, OneHotEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from models.iTransformer.cyclic_encoder import CyclicEncoder

    
class WindowGenerator():
    def __init__(self, hist_len, pred_len, shift, 
                 train_df, val_df, test_df, 
                 label_columns=None):
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indicies = {name: i for i, name in enumerate(label_columns)}
        self.column_indicies = {name: i for i, name in enumerate(train_df.columns)}
        
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.shift = shift
        
        self.total_window_size = hist_len + shift
        
        self.input_slice = slice(0, self.hist_len)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        
        self.label_start = self.total_window_size - self.pred_len
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
        
    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])
        

    def make_dataset(self, X, y):
        ts_dfa = tf.keras.utils.timeseries_dataset_from_array
        input_dataset = ts_dfa(
            data=X,
            targets=None,
            sequence_length=self.hist_len,
            sequence_stride=self.stride,
            sampling_rate=1,
            batch_size=self.batch_size,
            shuffle=False,
        )
        target_dataset = ts_dfa(
            data=y,
            targets=None,
            sequence_length=self.pred_len,
            sequence_stride=self.stride,
            sampling_rate=1,
            batch_size=self.batch_size,
            shuffle=False,
        )    
        return input_dataset, target_dataset
    
    @property
    def train(self):
        return self.make_dataset(self.train_df)
    
    @property
    def val(self):
        return self.make_dataset(self.val_df)
    
    @property
    def test(self):
        return self.make_dataset(self.test_df)
    
    @property
    def example(self):
        result = getattr(self, '_example', None)
        if result is None:
            result = next(iter(self.train))
            self._example = result
        return result
        
        
    
    
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
        batch_size,
        epoch_len,
        val_len,
        freq='H',
        normalize=True,     
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
            
        Returns:
            None
        """
        self.data_df = pd.read_csv(open(data_path, 'r'))
        self.data_df.fillna(0, inplace=True)
        self.data_df.set_index(pd.DatetimeIndex(self.data_df[datetime_col]), inplace=True)
        self.data_df.drop(columns=[datetime_col], inplace=True)

        self.num_cov_cols = numerical_cov_cols
        self.cat_cov_cols = categorical_cov_cols
        if timeseries_cols:
            self.ts_cols = timeseries_cols
        else:
            self.ts_cols = self.use_num_columns_as_ts_list()

        self.train_range = train_range
        self.val_range = val_range
        self.test_range = test_range
        
        data_df_idx = self.data_df.index
        """
        date_index = data_df_idx.union(
            pd.date_range(
                data_df_idx[-1] + pd.Timedelta(1, freq=freq),
                periods=pred_len + 1,
                freq=freq,
            )
        )
        """
        time_df = self.create_temporal_features(self.data_df.index)
        time_cols = time_df.columns
        cyclic_cov_cols.extend(time_cols)
        self.cyc_cov_cols = cyclic_cov_cols

        self.data_df = pd.concat([time_df, self.data_df], axis=1)
        
        if normalize:
            self._normalize()
        
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.stride = stride
        self.batch_size = batch_size
        self.freq = freq
        self.normalize = normalize
        self.epoch_len = epoch_len
        self.val_len = val_len
        
    def create_temporal_features(self, data: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Generate a DataFrame with temporal features extracted from a DatetimeIndex.

        Parameters:
        - data (pd.DatetimeIndex): The input datetime index from which to extract the time-related features.

        Returns:
        - pd.DataFrame: DataFrame with columns for hour, day of week, day of month, day of year, month, and quarter.
        """
        features = {
            'hour': data.hour,
            'dayofweek': data.dayofweek,
            'dayofmonth': data.day,
            'dayofyear': data.dayofyear,
            'month': data.month,
            'quarter': data.quarter,
        }
        return pd.DataFrame(features, index=data)

        
    def feature_target_split(self):
        y = self.data_df[self.ts_cols]
        lagged_features = pd.concat(
            [
                self.data_df,
                y.shift(self.pred_len).rename(f"lagged_feature_pred_len{self.pred_len}h"),
                y.shift(self.pred_len).rolling(self.pred_len).mean().rename(f"lagged_mean_{self.pred_len}h"),
                y.shift(self.pred_len).rolling(self.pred_len).max().rename(f"lagged_max_{self.pred_len}h"),
                y.shift(self.pred_len).rolling(self.pred_len).min().rename(f"lagged_min_{self.pred_len}h"),
            ],
            axis="columns",
        )
        lagged_features.dropna(inplace=True)
        X = self.data_df.drop(self.ts_cols, axis="columns")
        return X, y
    
    def get_time_series_split(self, X, y):
        ts_cv = TimeSeriesSplit(
            n_splits=5,
            gap=self.pred_len,
            max_train_size=None,
            test_size=None,
        )
        all_splits = list(ts_cv.split(X, y))
        return all_splits
    
    def _normalize(self):
        
        transformers_list = []
        transformers_list.append(("target", MinMaxScaler(), self.ts_cols))

        if self.num_cov_cols is not None:
            
            transformers_list.append(("numerical", MinMaxScaler(), self.num_cov_cols))

        if self.cat_cov_cols is not None:
            transformers_list.append(("categorical", OneHotEncoder(), self.cat_cov_cols))

        if self.cyc_cov_cols is not None:
            transformers_list.append(("cyclic", CyclicEncoder(), self.cyc_cov_cols))
        

        transformer = ColumnTransformer(
            transformers=transformers_list,
            remainder="passthrough",
        )
   
        transformed_df = transformer.fit_transform(self.data_df)
        column_names = transformer.get_feature_names_out()
        
        
        print(column_names)
  
        self.data_df = pd.DataFrame(transformed_df, columns=column_names, index=self.data_df.index)
        self.ts_cols = [target for target in column_names if "target" in target]
        self.num_cov_cols = [covariate for covariate in column_names if "numerical" in covariate]
        self.cat_cov_cols = [covariate for covariate in column_names if "categorical" in covariate]
        self.cyc_cov_cols = [covariate for covariate in column_names if "cyclic" in covariate]
         
        print(self.ts_cols)

    
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
    
    def get_data(self):
        return self.data_df
    
    def use_num_columns_as_ts_list(self):
        return [col for col in self.data_df.columns if str(col).isnumeric()]
