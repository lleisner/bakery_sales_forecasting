import tensorflow as tf
import pandas as pd
from typing import Tuple
from abc import ABC, abstractmethod
from data_provider.data_encoder import DataEncoder
from data_provider.data_provider import DataProvider
from utils.configs import ProviderConfigs, PipelineConfigs, Settings, ProcessorConfigs

class BaseGenerator:
    """Abstract base class for data generators."""
    def __init__(self, configs):
        self.configs = configs
    
    def preprocess(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Abstract method for preprocessing datasets. To be implemented by subclasses."""
        pass
    
    def __call__(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Allows the instance to be called like a function to preprocess the dataset."""
        return self.preprocess(dataset)
    

class WindowGenerator(BaseGenerator):
    """Class for generating windows of data from a dataset for use in time-series modeling."""
    def preprocess(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        windows = dataset.window(self.configs.window_size, shift=self.configs.sliding_step)
        def sub_to_batch(sub):
            return sub.batch(self.configs.window_size, drop_remainder=True)
        windows = windows.flat_map(sub_to_batch)
        return windows


class BatchGenerator(BaseGenerator):
    """Class for preparing batches of data by applying a transformation and batching operation to the dataset"""
    
    def set_feature_target_nums(self, num_features, num_targets):
        self.num_features = num_features
        self.num_targets = num_targets

    def _get_variate_covariate_tuple(self, data_slice):
        batch_x = data_slice[:, :self.num_targets]
        batch_x_mark = data_slice[:, self.num_targets:-self.num_targets]
        batch_y = data_slice[:, -self.num_targets:]
        return batch_x, batch_y, batch_x_mark
        # for TiDE it should be: batch_x, batch_y, batch_xcov, batch_y_cov, batch_x_catcov, batch_y_catcov

    def preprocess(self, dataset: tf.data.Dataset):
        dataset = dataset.map(self._get_variate_covariate_tuple, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.configs.batch_size, drop_remainder=True).prefetch(buffer_size=tf.data.AUTOTUNE)
        dataset = dataset.repeat(self.configs.num_repeats)
        return dataset



class DatasetSplitter(BaseGenerator):
    def preprocess(self, dataset: tf.data.Dataset) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        dataset_list = list(dataset.as_numpy_iterator())
        num_samples = len(dataset_list)
        valid_size = int(self.configs.validation_size * num_samples)
        test_size = int(self.configs.test_size * num_samples)
        train_size = num_samples - valid_size - test_size
        
        train_dataset = dataset.take(train_size)
        remaining_dataset = dataset.skip(train_size)
        valid_dataset = remaining_dataset.take(valid_size)
        test_dataset = remaining_dataset.skip(valid_size)

        return train_dataset, valid_dataset, test_dataset
    


class DataPipeline:
    def __init__(self, configs):
        self.window_generator = WindowGenerator(configs)
        self.data_splitter = DatasetSplitter(configs)
        self.batch_generator = BatchGenerator(configs)
    
    def generate_train_test_splits(self, data):
        num_features, num_targets = self.get_feature_target_nums(data)
        self.batch_generator.set_feature_target_nums(num_features, num_targets)
        
        data = tf.data.Dataset.from_tensor_slices(data.values)
        data = self.window_generator(data)
        train, val, test = self.data_splitter(data)
        train, val, test =  self.batch_generator(train), self.batch_generator(val), self.batch_generator(test)
        return train, val, test     
    
    def get_feature_target_nums(self, df):
        return (pd.to_numeric(df.columns, errors='coerce').isnull().sum(), pd.to_numeric(df.columns, errors='coerce').notnull().sum())

    

        

if __name__ == "__main__":
    settings = Settings()
    provider_configs = ProviderConfigs()
    processor_configs = ProcessorConfigs(settings=settings)

    provider = DataProvider(provider_configs)
    df = provider.load_database()
    processor = DataEncoder(configs=processor_configs)
    encoding = processor.process_data(df, encode=True)

    num_features, num_targets = processor.get_shape()
    pipeline_configs = PipelineConfigs(settings, num_features, num_targets)

    pipeline = DataPipeline(pipeline_configs)
    train, val, test = pipeline.generate_train_test_splits(encoding)
    #trains, vals, tests = pipeline.generate_cross_validation_data(dataset)
    print(train, val, test)
   # print(trains, vals, tests)