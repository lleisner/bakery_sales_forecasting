import tensorflow as tf
import pandas as pd
from typing import Tuple
from abc import ABC, abstractmethod
from data_provider.data_encoder import DataEncoder
from data_provider.data_provider import DataProvider
from utils.configs import ProviderConfigs, PipelineConfigs, Settings, ProcessorConfigs


class BatchGenerator():
    """Class for preparing batches of data by applying a transformation and batching operation to the dataset"""
    def __init__(self, configs):
        self.configs = configs
        
    def set_feature_target_nums(self, num_features, num_targets):
        self.num_features = num_features
        self.num_targets = num_targets
        
    def _get_variate_covariate_tuple(self, data_slice):
        batch_x = data_slice[:, :self.num_targets]
        batch_x_mark = data_slice[:, self.num_targets:-self.num_targets]
        batch_y = data_slice[:, -self.num_targets:]
        return batch_x, batch_y, batch_x_mark

    def __call__(self, dataset: tf.data.Dataset):
        dataset = dataset.map(self._get_variate_covariate_tuple, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.configs.batch_size, drop_remainder=True).prefetch(buffer_size=tf.data.AUTOTUNE)
        dataset = dataset.repeat(self.configs.num_repeats)
        return dataset

class WindowGenerator():
    """Class for generating windows of data from a dataset for use in time-series modeling."""
    def __init__(self, configs):
        self.configs = configs
        
    def __call__(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        windows = dataset.window(self.configs.window_size, shift=self.configs.sliding_step)
        def sub_to_batch(sub):
            return sub.batch(self.configs.window_size, drop_remainder=True)
        windows = windows.flat_map(sub_to_batch)
        return windows

class NewDataPipeline:
    def __init__(self, configs):
        self.batch_generator = BatchGenerator(configs)
        self.window_generator = WindowGenerator(configs)
        self.configs=configs

    def generate_train_val_splits(self, dataset):
        num_features, num_targets = self.get_feature_target_nums(dataset)
        self.batch_generator.set_feature_target_nums(num_features, num_targets)
        dataset = tf.data.Dataset.from_tensor_slices(dataset.values)
        
        # List to store the datasets for each walk-forward step
        train_val_splits = []

        # Total number of samples in the dataset
        total_samples = tf.data.experimental.cardinality(dataset).numpy()
        print(f"total_samples: {total_samples}")
        # Calculate the starting point for the test set to ensure it's not included in train/val splits
        test_start_index = total_samples - self.configs.validation_window_size
        train_end = 0

        while True:
            # Define the end of the training set and the end of the validation set within this step
            train_end += self.configs.validation_window_size
            val_end = train_end + self.configs.validation_window_size

            # Ensure there's enough data for this step without encroaching on the test set
            if val_end >= total_samples:
                print(f"val end: {val_end}, test start: {test_start_index}")
                break  # Stop if this step would overlap with the test set

            # Define train and validation datasets for the current step
            train_ds = dataset.take(train_end)
            val_ds = dataset.skip(train_end).take(val_end)

            # Window and batch the datasets
            train_ds = self.window_and_batch(train_ds)
            val_ds = self.window_and_batch(val_ds)
            
            train_size = (train_end - self.configs.window_size) // self.configs.sliding_step
            val_size = (self.configs.validation_window_size - self.configs.window_size) // self.configs.sliding_step
            
            print(f"train_size: {train_size}, val_size: {val_size}")

            train_val_splits.append((train_size, train_ds, val_size, val_ds))
            print(train_size)
        return train_val_splits

    def get_test_set(self, dataset):
        total_samples = tf.data.experimental.cardinality(dataset).numpy()

        # Define the test set as the last segment of the dataset
        test_ds = dataset.skip(total_samples - self.configs.validation_window_size)
        test_ds = self.window_and_batch(test_ds)
        return test_ds


    def window_and_batch(self, dataset):
        # Apply windowing
        dataset = self.window_generator(dataset)
        # Apply batching and feature-target separation using BatchGenerator
        dataset = self.batch_generator(dataset)
        return dataset

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


    pipeline_configs = PipelineConfigs(settings)

    pipeline = NewDataPipeline(pipeline_configs)
    splits = pipeline.generate_train_val_splits(encoding)
    print(splits)
    #test = pipeline.get_test_set(dataset)
    #print(test)