import tensorflow as tf
import pandas as pd
from typing import Tuple
from abc import ABC, abstractmethod
from data_provider.data_encoder import DataProcessor
from data_provider.data_provider import DataProvider
from utils.configs import ProviderConfigs, PipelineConfigs, Settings

class BaseGenerator(ABC):
    @abstractmethod
    def preprocess(self, dataset):
        pass
    
    def __call__(self, x):
        return self.preprocess(x)
    

class WindowGenerator(BaseGenerator):
    def __init__(self, window_size: int, sliding_step: int):
        self.window_size = window_size
        self.sliding_step = sliding_step

    def preprocess(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        windows = dataset.window(self.window_size, shift=self.sliding_step)
        def sub_to_batch(sub):
            return sub.batch(self.window_size, drop_remainder=True)
        windows = windows.flat_map(sub_to_batch)
        return windows
    

class BatchGenerator(BaseGenerator):
    def __init__(self, num_targets: int, num_features: int, num_epochs: int, batch_size: int):
        self.num_targets = num_targets
        self.num_features = num_features
        self.num_epochs = num_epochs
        self.batch_size = batch_size

    def _get_variate_covariate_tuple(self, data_slice):
        batch_x = data_slice[:, :self.num_targets]
        batch_x_mark = data_slice[:, self.num_targets:-self.num_targets]
        batch_y = data_slice[:, -self.num_targets:]
        return batch_x, batch_y, batch_x_mark

    def preprocess(self, dataset: tf.data.Dataset):
        dataset = dataset.shuffle(buffer_size=1000, seed=42)
        dataset = dataset.map(self._get_variate_covariate_tuple, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        dataset = dataset.repeat(self.num_epochs)
        return dataset


class DatasetSplitter(BaseGenerator):
    def __init__(self, validation_size: float=0.1, test_size: float=0.1, buffer_size: int=1000, seed: int=42):
        self.validation_size = validation_size
        self.test_size = test_size
        self.buffer_size = buffer_size
        self.seed = seed
    
    def preprocess(self, dataset: tf.data.Dataset) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        dataset_list = list(dataset.as_numpy_iterator())
        num_samples = len(dataset_list)
        valid_size = int(self.validation_size * num_samples)
        test_size = int(self.test_size * num_samples)

        #shuffled_dataset = dataset.shuffle(buffer_size=self.buffer_size, seed=self.seed)
        shuffled_dataset = dataset
        validation_dataset = shuffled_dataset.take(valid_size)
        remaining_dataset = shuffled_dataset.skip(valid_size)
        test_dataset = remaining_dataset.take(test_size)
        train_dataset = remaining_dataset.skip(test_size)

        return train_dataset, validation_dataset, test_dataset
    

class DataPipeline:
    def __init__(self, configs):
        self.window_generator = WindowGenerator(configs.window_size, configs.sliding_step)
        self.data_splitter = DatasetSplitter(configs.validation_size, configs.test_size, configs.buffer_size, configs.seed)
        self.batch_generator = BatchGenerator(configs.num_targets, configs.num_features, configs.num_epochs, configs.batch_size)
    
    def generate_data(self, data):
        data = self.window_generator(data)
        train, val, test = self.data_splitter(data)
        train, val, test =  self.batch_generator(train), self.batch_generator(val), self.batch_generator(test)
        return train, val, test     



if __name__ == "__main__":
    provider_configs = ProviderConfigs()
    settings = Settings()

    provider = DataProvider(provider_configs)
    df = provider.load_database()
    processor = DataProcessor(data=df, future_days = 4)
    try:
        encoding = processor.encode()
    except:
        encoding = processor.fit_and_encode()
    print(encoding)

    num_features, num_targets = processor.get_shape()
    pipeline_configs = PipelineConfigs(settings, num_features, num_targets)

    dataset = tf.data.Dataset.from_tensor_slices(encoding.values)
    pipeline = DataPipeline(pipeline_configs)
    train, val, test = pipeline.generate_data(dataset)
    print(train, val, test)