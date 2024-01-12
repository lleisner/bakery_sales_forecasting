import tensorflow as tf
import pandas as pd
from typing import Tuple
from abc import ABC, abstractmethod
from data_provider.data_encoder import DataProcessor
from data_provider.data_provider import DataProvider
from utils.configs import ProviderConfigs, PipelineConfigs, Settings

class BaseGenerator(ABC):
    def __init__(self, configs):
        self.configs = configs
    
    @abstractmethod
    def preprocess(self, dataset):
        pass
    
    def __call__(self, x):
        return self.preprocess(x)
    

class WindowGenerator(BaseGenerator):
    def preprocess(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        windows = dataset.window(self.configs.window_size, shift=self.configs.sliding_step)
        def sub_to_batch(sub):
            return sub.batch(self.configs.window_size, drop_remainder=True)
        windows = windows.flat_map(sub_to_batch)
        return windows
    

class BatchGenerator(BaseGenerator):
    def _get_variate_covariate_tuple(self, data_slice):
        batch_x = data_slice[:, :self.configs.num_targets]
        batch_x_mark = data_slice[:, self.configs.num_targets:-self.configs.num_targets]
        batch_y = data_slice[:, -self.configs.num_targets:]
        return batch_x, batch_y, batch_x_mark

    def preprocess(self, dataset: tf.data.Dataset):
        #dataset = dataset.shuffle(buffer_size=self.configs.buffer_size, seed=42)
        dataset = dataset.map(self._get_variate_covariate_tuple, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.configs.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        dataset = dataset.repeat(self.configs.num_epochs)
        return dataset


class DatasetSplitter(BaseGenerator):
    def preprocess(self, dataset: tf.data.Dataset) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        dataset_list = list(dataset.as_numpy_iterator())
        num_samples = len(dataset_list)
        valid_size = int(self.configs.validation_size * num_samples)
        test_size = int(self.configs.test_size * num_samples)

        #shuffled_dataset = dataset.shuffle(buffer_size=self.buffer_size, seed=self.seed)
        shuffled_dataset = dataset
        validation_dataset = shuffled_dataset.take(valid_size)
        remaining_dataset = shuffled_dataset.skip(valid_size)
        test_dataset = remaining_dataset.take(test_size)
        train_dataset = remaining_dataset.skip(test_size)

        return train_dataset, validation_dataset, test_dataset
    

class DataPipeline:
    def __init__(self, configs):
        self.window_generator = WindowGenerator(configs)
        self.data_splitter = DatasetSplitter(configs)
        self.batch_generator = BatchGenerator(configs)
    
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
    processor = DataProcessor(data=df, future_steps=settings.future_steps)
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