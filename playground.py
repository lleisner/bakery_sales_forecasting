import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import os
from tensorboard.plugins.hparams import api as hp

from utils.loss import custom_time_series_loss, CustomLoss
from utils.plot_hist import plot_training_history
from utils.visual_season import visualize_seasonality
from utils.configs import Settings, ProviderConfigs, PipelineConfigs, TransformerConfigs
from models.lstm_model.lstm import CustomLSTM

from models.iTransformer.i_transformer import Model
from models.iTransformer.configs import Configurator

from data_provider.data_provider import DataProvider
from data_provider.data_encoder import DataProcessor
from data_provider.data_pipeline import DataPipeline
from data_provider.time_configs import TimeConfigs
from tensorflow.keras.callbacks import Callback, EarlyStopping, TensorBoard, ModelCheckpoint

if __name__ == "__main__":
    
    settings = Settings()
    provider_configs = ProviderConfigs()

    provider = DataProvider(provider_configs)
    df = provider.load_database()
    print(df['10'])
    visualize_seasonality(df, '10')