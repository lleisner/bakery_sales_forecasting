import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import os
from tensorboard.plugins.hparams import api as hp

from utils.loss import custom_time_series_loss, CustomLoss
from utils.plot_hist import plot_training_history
from utils.visual_season import visualize_seasonality
from utils.configs import Settings, ProviderConfigs, ProcessorConfigs, PipelineConfigs, TransformerConfigs
from models.lstm_model.lstm import CustomLSTM

from models.iTransformer.i_transformer import Model
from models.training import CustomModel
from models.iTransformer.configs import Configurator

from data_provider.data_provider import DataProvider
from data_provider.data_encoder import DataProcessor
from data_provider.data_pipeline import DataPipeline
from data_provider.time_configs import TimeConfigs
from tensorflow.keras.callbacks import Callback, EarlyStopping, TensorBoard, ModelCheckpoint

import warnings


if __name__ == "__main__":
    
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.utils.validation")

    settings = Settings()
    provider_configs = ProviderConfigs()
    processor_configs = ProcessorConfigs(settings)

    provider = DataProvider(provider_configs)
    
    provider.create_new_database(provider_list=['sales'])
        
    df = provider.load_database()
        
    processor = DataProcessor(data=df, configs=processor_configs)
    
    encoding = processor.fit_and_encode()
        
    try:
        encoding = processor.encode()
    except:
        encoding = processor.fit_and_encode()
        
    #encoding = processor.get_uncoded_data()
    
    print("dataset features: ", encoding.columns)
    num_features, num_targets = processor.get_shape()
    pipeline_configs = PipelineConfigs(settings, num_features, num_targets)


    to_predict = encoding.tail(settings.seq_length)
    # Set train cutoff to two months ago
    dataset = encoding[:-512]

    steps_per_epoch, validation_steps, test_steps = settings.calculate_steps(dataset.shape[0])

    dataset = tf.data.Dataset.from_tensor_slices(dataset.values)
    pipeline = DataPipeline(pipeline_configs)
    train, val, test = pipeline.generate_data(dataset)

    #loss = CustomLoss(settings.length_of_day)
    loss = tf.keras.losses.MeanSquaredError()

   # configs = Configurator()
    t_configs = TransformerConfigs(settings, num_features, num_targets)
    model = Model(t_configs)
    #model = CustomLSTM(t_configs)
    #model = CustomModel(t_configs)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=settings.learning_rate), loss=loss, metrics=[tf.keras.metrics.MeanSquaredError()], weighted_metrics=[])


    log_dir = "logs/fit/"
    checkpoint_path = 'saved_models/i_transformer_weights/checkpoint.ckpt'

    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',  # Monitor loss
        patience=settings.early_stopping_patience,         # Number of epochs with no improvement to wait
        restore_best_weights=True  # Restore the best weights when stopped
    )
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    #model.load_weights(checkpoint_path)

    hist = model.fit(train, epochs=settings.num_epochs, steps_per_epoch=steps_per_epoch, validation_data=val, validation_steps=validation_steps, callbacks=[early_stopping, tensorboard_callback, checkpoint], use_multiprocessing=True)

    plot_training_history(hist)
    
    model.summary()

    model.evaluate(test, steps=test_steps)


    index = to_predict.index[-settings.future_steps:]
    X = to_predict.iloc[:, :num_features]

    x_s , x_mark = X.iloc[:, :num_targets].values, X.iloc[:, num_targets:].values
    x_s, x_mark = np.expand_dims(x_s, axis=0), np.expand_dims(x_mark, axis=0)
    x_s, x_mark = tf.convert_to_tensor(x_s), tf.convert_to_tensor(x_mark)

    prediction = model.predict((x_s, x_mark), batch_size=1)
    prediction = pd.DataFrame(np.squeeze(prediction), index=index)

    prediction = processor.decode_data(prediction)
    print(prediction)
    prediction.to_csv('data/prediction.csv')

    
