
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import os
from tensorboard.plugins.hparams import api as hp

from utils.loss import custom_time_series_loss, CustomLoss
from utils.plot_hist import plot_training_history
from models.lstm_model.lstm import CustomLSTM

from models.iTransformer.i_transformer import Model
from models.iTransformer.configs import Configurator

from data_provider.data_provider import DataProvider
from data_provider.data_encoder import DataProcessor
from data_provider.data_pipeline import DataPipeline
from data_provider.time_configs import TimeConfigs
from tensorflow.keras.callbacks import Callback, EarlyStopping, TensorBoard, ModelCheckpoint



if __name__=="__main__":

    time_configs = TimeConfigs()
    provider = DataProvider(configs=time_configs)
    df = provider.get_data()

    # Set some parameters
    past_days = 32
    future_days = 4
    length_of_day = 16
    strides = length_of_day
    future_steps = future_days * length_of_day
    seq_length = past_days * length_of_day

    # Encode data
    data_processor = DataProcessor(df, future_days)

    try:
        encoded_data = data_processor.encode()
    except: 
        encoded_data  = data_processor.fit_and_encode()
        
    print(f'encoded dataset: {encoded_data}')
    


    num_epochs = 1000

    batch_size = 32
    validation_size = 0.2
    test_size = 0.1

    num_targets =  len([col for col in df.columns if str(col).isnumeric()])
    num_features = encoded_data.shape[1] - num_targets
    steps_per_epoch = (encoded_data.shape[0] // strides) * (1-(test_size + validation_size)) // batch_size -1
    validation_steps = max((encoded_data.shape[0] // strides) * validation_size // batch_size, 1) 
    test_steps = max((encoded_data.shape[0] // strides) * test_size // batch_size, 1) 


    log_dir = "logs/fit/"
    checkpoint_path = 'saved_models/i_transformer_weights/checkpoint.ckpt'

    
    # Create train validation test splits
    data_pipeline = DataPipeline(
                                window_size=seq_length, 
                                sliding_step=length_of_day, 
                                num_targets=num_targets, 
                                num_features=num_features, 
                                num_epochs=num_epochs, 
                                batch_size=batch_size,
                                validation_size=validation_size,
                                test_size=test_size
                                )
    dataset = tf.data.Dataset.from_tensor_slices(encoded_data.values)
    train, val, test = data_pipeline.generate_data(dataset)

    loss = CustomLoss(length_of_day)



    # Create a model     
    #model = CustomLSTM(seq_length, future_steps, num_features, num_targets)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    # Apply the configurations
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)

    configs = Configurator()
    model = Model(configs)
   # model = tf.keras.models.load_model('saved_models/itransformer')

    #model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=custom_time_series_loss(future_steps, length_of_day), metrics=[tf.keras.metrics.MeanSquaredError()])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=loss, metrics=[tf.keras.metrics.MeanSquaredError()], weighted_metrics=[])
    #model.summary()
    
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',  # Monitor loss
        patience=300,         # Number of epochs with no improvement to wait
        restore_best_weights=True  # Restore the best weights when stopped
    )
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    
    #model.load_weights(checkpoint_path)


    # Fit the model
    hist = model.fit(train, epochs=num_epochs, steps_per_epoch=steps_per_epoch, validation_data=val, validation_steps=validation_steps, callbacks=[early_stopping, tensorboard_callback, checkpoint], use_multiprocessing=True)
    #model.save('saved_models/itransformer.keras')
    plot_training_history(hist)
    
    print("model evaluation:")
    model.evaluate(test, steps=test_steps)
    
    day_to_predict = encoded_data.iloc[:, :74].tail(seq_length)
    print(day_to_predict.shape)
    day_x, day_x_mark = day_to_predict.iloc[:, :52].values, day_to_predict.iloc[:, 52:].values
    
    day_x, day_x_mark = np.expand_dims(day_x, axis=0), np.expand_dims(day_x_mark, axis=0)
    
    day_x, day_x_mark = tf.convert_to_tensor(day_x), tf.convert_to_tensor(day_x_mark)
    
    
    prediction = model.predict((day_x, day_x_mark))
    df = pd.DataFrame(np.squeeze(prediction))

    decoded_pred = data_processor.decode_data(df)
    print(decoded_pred)
    
