
import tensorflow as tf
from tensorflow import keras
from tensorboard.plugins.hparams import api as hp

from utils.loss import custom_time_series_loss
from models.lstm_model.training import plot_training_history
from models.lstm_model.lstm import create_lstm_model

from data_provider.data_merger import DataMerger
from data_provider.data_encoder import DataProcessor
from data_provider.data_pipeline import DataPipeline


if __name__=="__main__":

    # Get data
    provider = DataMerger()
    df = provider.merge()
    print(f'dataset: {df}')


    # Encode data
    data_processor = DataProcessor(df)
    #encoded_data = data_processor.fit_and_encode()
    encoded_data = data_processor.encode()
    print(f'encoded dataset: {encoded_data}')
    

    # Set some parameters
    past_days = 31
    future_days = 1
    length_of_day = 16
    strides = length_of_day
    future_steps = future_days * length_of_day
    seq_length = past_days * length_of_day

    num_epochs = 10
    batch_size = 32
    validation_size = 0.2
    test_size = 0.1

    num_targets =  len([col for col in df.columns if str(col).isnumeric()])
    num_features = encoded_data.shape[1] - num_targets
    steps_per_epoch = (encoded_data.shape[0] // strides) * (1-(test_size + validation_size)) // batch_size -1
    validation_steps = max((encoded_data.shape[0] // strides) * validation_size // batch_size, 1) 

    log_dir = "logs/fit/"

    
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


    # Create a model 
    model = create_lstm_model(seq_length, num_features=num_features, num_targets=num_targets)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=custom_time_series_loss(future_steps, length_of_day), metrics=[tf.keras.metrics.MeanSquaredError()])

    early_stopping = keras.callbacks.EarlyStopping(
        monitor='loss',  # Monitor loss
        patience=100,         # Number of epochs with no improvement to wait
        restore_best_weights=True  # Restore the best weights when stopped
    )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Fit the model
    hist = model.fit(train, epochs=num_epochs, steps_per_epoch=steps_per_epoch, validation_data=val, validation_steps=validation_steps, callbacks=[early_stopping, tensorboard_callback], use_multiprocessing=True)

    plot_training_history(hist)
    
    
    
    
    