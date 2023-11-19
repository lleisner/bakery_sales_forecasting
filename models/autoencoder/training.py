import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

from models.autoencoder.autoencoder import build_autoencoder



def calculate_dissimilarities(df1: pd.DataFrame, df2: pd.DataFrame) -> tuple:
    """
    Calculate the dissimilarity between two DataFrames.

    Args:
        df1 (pd.DataFrame): The first DataFrame.
        df2 (pd.DataFrame): The second DataFrame.

    Returns:
        tuple: A tuple containing the dissimilarity percentage and the count of dissimilar elements.
    """
    dissimilar_count = np.sum(np.sum(df1 != df2, axis=0))
    total_elements = df1.shape[0] * df1.shape[1]
    dissimilarity_percentage = (dissimilar_count / total_elements) * 100
    return dissimilarity_percentage, dissimilar_count

def test_encoding(data: pd.DataFrame, autoencoder: tf.keras.models.Model):
    """
    Test the encoding and decoding process using an AutoEncoder model.

    Args:
        data (pd.DataFrame): The input data to be encoded and decoded.
        autoencoder (str): The file path to the trained AutoEncoder model.

    Returns:
        tuple: A tuple containing the dissimilarity percentage and the count of dissimilar elements.
    """
    encoder = autoencoder.get_layer('encoder')
    decoder = autoencoder.get_layer('decoder')
    encoding = encoder.predict(data)
    decoding = decoder.predict(encoding)
    decoding = (decoding >= 0.5).astype(int)

    return calculate_dissimilarities(data, decoding)

def train_model(data: pd.DataFrame, autoencoder: tf.keras.models.Model, save_to_file: str='models/my_autoencoder.h5'):
    """
    Trains an AutoEncoder model on the input data.

    Args:
        data (pd.DataFrame): The input data for training the AutoEncoder.

    Returns:
        None
    """
    X_train, X_test, y_train, y_test = train_test_split(data, data, test_size=0.15, random_state=42)

    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=tf.keras.losses.BinaryCrossentropy(), metrics=[tf.keras.metrics.BinaryAccuracy()])
    
    # Define early stopping callback
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',  # Monitor validation loss
        patience=10,         # Number of epochs with no improvement to wait
        restore_best_weights=True  # Restore the best weights when stopped
    )
    
    hist = autoencoder.fit(X_train, y_train, batch_size=64, epochs=500, validation_split=0.2, callbacks=[early_stopping])
    eval = autoencoder.evaluate(X_test, y_test, batch_size=64)
    print(f'evaluation metrics: {eval}')
    percentage, count = test_encoding(data, autoencoder)
    print(f'autoencoder training finished with {percentage}% accuracy on the data and predicted {count} entries wrong')

    autoencoder.save(save_to_file)
