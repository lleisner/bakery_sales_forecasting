import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


def create_lstm_model(seq_length, num_features, num_targets):
    """
    Creates a customized LSTM-based model for multivariate time series forecasting.

    Args:
    - seq_length: Sequence length for input sequences.
    - num_features: Number of input features.
    - num_targets: Number of target variables.

    Returns:
    - model: Customized LSTM-based Keras model.
    """

    model = Sequential()
    model.add(LSTM(32, return_sequences=True, input_shape=(seq_length, num_features)))
    model.add(Dropout(0.2))  # Add dropout for regularization
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))  # Add dropout for regularization
    model.add(LSTM(32, return_sequences=True))
    model.add(Dense(num_targets))

    return model

class CustomLSTM(keras.Model):
    def __init__(self, seq_length, num_features, num_targets):
        super().__init__()
        self.lstm1 = LSTM(32, return_sequences=True, input_shape=(seq_length, num_features))
        self.lstm2 = LSTM(64, return_sequences=True)
        self.lstm3 = LSTM(32, return_sequences=True)
        
        self.dropout = Dropout(0.2)
        self.out = Dense(num_targets)

    @tf.function
    def call(self, x):
        x = self.lstm1(x)
        x = self.dropout(x)
        x = self.lstm2(x)
        x = self.dropout(x)
        x = self.lstm3(x)
        x = self.out(x)
        return x
        