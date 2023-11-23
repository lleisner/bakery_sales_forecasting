import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

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
        