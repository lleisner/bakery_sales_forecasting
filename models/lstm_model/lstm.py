import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, InputLayer, LeakyReLU

from models.training import CustomModel

class CustomLSTM(CustomModel):
    def __init__(self, configs):
        super().__init__(configs=configs)
        
        self.lstm1 = LSTM(64, return_sequences=True, input_shape=(configs.seq_len, configs.num_features))
        self.lstm2 = LSTM(64, return_sequences=True)
        self.lstm3 = LSTM(32, return_sequences=True)
        self.l_relu = LeakyReLU(alpha=0.5)
        
        self.dropout = Dropout(configs.dropout)
        self.out = Dense(configs.num_targets)
        
    """
        self.custom_layers = [
            LSTM(128, return_sequences=True, input_shape=(configs.seq_len, configs.num_features)),
            LeakyReLU(alpha=0.5),
            LSTM(128, return_sequences=True),
            LeakyReLU(alpha=0.5),
            Dropout(0.3),
            LSTM(64, return_sequences=False),
            Dropout(0.3),
            Dense(configs.num_targets)
        ]
    
    def call(self, x):
        batch_x, batch_x_mark = x
        x = tf.concat([batch_x, batch_x_mark], axis=-1)
        for layer in self.custom_layers:
            try:
                x = layer(x)
            except:
                x = layer(x)
        return x
    
    """   
    @tf.function
    def call(self, x, training):
        batch_x, batch_x_mark = x
        x = tf.concat([batch_x, batch_x_mark], axis=-1)
        x = self.lstm1(x)
        x = self.l_relu(x)
        x = self.lstm2(x)
        x = self.l_relu(x)
        x = self.dropout(x, training)
        x = self.lstm3(x)
        x = self.dropout(x, training)
        x = self.out(x)
        return x[:, -self.configs.pred_len:, :]
    

