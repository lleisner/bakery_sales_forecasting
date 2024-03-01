import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from models.training import CustomModel

class CustomLSTM(CustomModel):
    def __init__(self, configs):
        super().__init__(configs=configs)
        """
        self.lstm1 = LSTM(128, return_sequences=True, input_shape=(configs.seq_len, configs.num_features))
        self.lstm2 = LSTM(128, return_sequences=True)
        self.lstm3 = LSTM(64, return_sequences=True)
        self.l_relu = LeakyReLU(alpha=0.5)
        
        self.dropout = Dropout(configs.dropout)
        self.out = Dense(configs.num_targets)
        """
        # Encoder
        self.encoder_l1 = layers.LSTM(128, return_state=True)#, input_shape=(configs.seq_len, configs.num_features))
        # Decoder
        self.repeat_vector = layers.RepeatVector(configs.pred_len)
        self.decoder_l1 = layers.LSTM(128, return_sequences=True)
        self.time_distributed = layers.TimeDistributed(layers.Dense(configs.num_features))
        
    @tf.function
    def call(self, x, training):
        batch_x, batch_x_mark = x
        x = tf.concat([batch_x, batch_x_mark], axis=-1)
        
        print(f"input_shape: {x}, should be: (batch_size, seq_len, features)")
        
        
        # Encoder
        encoder_output, state_h, state_c = self.encoder_l1(x)
        encoder_states = [state_h, state_c]
        print(f"encoder_output: {encoder_output.shape}, should be: (None, 100)")

        # Encoder
        decoder_input = self.repeat_vector(encoder_output)
        decoder_output = self.decoder_l1(decoder_input, initial_state=encoder_states)
        print(f"decoder_output: {decoder_output.shape}, should be: (batch_size, future_steps, 100)")

        decoder_output = self.time_distributed(decoder_output)
        print(f"time_distributed: {decoder_output.shape}, should be: (batch_size, future_steps, features)")

        return decoder_output[:, :, :self.configs.num_targets]
        
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
    """

