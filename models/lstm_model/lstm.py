import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class CustomLSTM(keras.Model):
    def __init__(self, seq_length, pred_len, num_features, num_targets):
        super().__init__()
        self.pred_len = pred_len
        
        self.lstm1 = LSTM(32, return_sequences=True, input_shape=(seq_length, num_features))
        self.lstm2 = LSTM(64, return_sequences=True)
        self.lstm3 = LSTM(32, return_sequences=True)
        
        self.dropout = Dropout(0.2)
        self.out = Dense(num_targets)
        
        self.build((None,seq_length, num_features))

    @tf.function
    def call(self, x):
        x = self.lstm1(x)
        x = self.dropout(x)
        x = self.lstm2(x)
        x = self.dropout(x)
        x = self.lstm3(x)
        x = self.out(x)
        return x
    
    def build(self, input_shape):
        self.lstm1.build(input_shape)
        self.lstm2.build(self.lstm1.compute_output_shape(input_shape))
        self.lstm3.build(self.lstm2.compute_output_shape(input_shape))
        super().build(input_shape)
    

    @tf.function
    def train_step(self, data):
            inputs, targets = data
            inputs = tf.cast(inputs, dtype=tf.float32)
            targets = tf.cast(targets, dtype=tf.float32)

            with tf.GradientTape() as tape:
                outputs = self(inputs)
                outputs = outputs[:, -self.pred_len:, :]
                targets = targets[:, -self.pred_len:, :]
                loss = self.compute_loss(y=targets, y_pred=outputs)
                
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            
            for metric in self.metrics:
                if metric.name == "loss":
                    metric.update_state(loss)
                else:
                    metric.update_state(targets, outputs)
            
            return {m.name: m.result() for m in self.metrics}
    """   
    @tf.function
    def test_step(self, data):
        # Unpack the data
        x, y = data
        # Compute predictions
        y_pred = self(x)
        # Updates the metrics tracking the loss
        self.compute_loss(y=y, y_pred=y_pred)
        # Update the metrics.
        for metric in self.metrics:
            if metric.name != "loss":
                metric.update_state(y, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}
    """
    

        
    @tf.function
    def train_step(self, data):
        batch_x, batch_y, batch_x_mark = data   
        inputs = tf.concat([batch_x, batch_x_mark], axis=-1)
        
        inputs = tf.cast(inputs, dtype=tf.float32)
        batch_y = tf.cast(batch_y, dtype=tf.float32)

        with tf.GradientTape() as tape:
            outputs = self(inputs)
            
            outputs = outputs[:, -self.pred_len:, :]
            batch_y = batch_y[:, -self.pred_len:, :]

            loss = self.compute_loss(y=batch_y, y_pred=outputs)
            
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(batch_y, outputs)
        
        return {m.name: m.result() for m in self.metrics}
    
    
    @tf.function
    def test_step(self, data):
        batch_x, batch_y, batch_x_mark = data
        inputs = tf.concat([batch_x, batch_x_mark], axis=-1)
        
        outputs = self(inputs)
        outputs = outputs[:, -self.pred_len:, :]
        batch_y = batch_y[:, -self.pred_len:, :]
        
        self.compute_loss(y=batch_y, y_pred=outputs)
        
        for metric in self.metrics:
            if metric.name != "loss":
                metric.update_state(batch_y, outputs)

        return {m.name: m.result() for m in self.metrics}
    