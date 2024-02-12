import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from models.training import CustomModel

class HybridModel(CustomModel):
    def __init__(self, configs):
        super().__init__(configs=configs)
        
        self.conv1 = layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')
        self.conv2 = layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')
        
        self.concatenated = layers.Concatenate(axis=-1)
        
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(64, activation='relu')
        
        self.time_dist1 = layers.TimeDistributed(self.dense1)
        self.time_dist2 = layers.TimeDistributed(self.dense2)
        
        self.out = layers.Dense(configs.num_targets, activation='linear', name='output')
        
    @tf.function
    def call(self, x, training):
        batch_x, batch_x_mark = x
        
        batch_x = self.conv1(batch_x)
        batch_x = self.conv2(batch_x)
                
        x = self.concatenated([batch_x, batch_x_mark])
        
        x = self.time_dist1(x)
        x = self.time_dist2(x)
        x = self.out(x)
        
        return x[:, -self.configs.pred_len:, :]