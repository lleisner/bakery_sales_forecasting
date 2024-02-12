import tensorflow as tf
from tensorflow.keras import layers
from models.training import CustomModel

class FineTuner(CustomModel):
    def __init__(self, configs, base_model):
        super().__init__(configs=configs)
        
        self.base_model = base_model
        
        self.concatenated = layers.Concatenate(axis=-1)
        
        self.dense_layer = layers.Dense(configs.d_ff, activation="relu")
        self.time_dist_dense = layers.TimeDistributed(self.dense_layer)
        self.out = layers.Dense(configs.num_targets, activation="linear")
        
        
    @tf.function
    def call(self, x):
        x_enc, x_mark_enc = x
        x_new = self.base_model(x)
        
        x = self.concatenated([x_new, x_mark_enc[:, -self.configs.pred_len:, :]])
        
        #x = tf.concat([x_new, x_mark_enc[:, -self.configs.pred_len:, :]], axis=-1)
        x = self.time_dist_dense(x)
        x = self.out(x)
        return x[:, -self.configs.pred_len:, :]
        
        
        