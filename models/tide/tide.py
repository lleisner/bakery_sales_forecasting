import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from models.training import CustomModel

class ResBlock(layers.Layer):
    def __init__(self, hidden_dim, output_dim, layer_norm=False, dropout_rate=0.0):
        super(ResBlock, self).__init__()
        self.dense1 = layers.Dense(hidden_dim, activation='relu')
        self.dense2 = layers.Dense(output_dim, activation=None)
        self.res = layers.Dense(output_dim, activation=None)
        
        if layer_norm:
            self.lnorm = layers.LayerNormalization()
        self.layer_norm = layer_norm
        self.dropout = layers.Dropout(dropout_rate)
        
    def call(self, inputs):
        h_state = self.dense1(inputs)
        out = self.dense2(h_state)
        out = self.dropout(out)
        res = self.res(inputs)
        if self.layer_norm:
            return self.lnorm(out + res)
        return out + res
    
    
def make_dnn_residual(hidden_dims, layer_norm=False, dropout_rate=0.0):
    if len(hidden_dims) < 2:
        return layers.Dense(hidden_dims[-1], activation=None)
    layers = []
    for i, hdim in enumerate(hidden_dims[:-1]):
        layers.append(ResBlock(
            hdim, 
            hidden_dims[i + 1],
            layer_norm=layer_norm,
            dropout_rate=dropout_rate,
        )
    )
    return tf.keras.Sequential(layers)
        
class TIDE(CustomModel):
    def __init__(self, configs):
        super().__init__(configs=configs)
        if self.configs.transform:
            self.affine_weight = self.add_weight(
                name='affine_weight',
                shape=(self.configs.num_targets,),
                initializer='ones',
                trainable=True,
            )
        self.affine_bias = self.add_weight(
            name='affine_bias',
            shape=(self.configs.num_targets,),
            initializer='zeros',
            trainable=True,
        )
        self.encoder = make_dnn_residual(hidden_dims=configs.hidden_dims, 
                                         layer_norm=configs.layer_norm, 
                                         dropout_rate=configs.dropout_rate,
                                         )
        self.decoder = make_dnn_residual(hidden_dims=configs.hidden_dims[:-1] + [configs.dec_out_dims * configs.pred_len, ], 
                                         layer_norm=configs.layer_norm, 
                                         dropout_rate=configs.dropout_rate,
                                         )
        self.linear = layers.Dense(configs.pred_len, 
                                   activation=None,
                                   )
        self.time_encoder = make_dnn_residual(configs.time_enc_dims, 
                                              layer_norm=configs.layer_norm,
                                              dropout_rate=configs.dropout_rate,
                                              )
        
        self.final_decoder = ResBlock(hidden_dim=configs.final_dec_dim,
                                      output_dim=1, 
                                      layer_norm=configs.layer_norm,
                                      dropout_rate=configs.dropout_rate,
                                      )
        self.cat_embs = []
        for cat_size in configs.cat_sizes:
            self.cat_embs.append(tf.keras.layers.Embedding(input_dim=cat_size, 
                                                           output_dim=configs.cat_emb_size))
            
        self.ts_embs = layers.Embedding(input_dim=configs.num_targets, 
                                        output_dim=16)
            
    @tf.function
    def assemble_features(self, feats, cfeats):
        all_feats = [feats]
        for i, emb in enumerate(self.cat_embs):
            all_feats.append(tf.transpose(emb(cfeats[i,:])))
        return tf.concat(all_feats, axis=0)
    
    @tf.function
    def call(self, x, training):
        # should be: batch_x, batch_y, batch_xcov, batch_y_cov, batch_x_catcov, batch_y_catcov
        # batch_x, batch_x_cov, batch_x_catcov, batch_y, batch_y_cov, batch_y_catcov, tsidx

        batch_x, batch_x_mark = x
        categorical_x = 0 #TBD!
        bsize = 0 #TBD!
        timeseries_idx = 0 #TBD!
        
        
        
