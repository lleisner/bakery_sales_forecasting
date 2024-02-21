
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from models.iTransformer.attention import AttentionLayer, FullAttention
from models.iTransformer.layers import EncoderLayer, Encoder
from models.iTransformer.embedding import DataEmbeddingInverted

from models.training import CustomModel

class Model(CustomModel):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super().__init__(configs=configs)
    
        # Embedding
        self.enc_embedding = DataEmbeddingInverted(configs.seq_len, configs.d_model, configs.dropout)
        # Encoder-only architecture
        self.encoder = Encoder(
            attn_layers=[
                EncoderLayer(
                    attention=AttentionLayer(
                                    FullAttention(
                                        False, attention_dropout=configs.dropout,output_attention=configs.output_attention
                                        ),
                                    d_model=configs.d_model, n_heads=configs.n_heads),
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=keras.layers.LayerNormalization()
        )
        self.projector = keras.layers.Dense(configs.pred_len)  
            
        self.tester = keras.layers.Dense(configs.d_ff, activation="relu")
        self.out = keras.layers.Dense(configs.num_targets)

    @tf.function
    def call(self, x, training):
        # Normalization from Non-stationary Transformer
        x_enc, x_mark_enc = x
        print(f" x, x_mark: {x_enc.shape, x_mark_enc.shape}")

        means = tf.reduce_mean(x_enc, axis=1, keepdims=True)
        x_enc = x_enc - means
        stdev = tf.sqrt(tf.math.reduce_variance(x_enc, axis=1, keepdims=True) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape  # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E          
        emb_out = self.enc_embedding(x_enc, x_mark_enc, training=training)  # covariates (e.g timestamp) can be also embedded as tokens
        print("embedding_out:", emb_out.shape)
        
        # B N E -> B N E               
        # the dimensions of embedded time series have been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(emb_out, attn_mask=None, training=training)
        # B N E -> B N S -> B S N 
        dec_out = self.projector(enc_out)
        dec_out = tf.transpose(dec_out, perm=[0, 2, 1])[:, :, :N]  # filter the covariates 
        # De-Normalization from Non-stationary Transformer
        
        
        #dec_out = self.tester(tf.concat([dec_out, x_mark_enc[:, -self.configs.pred_len:, :]], axis=-1))
        #dec_out = self.out(dec_out)

        
        if self.configs.use_norm:
            dec_out = self.norm(dec_out, means, stdev)   
         
        if self.configs.output_attention:
            return dec_out, attns
        
        return dec_out
    
    def norm(self, dec_out, means, stdev):
        # De-Normalization in TensorFlow
        stdev_t = stdev[:, 0, :]  # Selecting standard deviations for each sample in the batch
        stdev_t = tf.expand_dims(stdev_t, axis=1)  # Equivalent to unsqueeze(1) in PyTorch
        stdev_t = tf.tile(stdev_t, [1, self.configs.pred_len, 1])  # Equivalent to repeat() in PyTorch
        
        means_t = means[:, 0, :]  # Selecting means for each sample in the batch
        means_t = tf.expand_dims(means_t, axis=1)  # Equivalent to unsqueeze(1) in PyTorch
        means_t = tf.tile(means_t, [1, self.configs.pred_len, 1])  # Equivalent to repeat() in PyTorch

        dec_out = dec_out * stdev_t + means_t  # De-normalization computation
        return dec_out
     