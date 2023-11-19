
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from models.iTransformer.attention import AttentionLayer, FullAttention
from models.iTransformer.layers import EncoderLayer, Encoder
from models.iTransformer.embedding import DataEmbeddingInverted

class Model(keras.Model):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        # Embedding
        self.enc_embedding = DataEmbeddingInverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=keras.layers.LayerNormalization()
        )
        self.projector = keras.layers.Dense(configs.pred_len, use_bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = tf.reduce_mean(x_enc, axis=1, keepdims=True)
        x_enc = x_enc - means
        stdev = tf.sqrt(tf.math.reduce_variance(x_enc, axis=1, keepdims=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape  # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # covariates (e.g timestamp) can be also embedded as tokens

        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series have been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N 
        dec_out = self.projector(enc_out)
        dec_out = tf.transpose(dec_out, perm=[0, 2, 1])[:, :, :N]  # filter the covariates 
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, tf.newaxis].repeat(self.pred_len, axis=1))
        dec_out = dec_out + (means[:, 0, tf.newaxis].repeat(self.pred_len, axis=1))
        return dec_out

    def call(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]