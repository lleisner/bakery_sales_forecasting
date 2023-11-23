
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

    @tf.function
    def call(self, x_enc, x_mark_enc):
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
        # B L N -> B N E                
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # covariates (e.g timestamp) can be also embedded as tokens

        # B N E -> B N E               
        # the dimensions of embedded time series have been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N 
        dec_out = self.projector(enc_out)
        dec_out = tf.transpose(dec_out, perm=[0, 2, 1])[:, :, :N]  # filter the covariates 
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, tf.newaxis].repeat(self.pred_len, axis=1))
        dec_out = dec_out + (means[:, 0, tf.newaxis].repeat(self.pred_len, axis=1))
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]


    @tf.function
    def train_step(self, data):
        # previous sales, sales_to_predict, covariates, covariates_to_predict
        batch_x, batch_y, batch_x_mark, batch_y_mark = data
        
        batch_x = tf.cast(batch_x, dtype=tf.float32)
        batch_y = tf.cast(batch_y, dtype=tf.float32)
        batch_x_mark = tf.cast(batch_x_mark, dtype=tf.float32)
        
        with tf.GradientTape() as tape:

            outputs = self(batch_x, batch_x_mark)

            # restrict loss calculation to pred_len
            f_dim = 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

            loss = self.compute_loss(outputs, batch_y)

        gradients = tape.gradient(loss, self.trainable_variables)
        if self.args.use_amp:
            gradients = scaler.get_scaled_gradients(gradients)
        gradients = [tf.clip_by_value(grad, -self.args.clip, self.args.clip) for grad in gradients]
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss