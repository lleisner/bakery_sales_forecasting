
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from models.iTransformer.attention import AttentionLayer, FullAttention
from models.iTransformer.layers import EncoderLayer, Encoder
from models.iTransformer.embedding import DataEmbeddingInverted

from models.training import CustomModel

class ITransformer(keras.Model):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """
    def __init__(self, 
                 seq_len,
                 pred_len,
                 num_ts,
                 d_model=32,
                 n_heads=8,
                 d_ff=128,
                 e_layers=2,
                 dropout=0.1,
                 output_attention=True,
                 activation='gelu',
                 clip=None,
                 use_norm=True,
                 ):
        
        super().__init__()
        self.use_norm = use_norm
        self.output_attention = output_attention
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.clip = clip
        self.attn_scores=None
        self.attns = None
        
        # Embedding
        self.enc_embedding = DataEmbeddingInverted(seq_len, d_model, dropout)
        # Encoder-only architecture
        self.encoder = Encoder(
            attn_layers=[
                EncoderLayer(
                    attention=AttentionLayer(
                                    FullAttention(
                                        False, attention_dropout=dropout,output_attention=output_attention
                                        ),
                                    d_model=d_model, n_heads=n_heads),
                    d_model=d_model,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=keras.layers.LayerNormalization()
        )
        self.projector = keras.layers.Dense(pred_len)
        #self.tester = keras.layers.Dense(d_ff, activation="relu")
        #self.out = keras.layers.Dense(num_ts)
        
        
    @tf.function
    def call(self, x, training):
        # Normalization from Non-stationary Transformer
        x_enc, x_mark_enc = x
        #print(f" x, x_mark: {x_enc.shape, x_mark_enc.shape}")

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
        enc_out = self.enc_embedding(x_enc, x_mark_enc, training=training)  # covariates (e.g timestamp) can be also embedded as tokens
        #print("embedding_out:", enc_out.shape)
        
        # B N E -> B N E               
        # the dimensions of embedded time series have been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None, training=training)
        self.attns = attns
        #print("attention_scores:", attns)
        
        # B N E -> B N S -> B S N 
        dec_out = self.projector(enc_out)
        dec_out = tf.transpose(dec_out, perm=[0, 2, 1])[:, :, :N]  # filter the covariates 
        
        
        #dec_out = self.tester(tf.concat([dec_out, x_mark_enc[:, -self.pred_len:, :]], axis=-1))
        #dec_out = self.out(dec_out)

        
        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = self.norm(dec_out, means, stdev)   
         
        if self.output_attention:
            return dec_out, attns
        
        return dec_out
    
    def norm(self, dec_out, means, stdev):
        # De-Normalization in TensorFlow
        stdev_t = stdev[:, 0, :]  # Selecting standard deviations for each sample in the batch
        stdev_t = tf.expand_dims(stdev_t, axis=1)  # Equivalent to unsqueeze(1) in PyTorch
        stdev_t = tf.tile(stdev_t, [1, self.pred_len, 1])  # Equivalent to repeat() in PyTorch
        
        means_t = means[:, 0, :]  # Selecting means for each sample in the batch
        means_t = tf.expand_dims(means_t, axis=1)  # Equivalent to unsqueeze(1) in PyTorch
        means_t = tf.tile(means_t, [1, self.pred_len, 1])  # Equivalent to repeat() in PyTorch

        dec_out = dec_out * stdev_t + means_t  # De-normalization computation
        
        return dec_out
    

    @tf.function
    def train_step(self, data):
        batch_x, batch_y, batch_x_mark = [tf.cast(tensor, dtype=tf.float32) for tensor in data]
        
        with tf.GradientTape() as tape:
            if self.output_attention:
                outputs, attns = self((batch_x, batch_x_mark), training=True)
                self.attn_scores = attns
            else:
                outputs = self((batch_x, batch_x_mark), training=True)
            print(outputs.shape)
            
            outputs = outputs[:, -self.pred_len:, :]
            batch_y = batch_y[:, -self.pred_len:, :]
            
            loss = self.compute_loss(y=batch_y, y_pred=outputs)
            
        gradients = tape.gradient(loss, self.trainable_variables)
        
        if self.clip:
            gradients = [tf.clip_by_value(grad, -self.clip, self.clip) for grad in gradients]
        
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(batch_y, outputs)
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):
        batch_x, batch_y, batch_x_mark = [tf.cast(tensor, dtype=tf.float32) for tensor in data]
        
        if self.output_attention:
            outputs, attns = self((batch_x, batch_x_mark), training=False)
            self.attn_scores = attns
        else:
            outputs = self((batch_x, batch_x_mark), training=False)
        
        outputs = outputs[:, -self.pred_len:, :]
        batch_y = batch_y[:, -self.pred_len:, :]
        
        self.compute_loss(y=batch_y, y_pred=outputs)
        
        for metric in self.metrics:
            if metric.name != "loss":
                metric.update_state(batch_y, outputs)
        return {m.name: m.result() for m in self.metrics}
    
    @tf.function
    def predict_step(self, data):
        batch_x, batch_y, batch_x_mark = [tf.cast(tensor, dtype=tf.float32) for tensor in data]
        if self.output_attention:
            return self((batch_x, batch_x_mark), training=False)
        return self((batch_x, batch_x_mark), training=False)
