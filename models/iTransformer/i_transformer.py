
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
                 d_ff=256,
                 e_layers=2,
                 dropout=0.1,
                 output_attention=False,
                 activation='relu',
                 clip=None,
                 use_norm=True,
                 mask=True,
                 ):
        
        super().__init__()
        self.use_norm = use_norm
        self.output_attention = output_attention
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.clip = clip
        self.mask = mask
        self.attn_scores=None
        self.attns = None
        self.mae_tracker = tf.keras.metrics.MeanAbsoluteError(name="mae")
        self.rmse_tracker = tf.keras.metrics.RootMeanSquaredError(name="rmse")
        
        
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
        
        
        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = self.norm(dec_out, means, stdev)   
         
        if self.output_attention:
            return dec_out, attns
        
        return dec_out
    
    
    def norm(self, dec_out, means, stdev):
        # De-normalization
        stdev_exp = tf.tile(tf.expand_dims(stdev[:,0,:], axis=1), [1, self.pred_len, 1])
        means_exp = tf.tile(tf.expand_dims(means[:,0,:], axis=1), [1, self.pred_len, 1])

        dec_out = dec_out * stdev_exp + means_exp
        return dec_out
    
    
    
    def process_data(self, data):
        return [tf.cast(tensor, dtype=tf.float32) for tensor in data]
    
    def adjust_size(self, tensor):
        return tensor[:, -self.pred_len:, :]
    
    def forward_pass(self, batch_x, batch_x_mark, training):
        if self.output_attention:
            outputs, attns = self((batch_x, batch_x_mark), training=training)
            return outputs, attns
        else:
            outputs = self((batch_x, batch_x_mark), training=training)
            return outputs, None
        
    def apply_mask(self, outputs, batch_y, batch_x_mark):
        mask_tensor = tf.not_equal(batch_x_mark[:, :, 0], 0)
        
        outputs_masked = tf.boolean_mask(outputs, mask_tensor)
        batch_y_masked = tf.boolean_mask(batch_y, mask_tensor)
            
        return outputs_masked, batch_y_masked
    
    def set_masked_values_to_zero(self, outputs, batch_x_mark):
        mask_tensor = tf.not_equal(batch_x_mark[:, :, 0], 0)
        outputs = tf.where(mask_tensor[:, :, tf.newaxis], outputs, tf.zeros_like(outputs))
        return outputs
    
    def compute_and_apply_gradients(self, tape, loss):
        gradients = tape.gradient(loss, self.trainable_variables)
        
        if self.clip:
            gradients = [tf.clip_by_value(grad, -self.clip, self.clip) for grad in gradients]

        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
    def update_metrics(self, loss, batch_y, outputs):
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(batch_y, outputs)
        return {m.name: m.result() for m in self.metrics}
    

    
    
    @tf.function
    def train_step(self, data):
        batch_x, batch_y, batch_x_mark = self.process_data(data)
        
        with tf.GradientTape() as tape:
            outputs, _ = self.forward_pass(batch_x, batch_x_mark, training=True)

            # Get the correct number of prediction steps for loss calculation
            outputs, batch_y, batch_x_mark = map(self.adjust_size, [outputs, batch_y, batch_x_mark])
            
            if self.mask:
                outputs, batch_y = self.apply_mask(outputs, batch_y, batch_x_mark)
                
            loss = self.compute_loss(y=batch_y, y_pred=outputs)
        
        self.compute_and_apply_gradients(tape, loss)
        
        return self.update_metrics(loss, batch_y, outputs)
        
         
    @tf.function
    def test_step(self, data):
        batch_x, batch_y, batch_x_mark = self.process_data(data)
        outputs, attns = self.forward_pass(batch_x, batch_x_mark, training=False)
        
        outputs, batch_y, batch_x_mark = map(self.adjust_size, [outputs, batch_y, batch_x_mark])
        
        if self.mask:
            outputs, batch_y = self.apply_mask(outputs, batch_y, batch_x_mark)
            
        self.compute_loss(y=batch_y, y_pred=outputs)
        
        for metric in self.metrics:
            if metric.name != "loss":
                metric.update_state(batch_y, outputs)
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def predict_step(self, data):
        batch_x, batch_y, batch_x_mark = self.process_data(data)
        
        outputs, attns = self.forward_pass(batch_x, batch_x_mark, training=False)
        
        outputs, batch_y, batch_x_mark = map(self.adjust_size, [outputs, batch_y, batch_x_mark])

        if self.mask:
            outputs = self.set_masked_values_to_zero(outputs, batch_x_mark)
            
        if self.output_attention:
            return outputs, attns
        
        return outputs, batch_y
    