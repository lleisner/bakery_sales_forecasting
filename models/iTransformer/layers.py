import tensorflow as tf
from tensorflow.keras import layers

class EncoderLayer(layers.Layer):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = layers.Conv1D(
            filters=d_ff,
            kernel_size=1,
            use_bias=False,
        )
        self.conv2 = layers.Conv1D(
            filters=d_model,
            kernel_size=1,
            use_bias=False,
        )
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(rate=dropout)
        self.activation = layers.ReLU() if activation == "relu" else layers.ELU()

    @tf.function
    def call(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(tf.transpose(y, perm=[0, 2, 1]))))
        y = self.dropout(self.conv2(tf.transpose(y, perm=[0, 2, 1])))
        return self.norm2(x + y), attn

class Encoder(layers.Layer):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = attn_layers
        self.conv_layers = conv_layers if conv_layers is not None else None
        self.norm = norm_layer

    @tf.function
    def call(self, x, attn_mask=None, tau=None, delta=None):
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns

