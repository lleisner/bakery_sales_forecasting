import tensorflow as tf
from tensorflow.keras import layers

class ConvLayer(layers.Layer):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = layers.Conv1D(
            filters=c_in,
            kernel_size=3,
            padding='same',
            use_bias=False,
        )
        self.norm = layers.BatchNormalization()
        self.activation = layers.ELU()
        self.maxPool = layers.MaxPool1D(pool_size=3, strides=2, padding='same')

    def call(self, x):
        x = self.downConv(tf.transpose(x, perm=[0, 2, 1]))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = tf.transpose(x, perm=[0, 2, 1])
        return x

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

class DecoderLayer(layers.Layer):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
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
        self.norm3 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(rate=dropout)
        self.activation = layers.ReLU() if activation == "relu" else layers.ELU()

    def call(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask, tau=tau, delta=None)[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask, tau=tau, delta=delta)[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(tf.transpose(y, perm=[0, 2, 1]))))
        y = self.dropout(self.conv2(tf.transpose(y, perm=[0, 2, 1])))
        return self.norm3(x + y)

class Decoder(layers.Layer):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = layers
        self.norm = norm_layer
        self.projection = projection

    def call(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x
