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
            filters=74, # had to change this, in the original code, filters=d_model, but i dont understand why and it messes up the dim of the output
            kernel_size=1,
            use_bias=False,
        )
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(rate=dropout)
        self.activation = layers.ReLU() if activation == "relu" else layers.ELU()

    @tf.function
    def call(self, x, attn_mask=None, tau=None, delta=None):
        #print("what the hell is going on here")
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)
        #print("Shape post projection:", new_x.shape)

        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(tf.transpose(y, perm=[0, 2, 1]))))
        y = self.dropout(self.conv2(tf.transpose(y, perm=[0, 2, 1])))
        return self.norm2(x), attn
        #return self.norm2(x + y), attn

    @tf.function
    def call(self, x, attn_mask=None, tau=None, delta=None):
        # x (B, N, E) batch, number of variates, d_model
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)
        #print("Shape post projection:", new_x.shape)
        x = x + self.dropout(new_x)

        y = self.norm1(x)
        y = tf.transpose(y, perm=[0, 2, 1])  # Transpose for TensorFlow's Conv1D input format batch_size, seq_len, num_features
        #print("conv1 input shape:", y.shape)
        y = self.dropout(self.activation(self.conv1(y)))
        #print("conv2 input shape:", y.shape)
        y = self.dropout(self.conv2(y))
        #print("conv2 output shape:", y.shape)
        y = tf.transpose(y, perm=[0, 2, 1])  # Transpose back to original shape
        #print("y shape:", y.shape)
        return self.norm2(x + y), attn


    def call_test(self, inputs, attn_mask=None, tau=None, delta=None):
        x, attn = self.attention(inputs, inputs, inputs, attn_mask=attn_mask, tau=tau, delta=delta)
        x = self.dropout(x)
        res = x + inputs

        x = self.dropout(self.activation(self.conv1(tf.transpose(res, perm=[0, 2, 1]))))
        x = self.dropout(self.conv2(tf.transpose(x, perm=[0, 2, 1])))
        return x + res

class Encoder(layers.Layer):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = attn_layers
        self.norm = norm_layer

    @tf.function
    def call(self, x, attn_mask=None, tau=None, delta=None):
        attns = []

        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns

