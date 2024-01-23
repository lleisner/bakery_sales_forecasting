import tensorflow as tf
from tensorflow.keras import layers

class DataEmbeddingInverted(tf.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbeddingInverted, self).__init__()
        self.value_embedding = layers.Dense(d_model)
        self.dropout = layers.Dropout(rate=dropout)

    def __call__(self, x, x_mark, training):
        x = tf.transpose(x, perm=[0, 2, 1])
        # x: [Batch Time Variate]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            # the potential to take covariates (e.g. timestamps) as tokens
            x = self.value_embedding(tf.concat([x, tf.transpose(x_mark, perm=[0, 2, 1])], axis=1))
        # x: [Batch Time d_model]
        return self.dropout(x, training=training)