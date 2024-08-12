# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#
# Modifications made by Lorenz Leisner, 2024


import tensorflow as tf
from tensorflow import keras


class MLPResidual(keras.layers.Layer):
  """Simple one hidden state residual network."""

  def __init__(
      self, hidden_dim, output_dim, layer_norm=False, dropout=0.0, activation='relu',
  ):
    super(MLPResidual, self).__init__()
    self.lin_a = tf.keras.layers.Dense(
        hidden_dim,
        activation=activation,
    )
    self.lin_b = tf.keras.layers.Dense(
        output_dim,
        activation=None,
    )
    self.lin_res = tf.keras.layers.Dense(
        output_dim,
        activation=None,
    )
    if layer_norm:
      self.lnorm = tf.keras.layers.LayerNormalization()
    self.layer_norm = layer_norm
    self.dropout = tf.keras.layers.Dropout(dropout)

  def call(self, inputs):
    """Call method."""
    h_state = self.lin_a(inputs)
    out = self.lin_b(h_state)
    out = self.dropout(out)
    res = self.lin_res(inputs)
    if self.layer_norm:
      return self.lnorm(out + res)
    return out + res


def _make_dnn_residual(hidden_dims, layer_norm=False, dropout=0.0, activation='relu'):
  """Multi-layer DNN residual model."""
  print("hidden_dims:", hidden_dims)
  print(len(hidden_dims))

  if len(hidden_dims) < 2:
    return keras.layers.Dense(
        hidden_dims[-1],
        activation=None,
    )
  layers = []
  for i, hdim in enumerate(hidden_dims[:-1]):
    layers.append(
        MLPResidual(
            hdim,
            hidden_dims[i + 1],
            layer_norm=layer_norm,
            dropout=dropout,
            activation=activation,
        )
    )
  return keras.Sequential(layers)
  

class TiDE(keras.Model):
  """Main class for multi-scale DNN model."""

  def __init__(
      self,
      seq_len,
      pred_len,
      num_ts,
      cat_sizes=[],
      hidden_size=256,
      decoder_output_dim=8,
      final_decoder_hidden=64,
      time_encoder_dims=[64, 4],
      num_layers=3,
      dropout=0.2,
      activation='relu',
      transform=True,
      cat_emb_size=4,
      layer_norm=True,
      mask=True,
  ):
    """Tide model.

    Args:
      pred_len: prediction horizon length.
      cat_sizes: number of categories in each categorical covariate.
      num_ts: number of time-series in the dataset
      transform: apply reversible transform or not.
      cat_emb_size: embedding size of categorical variables.
      layer_norm: use layer norm or not.
      dropout: level of dropout.
    """
    super().__init__()
    hidden_dims = [hidden_size] * num_layers
    
    self.mask = mask
    self.transform = transform

    self.mse_tracker = tf.keras.metrics.MeanSquaredError(name="mse")
    self.mae_tracker = tf.keras.metrics.MeanAbsoluteError(name="mae")
    self.rmse_tracker = tf.keras.metrics.RootMeanSquaredError(name="rmse")
    self.mape_tracker = tf.keras.metrics.MeanAbsolutePercentageError(name="mape")


    if self.transform:
      self.affine_weight = self.add_weight(
          name='affine_weight',
          shape=(num_ts,),
          initializer='ones',
          trainable=True,
      )

      self.affine_bias = self.add_weight(
          name='affine_bias',
          shape=(num_ts,),
          initializer='zeros',
          trainable=True,
      )
    self.pred_len = pred_len
    self.encoder = _make_dnn_residual(
        hidden_dims,
        layer_norm=layer_norm,
        dropout=dropout,
        activation=activation
    )
    self.decoder = _make_dnn_residual(
        hidden_dims[:-1]
        + [
            decoder_output_dim * self.pred_len,
        ],
        layer_norm=layer_norm,
        dropout=dropout,
        activation=activation,
    )
    self.linear = tf.keras.layers.Dense(
        self.pred_len,
        activation=None,
    )
    self.time_encoder = _make_dnn_residual(
        time_encoder_dims,
        layer_norm=layer_norm,
        dropout=dropout,
        activation=activation
    )
    self.final_decoder = MLPResidual(
        hidden_dim=final_decoder_hidden,
        output_dim=1,
        layer_norm=layer_norm,
        dropout=dropout,
        activation=activation,
    )
    self.cat_embs = []
    for cat_size in cat_sizes:
      self.cat_embs.append(
          tf.keras.layers.Embedding(input_dim=cat_size, output_dim=cat_emb_size)
      )
    self.ts_embs = tf.keras.layers.Embedding(input_dim=num_ts, output_dim=16)


  @tf.function
  def _assemble_feats(self, feats, cfeats):
    """assemble all features."""
    all_feats = [feats]
    for i, emb in enumerate(self.cat_embs):
      all_feats.append(tf.transpose(emb(cfeats[i, :])))
    return tf.concat(all_feats, axis=0)

  @tf.function
  def call(self, inputs):
    """Call function that takes in a batch of training data and features."""
    past_data = inputs[0]
    future_features = inputs[1]
    bsize = past_data[0].shape[0]
    tsidx = inputs[2]
    print(f"inputs: {inputs}\npast_data: {past_data}\nfuture_features: {future_features}\nbatch_size: {bsize}\ntsidx: {tsidx}")
    past_feats = self._assemble_feats(past_data[1], past_data[2])
    future_feats = self._assemble_feats(future_features[0], future_features[1])
    past_ts = past_data[0]
    if self.transform:
      affine_weight = tf.gather(self.affine_weight, tsidx)
      affine_bias = tf.gather(self.affine_bias, tsidx)
      batch_mean = tf.math.reduce_mean(past_ts, axis=1)
      batch_std = tf.math.reduce_std(past_ts, axis=1)
      batch_std = tf.where(
          tf.math.equal(batch_std, 0.0), tf.ones_like(batch_std), batch_std
      )
      past_ts = (past_ts - batch_mean[:, None]) / batch_std[:, None]
      past_ts = affine_weight[:, None] * past_ts + affine_bias[:, None]
    encoded_past_feats = tf.transpose(
        self.time_encoder(tf.transpose(past_feats))
    )
    encoded_future_feats = tf.transpose(
        self.time_encoder(tf.transpose(future_feats))
    )
    enc_past = tf.repeat(tf.expand_dims(encoded_past_feats, axis=0), bsize, 0)
    enc_past = tf.reshape(enc_past, [bsize, -1])
    enc_fut = tf.repeat(
        tf.expand_dims(encoded_future_feats, axis=0), bsize, 0
    )  # batch x fdim x H
    enc_future = tf.reshape(enc_fut, [bsize, -1])
    residual_out = self.linear(past_ts)
    ts_embs = self.ts_embs(tsidx)
    encoder_input = tf.concat([past_ts, enc_past, enc_future, ts_embs], axis=1)
    encoding = self.encoder(encoder_input)
    decoder_out = self.decoder(encoding)
    decoder_out = tf.reshape(
        decoder_out, [bsize, -1, self.pred_len]
    )  # batch x d x H
    final_in = tf.concat([decoder_out, enc_fut], axis=1)
    out = self.final_decoder(tf.transpose(final_in, (0, 2, 1)))  # B x H x 1
    out = tf.squeeze(out, axis=-1)
    out += residual_out
    if self.transform:
      out = (out - affine_bias[:, None]) / (affine_weight[:, None] + 1e-7)
      out = out * batch_std[:, None] + batch_mean[:, None]

    
    cheat = False
    if cheat:
      # this is used to ensure the TiDE model procudes the same Baseline results as the iTransformer
      past_data = inputs[0]
      past_ts = past_data[0]
      out = past_ts[:, -self.pred_len:]
      
    return out

  def update_metrics(self, loss, y_true, y_pred):
    for metric in self.metrics:
      if metric.name == "loss":
        metric.update_state(loss)
      #else:
        #metric.update_state(y_true, y_pred)
    return {m.name: m.result() for m in self.metrics}
  
  
  def get_mask(self, inputs):
    # Get the is_open feature from the inputs
    past_data, future_features, _ = inputs
    bsize = past_data[0].shape[0]
    bfeats_pred, _ = future_features
    is_open = bfeats_pred[7, :]
    
    # Create a mask for the loss calculation
    mask_tensor = tf.not_equal(is_open, 0)
    mask_tensor = tf.expand_dims(mask_tensor, axis=0)
    mask_tensor = tf.tile(mask_tensor, [bsize, 1])

    return mask_tensor

  
  def apply_mask(self, inputs, y_pred, y_true):

    mask_tensor = self.get_mask(inputs)
    y_pred_masked = tf.boolean_mask(y_pred, mask_tensor)
    y_true_masked = tf.boolean_mask(y_true, mask_tensor)

      
    # Make sure the mask does not result in zero tensors, or loss is NaN and terminates training!
    if tf.size(y_pred_masked) == 0 or tf.size(y_true_masked) == 0:
      #tf.print("Zero size in masked values", (y_pred_masked, y_true_masked, y_pred, y_true))
      return y_pred, y_true
    return y_pred_masked, y_true_masked
  
  def set_masked_values_to_zero(self, inputs, y_pred):
    mask_tensor = self.get_mask(inputs)
    y_pred = tf.where(mask_tensor, y_pred, tf.zeros_like(y_pred))
    return y_pred

  
  @tf.function
  def train_step(self, data):
    inputs, y_true = tsd.prepare_batch(*data) 
    
    #inputs, y_true = data
    with tf.GradientTape() as tape:
      y_pred = self(inputs, training=True)
      
      if self.mask:
        y_pred, y_true = self.apply_mask(inputs, y_pred, y_true)


      loss = self.compute_loss(y=y_true, y_pred=y_pred)
    grads = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
    
    return self.update_metrics(loss, y_true, y_pred)

  @tf.function
  def train_step2(self, data):
    # This function is used for the Baseline test with cheat=True
    inputs, y_true = tsd.prepare_batch(*data)

    y_pred = self(inputs, training=True)

    if self.mask:
      y_pred, y_true = self.apply_mask(inputs, y_pred, y_true)

    loss = self.compute_loss(y=y_true, y_pred=y_pred)

    return self.update_metrics(loss, y_true, y_pred)


  @tf.function
  def test_step(self, data):
    inputs, y_true = tsd.prepare_batch(*data)
    y_pred = self(inputs, training=False)
    
    if self.mask:
      y_pred, y_true = self.apply_mask(inputs, y_pred, y_true)
      
    self.compute_loss(y=y_true, y_pred=y_pred)
    
    for metric in self.metrics:
      if metric.name != "loss":
        metric.update_state(y_true, y_pred)
    return {m.name: m.result() for m in self.metrics}

  @tf.function
  def predict_step(self, data):
    inputs, y_true = tsd.prepare_batch(*data)
    y_pred = self(inputs, training=False)
    print(f"shape of y_pred before : {y_pred.shape}")
    
    if self.mask:
      y_pred = self.set_masked_values_to_zero(inputs, y_pred)
    
    # The resulting tensor has shape (n_variates, pred_len), so transpose to (pred_len, n_variates) to get the required output dims.
    y_pred_transposed = tf.transpose(y_pred, perm=[1, 0])
    y_true_transposed = tf.transpose(y_true, perm=[1, 0])
    print(f"shape of y_pred after : {y_pred_transposed.shape}")
    
    return y_pred_transposed, y_true_transposed