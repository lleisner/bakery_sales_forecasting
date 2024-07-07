import tensorflow as tf

class CustomModel(tf.keras.Model):
    def __init__(self, seq_len, pred_len, num_ts, mask=True):
        super(CustomModel, self).__init__()
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.mask = mask
        self.mse_tracker = tf.keras.metrics.MeanSquaredError(name="mse")
        self.mae_tracker = tf.keras.metrics.MeanAbsoluteError(name="mae")
        self.rmse_tracker = tf.keras.metrics.RootMeanSquaredError(name="rmse")
        
    @tf.function
    def call(self, x):
        x_enc, x_mark_enc = x
        outputs = x_enc[:, -self.pred_len:, :]
        return outputs
    
    def process_data(self, data):
        return [tf.cast(tensor, dtype=tf.float32) for tensor in data]
    
    def adjust_size(self, tensor):
        return tensor[:, -self.pred_len:, :]
    
    def forward_pass(self, batch_x, batch_x_mark, training):
        return self((batch_x, batch_x_mark), training=training)
    
    def update_metrics(self, loss, batch_y, outputs):
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(batch_y, outputs)
        return {m.name: m.result() for m in self.metrics}
        
    def apply_mask(self, outputs, batch_y, batch_x_mark):
        mask_tensor = tf.not_equal(batch_x_mark[:, :, 0], 0)
        
        outputs_masked = tf.boolean_mask(outputs, mask_tensor)
        batch_y_masked = tf.boolean_mask(batch_y, mask_tensor)
            
        return outputs_masked, batch_y_masked
    
    def set_masked_values_to_zero(self, outputs, batch_x_mark):
        mask_tensor = tf.not_equal(batch_x_mark[:, :, 0], 0)
        outputs = tf.where(mask_tensor[:, :, tf.newaxis], outputs, tf.zeros_like(outputs))
        return outputs
    
    
    @tf.function
    def train_step(self, data, mask=True):
        batch_x, batch_y, batch_x_mark = self.process_data(data)
        
        outputs = self.forward_pass(batch_x, batch_x_mark, training=True)
        
        outputs, batch_y, batch_x_mark = map(self.adjust_size, [outputs, batch_y, batch_x_mark])
        
        if mask:
            outputs, batch_y = self.apply_mask(outputs, batch_y, batch_x_mark)
        
        loss = self.compute_loss(y=batch_y, y_pred=outputs)
        return self.update_metrics(loss, batch_y, outputs)


    @tf.function
    def test_step(self, data, mask=True):
        batch_x, batch_y, batch_x_mark = self.process_data(data)
        
        outputs = self.forward_pass(batch_x, batch_x_mark, training=True)
        
        outputs, batch_y, batch_x_mark = map(self.adjust_size, [outputs, batch_y, batch_x_mark])
        
        if mask:
            outputs, batch_y = self.apply_mask(outputs, batch_y, batch_x_mark)
        
        self.compute_loss(y=batch_y, y_pred=outputs)
        
        for metric in self.metrics:
            if metric.name != "loss":
                metric.update_state(batch_y, outputs)
        return {m.name: m.result() for m in self.metrics}


    @tf.function
    def predict_step(self, data, mask=True):
        batch_x, batch_y, batch_x_mark = self.process_data(data)
        
        outputs = self.forward_pass(batch_x, batch_x_mark, training=False)
        
        outputs, batch_y, batch_x_mark = map(self.adjust_size, [outputs, batch_y, batch_x_mark])

        if mask:
            outputs = self.set_masked_values_to_zero(outputs, batch_x_mark)
        
        return outputs, batch_y
    