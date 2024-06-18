import tensorflow as tf

class CustomModel(tf.keras.Model):
    def __init__(self, seq_len, pred_len, num_ts):
        super(CustomModel, self).__init__()
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.mae_tracker = tf.keras.metrics.MeanAbsoluteError(name="mae")
        self.rmse_tracker = tf.keras.metrics.RootMeanSquaredError(name="rmse")
        
    @tf.function
    def call(self, x):
        x_enc, x_mark_enc = x
        outputs = x_enc[:, -self.pred_len:, :]
        return outputs
        
    @tf.function
    def train_step(self, data):
        batch_x, batch_y, batch_x_mark = [tf.cast(tensor, dtype=tf.float32) for tensor in data]
        
        with tf.GradientTape() as tape:

            outputs = self((batch_x, batch_x_mark), training=True)
            
            outputs = outputs[:, -self.pred_len:, :]
            batch_y = batch_y[:, -self.pred_len:, :]
            
            loss = self.compute_loss(y=batch_y, y_pred=outputs)
            
        gradients = tape.gradient(loss, self.trainable_variables) 
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
        return self((batch_x, batch_x_mark), training=False), batch_y
