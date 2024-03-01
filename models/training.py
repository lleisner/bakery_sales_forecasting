import tensorflow as tf

class CustomModel(tf.keras.Model):
    def __init__(self, configs):
        super(CustomModel, self).__init__()
        self.configs = configs
        
    @tf.function
    def call(self, x):
        x_enc, x_mark_enc = x
        return x_enc[:, -self.configs.pred_len:, :]
        
    @tf.function
    def train_step(self, data):
        batch_x, batch_y, batch_x_mark = [tf.cast(tensor, dtype=tf.float32) for tensor in data]
        
        with tf.GradientTape() as tape:
            outputs = self((batch_x, batch_x_mark), training=True)
            print(outputs.shape)
            
            outputs = outputs[:, -self.configs.pred_len:, :]
            batch_y = batch_y[:, -self.configs.pred_len:, :]
            
            loss = self.compute_loss(y=batch_y, y_pred=outputs)
            
        gradients = tape.gradient(loss, self.trainable_variables)
        
        if self.configs.clip:
            gradients = [tf.clip_by_value(grad, -self.configs.clip, self.configs.clip) for grad in gradients]
        
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
        
        outputs = outputs[:, -self.configs.pred_len:, :]
        batch_y = batch_y[:, -self.configs.pred_len:, :]
        
        self.compute_loss(y=batch_y, y_pred=outputs)
        
        for metric in self.metrics:
            if metric.name != "loss":
                metric.update_state(batch_y, outputs)
        return {m.name: m.result() for m in self.metrics}