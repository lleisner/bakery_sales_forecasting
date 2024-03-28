import tensorflow as tf

class CustomModel(tf.keras.Model):
    def __init__(self, seq_len, pred_len, output_attention=False, clip=None):
        super(CustomModel, self).__init__()
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.output_attention = output_attention
        self.clip = clip
        self.attn_scores = None
        
    @tf.function
    def call(self, x):
        x_enc, x_mark_enc = x
        outputs = x_enc[:, -self.pred_len:, :]
        if self.output_attention:
            return outputs, None
        return outputs
        
    @tf.function
    def train_step(self, data):
        batch_x, batch_y, batch_x_mark = [tf.cast(tensor, dtype=tf.float32) for tensor in data]
        
        with tf.GradientTape() as tape:
            if self.output_attention:
                print("using outputa-tetenions")
                outputs, attns = self((batch_x, batch_x_mark), training=True)
                self.attn_scores = attns
            else:
                outputs = self((batch_x, batch_x_mark), training=True)
            print(outputs.shape)
            
            outputs = outputs[:, -self.pred_len:, :]
            batch_y = batch_y[:, -self.pred_len:, :]
            print("outputs:", outputs.shape)
            print("batch_y:", batch_y.shape)
            
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
