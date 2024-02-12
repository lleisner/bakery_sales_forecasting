import tensorflow as tf

class CombinedHistory(tf.keras.callbacks.Callback):
    def __init__(self):
        self.history = {'loss': [], 'val_loss': []}  # Initialize with relevant metrics

    def on_epoch_end(self, epoch, logs=None):
        # Append the metrics for the current epoch to the history
        for metric_name, metric_value in logs.items():
            if metric_name in self.history:
                self.history[metric_name].append(metric_value)
