import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, CSVLogger, TerminateOnNaN
import time
import json
import numpy as np

            
class CustomModelCheckpoint(ModelCheckpoint):
    """
    Extends on the ModelCheckpoint class to add functionality for 
    loading and saving the best metric value across multiple training 
    runs. Only saves the model if it is better than the current best 
    model across all training runs.

    This class extends `tf.keras.callbacks.ModelCheckpoint` to include:
    - Loading the initial best value from a JSON file if it exists.
    - Saving the best metric value to a JSON file at the end of training.

    Args:
        keep_best_model: defaults to True, if False, functions exactly like ModelCheckpoint
        *args: Variable length argument list for passing to the parent class.
        **kwargs: Arbitrary keyword arguments for passing to the parent class.
    """
    
    def __init__(self, metrics_filepath, keep_best_model=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_filepath = metrics_filepath
        if keep_best_model:
            # Set the initial_value_threshold to the best model recorded
            self.best = self._load_best_value()
       
    def _load_best_value(self):
        metric_path = os.path.join(os.path.dirname(self.metrics_filepath), 'best_metric.json')
        if os.path.exists(metric_path):
            with open(metric_path, 'r') as f:
                best_metric = json.load(f).get(self.monitor, None)
                if best_metric is not None:
                    return best_metric
        return float('inf') if self.monitor_op == np.less else float('-inf')
        
    def on_train_end(self, logs=None):
        super().on_train_end(logs)
        self._save_best_metric()
        
    def _save_best_metric(self):
        metric_path = os.path.join(os.path.dirname(self.metrics_filepath), 'best_metric.json')
        with open(metric_path, 'w') as f:
            json.dump({self.monitor: self.best}, f)
            
        
def get_callbacks(num_epochs, model_name, dataset_name, mode='training'):
    # Set patience to 20% of total number of epochs
    patience = max(1, num_epochs // 5)

    # Create necessary directories
    base_dir = f'logs/{dataset_name}/{model_name}'
    checkpoint_dir = f'{base_dir}/checkpoints/'
    csv_log_dir = f'{base_dir}/csv'
    tensorboard_log_dir = f'{base_dir}/tensorboard'
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(csv_log_dir, exist_ok=True)
    os.makedirs(tensorboard_log_dir, exist_ok=True)


    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        verbose=1,
        restore_best_weights=True
    )
    
    model_checkpoint = CustomModelCheckpoint(
        filepath=f'{checkpoint_dir}weigths.h5',
        metrics_filepath=f'{checkpoint_dir}',
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=patience // 2,
        verbose=1
    )

    tensorboard = TensorBoard(
        log_dir=tensorboard_log_dir,
        histogram_freq=1
    )

    csv_logger = CSVLogger(f'{csv_log_dir}/training_log.csv')

    terminate_on_nan = TerminateOnNaN()

    if mode == 'tuning':
        return [early_stopping, reduce_lr, terminate_on_nan]
    else:
        return [early_stopping, model_checkpoint, tensorboard, reduce_lr, csv_logger, terminate_on_nan]
    