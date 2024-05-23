import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, CSVLogger, TerminateOnNaN
import time

class HeartbeatCallback(tf.keras.callbacks.Callback):
    def __init__(self, heartbeat_file, interval):
        super().__init__()
        self.heartbeat_file = heartbeat_file
        self.interval = interval
        self.last_update = time.time()

    def on_train_bach_end(self, batch, logs=None):
        current_time = time.time()
        if current_time - self.last_update > self.interval:
            with open(self.heartbeat_file, 'w') as f:
                f.write('alive')
            self.last_update = current_time

def get_callbacks(num_epochs, model_name, dataset_name, mode='training', heartbeat_file="/tmp/heartbeat", heartbeat_interval=120):
    # Set patience to 20% of total number of epochs
    patience = max(1, num_epochs // 5)

    # Create necessary directories
    base_dir = f'logs/{dataset_name}/{model_name}'
    checkpoint_dir = f'{base_dir}/checkpoints'
    csv_log_dir = f'{base_dir}/csv'
    tensorboard_log_dir = f'{base_dir}/tensorboard'
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(csv_log_dir, exist_ok=True)
    os.makedirs(tensorboard_log_dir, exist_ok=True)

    heartbeat = HeartbeatCallback(
        heartbeat_file=heartbeat_file,
        interval=heartbeat_interval,
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        verbose=1,
        restore_best_weights=True
    )
    
    model_checkpoint = ModelCheckpoint(
        filepath=f'{checkpoint_dir}/best_model',
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        verbose=0
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
    