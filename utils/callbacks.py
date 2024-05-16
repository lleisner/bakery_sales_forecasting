import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, CSVLogger, TerminateOnNaN

def get_callbacks(num_epochs, model_name, dataset_name):
    # Set patience to 10% of total number of epochs
    patience = max(1, num_epochs // 10)

    # Create necessary directories
    base_dir = f'logs/{dataset_name}/{model_name}'
    checkpoint_dir = f'{base_dir}/checkpoints'
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
    
    model_checkpoint = ModelCheckpoint(
        filepath=f'{checkpoint_dir}/best_model',
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

    return [early_stopping, model_checkpoint, reduce_lr, tensorboard, csv_logger, terminate_on_nan]
