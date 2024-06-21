

from models.iTransformer.new_data_loader import ITransformerData
from models.iTransformer.i_transformer import ITransformer
from models.iTransformer.model_tuner import build_itransformer
from models.tide_google.tide_model import TiDE
from models.tide_google.new_data_loader import TiDEData
from models.tide_google.model_tuner import build_tide
from models.training import CustomModel

from utils.analyze_data import analyze_all_datasets
from utils.plot_attention import plot_attention_weights
from utils.plot_preds_and_actuals import plot_time_series
from utils.callbacks import get_callbacks
import yaml
import argparse
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras import mixed_precision

from tensorflow.keras import backend
import gc
#from numba import cuda

#os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
#mixed_precision.set_global_policy('mixed_float16')
def clear_keras_session():
    backend.clear_session()

def free_gpu_mem():
    device = cuda.get_current_device()
    device.reset()

class ClearSessionTuner(kt.Hyperband):
    def run_trial(self, trial, *args, **kwargs):
        try:
            # Capture the result of the run_trial method
            result = super(ClearSessionTuner, self).run_trial(trial, *args, **kwargs)
        finally:
            # Ensure the session is cleared after each trial
            clear_keras_session()
        return result


def parse_arguments():
    parser = argparse.ArgumentParser(description='Configure model parameters.')

    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training the model.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs for training.')
    parser.add_argument('--config_file', type=str, default="experiment/dataset_analysis.yaml", help='Path to the YAML configuration file.')
    parser.add_argument('--data_directory', type=str, required=True, help='Path to data directory')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to be used for experiment')
    parser.add_argument('--model', type=str, default='Baseline', help='Model to be used for experiment')
    parser.add_argument('--tune_hps', type=bool, default=False, help='Tune hyperparameters for dataset')

    args = parser.parse_args()
    return args


def tune_model_on_dataset(name, model_builder, data_loader):

    clear_keras_session()
    #gc.collect()
    #free_gpu_mem()
    
    args = parse_arguments()
    directory = f"experiment/hyperparameters/{args.dataset}"
    

    
    loader_config = data_loader.create_loader_config(args)
    
    print(f"data directory: {args.data_directory}, filename: {loader_config['data_path']}")

    data_loader = data_loader(**loader_config)
    
    train, val, test = data_loader.get_train_test_splits()
    
    callbacks = get_callbacks(num_epochs=args.num_epochs, model_name=name, dataset_name=args.dataset, mode='tuning')

    hypermodel = lambda hp: model_builder(hp=hp, 
                                          learning_rate=args.learning_rate, 
                                          seq_len=loader_config['hist_len'], 
                                          pred_len=loader_config['pred_len'], 
                                          num_ts=len(loader_config['timeseries_cols']))
    
    tuner = ClearSessionTuner(hypermodel=hypermodel,
                         objective='val_loss',
                         max_epochs=args.num_epochs,
                         factor=3,
                         hyperband_iterations=3,
                         directory=directory,
                         project_name=name,
                         executions_per_trial=3,
                         max_retries_per_trial=1,
                         )


    tuner.search_space_summary()
    tuner.search(train, epochs=args.num_epochs, validation_data=val, callbacks=callbacks, verbose=2)
    #tuner.reload()
    summary = tuner.results_summary(3)

    best_hps = tuner.get_best_hyperparameters(3)
    print(f"best hyperparameters for {directory}/{name}: {best_hps}")

    
def train_model_on_dataset(name, model, data_loader):
    args = parse_arguments()
    directory = f"experiment/hyperparameters/{args.dataset}"
    
    loader_config = data_loader.create_loader_config(args)
    data_loader = data_loader(**loader_config)
    
    train, val, test = data_loader.get_train_test_splits()
    
    callbacks = get_callbacks(num_epochs=args.num_epochs, model_name=name, dataset_name=args.dataset, mode='training')

    model = model(seq_len=loader_config['hist_len'], 
                pred_len=loader_config['pred_len'], 
                num_ts=len(loader_config['timeseries_cols']))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.learning_rate),
        loss='mse',
        metrics=['mae'],
        weighted_metrics=[],
    )
    
    model.fit(train, epochs=args.num_epochs, validation_data=val, callbacks=callbacks)
    
    result = model.evaluate(test)
    
    update_results(dataset=args.dataset, 
                   model=name, 
                   metrics=result, 
                   file_path="experiment/one_day_forecast_results.csv",
                   )
    
    model.summary()
    
def common_setup(data_loader):
    """
    Common setup function to parse arguments, create data loader configuration, 
    and obtain train, validation, and test splits.

    Parameters:
    - data_loader (function): The function to create and load the data.

    Returns:
    - args: Parsed arguments.
    - loader_config: Configuration for the data loader.
    - train: Training data split.
    - val: Validation data split.
    - test: Test data split.
    """
    args = parse_arguments()
    
    loader_config = data_loader.create_loader_config(args)
    data_loader = data_loader(**loader_config)
    
    train, val, test = data_loader.get_train_test_splits()
    
    clear_keras_session()
    
    return args, loader_config, train, val, test

def tune_model_on_dataset(name, model_builder, data_loader):
    """
    Tunes the model on a given dataset using Hyperband for hyperparameter optimization.

    Parameters:
    - name (str): Name of the model.
    - model_builder (function): Function to build the model.
    - data_loader (function): Function to create and load the data.
    """
    
    args, loader_config, train, val, _ = common_setup(data_loader)
    
    directory = f"experiment/hyperparameters/{args.dataset}"
    
    callbacks = get_callbacks(num_epochs=args.num_epochs, model_name=name, dataset_name=args.dataset, mode='tuning')

    hypermodel = lambda hp: model_builder(hp=hp, 
                                          learning_rate=args.learning_rate, 
                                          seq_len=loader_config['hist_len'], 
                                          pred_len=loader_config['pred_len'], 
                                          num_ts=len(loader_config['timeseries_cols']))
    
    tuner = ClearSessionTuner(hypermodel=hypermodel,
                         objective='val_loss',
                         max_epochs=args.num_epochs,
                         factor=3,
                         hyperband_iterations=3,
                         directory=directory,
                         project_name=name,
                         executions_per_trial=3,
                         max_retries_per_trial=1,
                         )

    tuner.search_space_summary()
    tuner.search(train, epochs=args.num_epochs, validation_data=val, callbacks=callbacks, verbose=2)
    tuner.results_summary(3)

    best_hps = tuner.get_best_hyperparameters(3)
    print(f"best hyperparameters for {directory}/{name}: {best_hps}")

def train_model_on_dataset(name, model, data_loader):
    """
    Trains the model on a given dataset with the provided configuration and saves the evaluation results.

    Parameters:
    - name (str): Name of the model.
    - model (tf.keras.Model): The model to be trained.
    - data_loader (function): Function to create and load the data.
    """
    args, loader_config, train, val, test = common_setup(data_loader)
    
    callbacks = get_callbacks(num_epochs=args.num_epochs, model_name=name, dataset_name=args.dataset, mode='training')

    model = model(seq_len=loader_config['hist_len'], 
                pred_len=loader_config['pred_len'], 
                num_ts=len(loader_config['timeseries_cols']))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.learning_rate),
        loss='mse',
        metrics=['mae'],
        weighted_metrics=[],
    )
    
    model.fit(train, epochs=args.num_epochs, validation_data=val, callbacks=callbacks)
    
    result = model.evaluate(test)
    
    update_results(dataset=args.dataset, 
                   model=name, 
                   metrics=result, 
                   file_path="experiment/one_day_forecast_results.csv",
                   )
    
    model.summary()

def update_results(dataset, model, metrics, file_path="model_evaluation_results.csv"):
    """
    Updates the results for a given dataset and model with new metrics and saves the updated DataFrame to a CSV file.
    
    Parameters:
    - dataset (str): The name of the dataset.
    - model (str): The name of the model.
    - metrics (tuple): A tuple containing the metrics (mse, mae, rmse).
    - file_path (str): The path to the CSV file. Default is "model_evaluation_results.csv".
    """
    # Read the existing CSV file
    df = pd.read_csv(file_path, header=[0, 1], index_col=0)
    
    # Ensure the DataFrame has the necessary columns and index
    if dataset not in df.index:
        # Add the dataset row if it doesn't exist
        print("name does not fit")
        df.loc[dataset] = pd.Series(dtype='float64')
    
    # Update the DataFrame with new results
    try:
        mse, mae, rmse = metrics
        df.loc[dataset, (model, 'mse')] = mse
        df.loc[dataset, (model, 'mae')] = mae
        df.loc[dataset, (model, 'rmse')] = rmse
    except:
        df.loc[dataset, (model, 'mse')] = metrics
    # Save the updated DataFrame to the CSV file
    df.to_csv(file_path)
    
if __name__ == "__main__":
    
    
    #os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

 #   print("Is GPU available:", tf.test.is_gpu_available())
#    print("Built with CUDA:", tf.test.is_built_with_cuda())

  #  gpus = tf.config.list_physical_devices('GPU')
 #   if gpus:
 #       try:
  #          for gpu in gpus:
   #             tf.config.experimental.set_memory_growth(gpu, True)
    #    except RuntimeError as e:
     #       print(e)

   # print("GPUs:", gpus)

    #train_model_on_dataset("TiDE", TiDE, TiDEData)
    train_model_on_dataset("iTransformer", ITransformer, ITransformerData)
    #train_model_on_dataset("Baseline", CustomModel, ITransformerData)
    

    #tune_model_on_dataset("TiDE", build_tide, TiDEData)
    #tune_model_on_dataset("iTransformer", build_itransformer, ITransformerData)
    #return result
    
    """
    yaml_filepath = "experiment/dataset_analysis.yaml"
    args = parse_arguments(yaml_filepath)
    
    analyze_all_datasets(args.data_directory, infer_ts_cols=True, yaml_output_path=yaml_filepath)
    
    itransformer_data_config = ITransformerData.create_loader_config(args)
    itransformer_data = ITransformerData(**itransformer_data_config)

    df = itransformer_data.get_data()
    print(df)
    print(df.columns)
    
    train, val, test = itransformer_data.get_train_test_splits()
    print(itransformer_data.get_feature_names_out())
    
    baseline = CustomModel(seq_len=itransformer_data_config['hist_len'], 
                           pred_len=itransformer_data_config['pred_len'])
    
    
    loss = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    callbacks = get_callbacks(num_epochs=args.num_epochs, model_name="iTransformer", dataset_name=args.dataset)

    
    baseline.compile(optimizer=optimizer, loss=loss, weighted_metrics=[])
    
    tune_model_on_dataset("iTransformer", build_itransformer, itransformer_data, callbacks)
    
    
    #itransformer = ITransformer(**model_config)
    
    #tide = TiDE(**model_config)
    
    loss = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    
    baseline.compile(optimizer=optimizer, loss=loss, weighted_metrics=[])
    
    #itransformer.compile(optimizer=optimizer, loss=loss, metrics=['mae'],weighted_metrics=[])
    
    #tide.compile(optimizer=optimizer, loss=loss, metrics=['mae'], weighted_metrics=[])
    
    callbacks = get_callbacks(num_epochs=args.num_epochs, model_name="iTransformer", dataset_name=args.dataset)
    
    baseline.fit(train, epochs=1, validation_data=val)
    
    #hist = itransformer.fit(train, epochs=args.num_epochs, validation_data=val, callbacks=callbacks)
    
    print(f"baseline evaluation on test data: {baseline.evaluate(test)}")
    print(f"itransformer evaluation on test data: {itransformer.evaluate(test)}")
    
    itransformer.summary()
    
    sample = test.take(1)
    itransformer_preds, attns = itransformer.predict(sample)
    baseline_preds, actuals = baseline.predict(sample)
    
    p, a = itransformer.predict(test)
    b, r = baseline.predict(test)
    print(p.shape)
    print(p.shape, b.shape, r.shape)
    
    p, r, b = p.reshape(args.batch_size * loader_config['pred_len'], model_config['num_targets']), r.reshape(args.batch_size * loader_config['pred_len'], model_config['num_targets']), b.reshape(args.batch_size * loader_config['pred_len'],  model_config['num_targets'])
    labels = transformer_data.get_feature_names_out()[:-model_config['num_targets']]
    
    t_preds = itransformer_preds[0]
    b_preds = baseline_preds[0]
    actuals = actuals[0]
    attn_heads = attns[0][0]
    
    
    plot_attention_weights(labels, attn_heads)
    plot_time_series(actuals, b_preds, t_preds)
    plot_time_series(r, b, p)
    """