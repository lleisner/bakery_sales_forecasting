import argparse
import yaml
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras import mixed_precision
from tensorflow.keras import backend
import gc

from models.iTransformer.new_data_loader import ITransformerData
from models.iTransformer.i_transformer import ITransformer
from models.iTransformer.model_tuner import build_itransformer
from models.tide_google.tide_model import TiDE
from models.tide_google.new_data_loader import TiDEData
from models.tide_google.model_tuner import build_tide
from models.training import CustomModel
from utils.callbacks import get_callbacks
from scrabble import analyze_data
from utils.plot_dist import visualize_overall_data_distribution


def clear_keras_session():
    backend.clear_session()


class ClearSessionTuner(kt.Hyperband):
    def run_trial(self, trial, *args, **kwargs):
        try:
            result = super(ClearSessionTuner, self).run_trial(trial, *args, **kwargs)
        finally:
            clear_keras_session()
        return result


def parse_arguments():
    parser = argparse.ArgumentParser(description='Configure model parameters.')

    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training the model.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs for training.')
    parser.add_argument('--config_file', type=str, default="experiment/dataset_analysis.yaml", help='Path to the YAML configuration file.')
    parser.add_argument('--data_directory', type=str, default="data/sales_forecasting/sales_forecasting_8h", help='Path to data directory')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to be used for experiment')
    parser.add_argument('--model', type=str, default='Baseline', help='Model to be used for experiment')
    parser.add_argument('--tune_hps', type=bool, default=False, help='Tune hyperparameters for dataset')
    parser.add_argument('--normalize', type=bool, default=False, help='Normalize the data')

    args = parser.parse_args()
    return args


def get_model_components(model_name):
    model_components = {
        'iTransformer': (ITransformer, ITransformerData, build_itransformer),
        'TiDE': (TiDE, TiDEData, build_tide),
        'Baseline': (CustomModel, ITransformerData, None)  # Assuming Baseline uses ITransformerData
    }
    return model_components[model_name]


def tune_model_on_dataset(args, hypermodel, data_loader):
    clear_keras_session()
    directory = f"experiment/hyperparameters/{args.dataset}"

    train, val, test = data_loader.get_train_test_splits()
    callbacks = get_callbacks(num_epochs=args.num_epochs, model_name=args.model, dataset_name=args.dataset, mode='tuning')

    tuner = ClearSessionTuner(hypermodel=hypermodel, 
                              objective='val_loss', 
                              max_epochs=args.num_epochs, 
                              factor=3, 
                              hyperband_iterations=1, 
                              directory=directory, 
                              project_name=args.model, 
                              executions_per_trial=2, 
                              max_retries_per_trial=1)

    tuner.search_space_summary()
    tuner.search(train, epochs=args.num_epochs, validation_data=val, callbacks=callbacks, verbose=2)
    summary = tuner.results_summary(3)
    best_hps = tuner.get_best_hyperparameters(3)
    print(f"best hyperparameters for {directory}/{args.model}: {best_hps}")


def train_model_on_dataset(args, model, data_loader):
    clear_keras_session()
    train, val, test = data_loader.get_train_test_splits()
    callbacks = get_callbacks(num_epochs=args.num_epochs, model_name=args.model, dataset_name=args.dataset, mode='training')

    model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate), 
                  loss='mse', 
                  metrics=['mae'],
                  weighted_metrics=[])

    if args.model == 'Baseline':
        model.fit(train, epochs=1, validation_data=val, callbacks=callbacks)
    else:
        model.fit(train, epochs=args.num_epochs, validation_data=val, callbacks=callbacks)

    result = model.evaluate(test, return_dict=True)
    print(f"Training finished with {result} on test data")

    save_results = True
    if save_results:
        update_results(dataset=args.dataset, model=args.model, metrics=result, file_path="experiment/masked_results_one_day_baseline.csv")
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
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, header=[0, 1], index_col=0)
    else:
        df = pd.DataFrame(columns=pd.MultiIndex.from_tuples([], names=["Model", "Metric"]))
        df.index.name = "Dataset"
        
    for metric, value in metrics.items():
        if (model, metric) not in df.columns:
            df[(model, metric)] = None
        df.loc[dataset, (model, metric)] = value
        
    df.to_csv(file_path)
    
      
def init_comps(args):
    model_class, data_loader_class, hypermodel_func = get_model_components(args.model)
    config = analyze_data(file_path = os.path.join(args.data_directory, f"{args.dataset}.csv"),
                          train_split=0.5,
                          val_split=0.1667,
                          test_split=0.1667,
                          lookback_days=14,
                          forecast_days=7,
                          )
    config['batch_size'] = args.batch_size
    config['normalize'] = args.normalize
    
    data_loader_instance = data_loader_class(**config)
    
    if args.tune_hps:
        if hypermodel_func is None:
            raise ValueError("Hyperparameter tuning is not supported for the Baseline model.")
        hypermodel_instance = lambda hp: hypermodel_func(hp=hp, 
                                                         learning_rate=args.learning_rate, 
                                                         seq_len=config['hist_len'], 
                                                         pred_len=config['pred_len'], 
                                                         num_ts=len(config['timeseries_cols']))
        return hypermodel_instance, data_loader_instance
    else:
        model_instance = model_class(seq_len=config['hist_len'], 
                                     pred_len=config['pred_len'], 
                                     num_ts=len(config['timeseries_cols']))
        
        return model_instance, data_loader_instance
    


def main():
    args = parse_arguments()
    print("tune arg is turned to ", args.tune_hps)
    model_instance, data_loader_instance = init_comps(args)
    
    if args.tune_hps:
        tune_model_on_dataset(args, model_instance, data_loader_instance)
    else:
        train_model_on_dataset(args, model_instance, data_loader_instance)
        
    data = data_loader_instance.get_data()    
    print(data)
    
    data_loader_instance.stride = 7
    train, val, test = data_loader_instance.get_train_test_splits()

    to_predict, index = data_loader_instance.get_prediction_set()
    predictions, actuals = model_instance.predict(to_predict)
    
    ### ATTENTION: reshape() still needs to be set to num_targets for now!!
    predictions, actuals = predictions.reshape(-1, 2).tolist(), actuals.reshape(-1, 2).tolist()
    #visualize_overall_data_distribution(actuals)
    
    predictions, actuals = np.asarray(predictions), np.asarray(actuals)
    
    if args.normalize:
        target_transformer = data_loader_instance.get_target_transformer()
        predictions = target_transformer.inverse_transform(predictions)
        actuals = target_transformer.inverse_transform(actuals)
        #visualize_overall_data_distribution(actuals)
    else:
        predictions, actuals = np.asarray(predictions), np.asarray(actuals)
    
    print("shapes:")
    print(predictions.shape)
    print(actuals.shape)
    assert predictions.shape == actuals.shape
    combined = np.hstack((predictions, actuals))
    
    df = pd.DataFrame(combined, index=index)
    df = df.astype(int)
    df.to_csv('experiment/preds_and_actuals.csv')
    
if __name__ == "__main__":
    main()
    


