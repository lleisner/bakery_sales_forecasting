import os
import tensorflow as tf
import pandas as pd
from tensorflow.keras import backend
import keras_tuner as kt

from models.iTransformer.new_data_loader import ITransformerData
from models.iTransformer.i_transformer import ITransformer
from models.iTransformer.model_tuner import build_itransformer
from models.tide_google.tide_model import TiDE
from models.tide_google.new_data_loader import TiDEData
from models.tide_google.model_tuner import build_tide
from models.training import CustomModel
from utils.callbacks import get_callbacks
from scrabble import analyze_data
from utils.loss import AsymmetricMSELoss


class ClearSessionTuner(kt.Hyperband):
    def run_trial(self, trial, *args, **kwargs):
        try:
            result = super(ClearSessionTuner, self).run_trial(trial, *args, **kwargs)
        finally:
            backend.clear_session()
        return result


def init_comps(model, args):
    model_class, data_loader_class, hypermodel_func = get_model_components(model)
    config = analyze_data(file_path = os.path.join(args.data_directory, f"{args.dataset}.csv"),
                          train_split=0.5,
                          val_split=0.1667,
                          test_split=0.1667,
                          lookback_days=28,
                          forecast_days=7,
                          )
    config['batch_size'] = args.batch_size
    config['normalize'] = args.normalize
    
    data_loader_instance = data_loader_class(**config)
    
    if args.mode == 'tune':
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
    
def load_existing_model(args, model_name, model, data_loader):
    _, _, test = data_loader.get_train_test_splits()
    optimizer = tf.keras.optimizers.Adam(args.learning_rate)
    loss = tf.keras.losses.MeanSquaredError()
    
    model.compile(optimizer=optimizer,
                  loss=loss,
                  weighted_metrics=[],)
    
    callbacks = get_callbacks(num_epochs=args.num_epochs, model_name=model_name, dataset_name=args.dataset, mode='training')
    checkpoint_path = callbacks[1].filepath
    
    if model_name != "Baseline":
        model(data_loader.get_dummy())
        model.load_weights(checkpoint_path)
        print(f"Loading saved model from {checkpoint_path}")
    
    result = model.evaluate(test, return_dict=True)
    print(f"Saved model achieves {result} on test data")
    
    save_results = True
    if save_results:
        update_results(dataset=args.dataset, model=model_name, metrics=result, file_path="experiment/masked_results_one_day_baseline.csv")
    
def train_model_on_dataset(args, model_name, model, data_loader):
    backend.clear_session()
    train, val, test = data_loader.get_train_test_splits()
    callbacks = get_callbacks(num_epochs=args.num_epochs, model_name=model_name, dataset_name=args.dataset, mode='training')
    optimizer = tf.keras.optimizers.Adam(args.learning_rate)
    loss = tf.keras.losses.MeanSquaredError()
    #loss = AsymmetricMSELoss()

    model.compile(optimizer=optimizer, 
                  loss=loss, 
                  weighted_metrics=[])
    

    if model_name == 'Baseline':
        model.fit(train, epochs=1, validation_data=val, callbacks=callbacks)
        result = model.evaluate(test, return_dict=True)
    elif model_name == 'TiDE':
        steps_per_epoch, validation_steps, test_steps = data_loader.split_sizes
        steps_per_epoch = steps_per_epoch // 8
        hist = model.fit(train, epochs=args.num_epochs, steps_per_epoch=steps_per_epoch, validation_data=val, validation_steps=validation_steps, callbacks=callbacks)
        result = model.evaluate(test, steps=test_steps, return_dict=True)
    elif model_name == 'iTransformer':
        hist = model.fit(train, epochs=args.num_epochs, validation_data=val, callbacks=callbacks)
        result = model.evaluate(test, return_dict=True)
        #plot_metrics(hist)
        
        
    print(f"Training finished with {result} on test data")

    save_results = True
    if save_results:
        update_results(dataset=args.dataset, model=model_name, metrics=result, file_path="experiment/masked_results_one_day_baseline.csv")
        
    
    
def tune_model_on_dataset(args, model_name, hypermodel, data_loader):
    backend.clear_session()
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
    print(f"best hyperparameters for {directory}/{model_name}: {best_hps}")
    
    
def get_model_components(model_name):
    model_components = {
        'iTransformer': (ITransformer, ITransformerData, build_itransformer),
        'TiDE': (TiDE, TiDEData, build_tide),
        'Baseline': (CustomModel, ITransformerData, None)  
    }
    return model_components[model_name]


    
def calculate_metrics(y_true, y_pred):
    """
    Calculate and print MSE, MAE, and RMSE metrics.

    Parameters:
    - y_true: The ground truth values.
    - y_pred: The predicted values.
    """
    metrics = {
        'Mean Squared Error (MSE)': tf.keras.metrics.MeanSquaredError(),
        'Mean Absolute Error (MAE)': tf.keras.metrics.MeanAbsoluteError(),
        'Root Mean Squared Error (RMSE)': tf.keras.metrics.RootMeanSquaredError()
    }
    
    results = {}
    for name, metric in metrics.items():
        metric.update_state(y_true, y_pred)
        results[name] = metric.result().numpy()
    
    # Print the results
    for name, result in results.items():
        print(f'{name}: {result}')
        