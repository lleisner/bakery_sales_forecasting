

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
import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras import mixed_precision

from tensorflow.keras import backend
import gc
from numba import cuda

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
#mixed_precision.set_global_policy('mixed_float16')



def parse_arguments(yaml_filepath):
    parser = argparse.ArgumentParser(description='Configure model parameters.')

    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training the model.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs for training.')
    parser.add_argument('--config_file', type=str, default=yaml_filepath, help='Path to the YAML configuration file.')
    parser.add_argument('--data_directory', type=str, default="data/sales_forecasting/sales_forecasting_8h", help='Path to data directory')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to be used for experiment')

    args = parser.parse_args()
    return args

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


def save_hyperparameters_to_yaml(hyperparameter_summary, dataset, filepath):
    data = {
        'dataset': hyperparameter_summary
    }
    with open(filepath, 'w') as file:
        yaml.dump(data, file)



def tune_model_on_dataset(name, model_builder, data_loader):

    clear_keras_session()
    #gc.collect()
    #free_gpu_mem()
    
    yaml_filepath = "experiment/dataset_analysis.yaml"
    args = parse_arguments(yaml_filepath=yaml_filepath)
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
                         factor=5,
                         hyperband_iterations=3,
                         directory=directory,
                         project_name=name,
                         executions_per_trial=3,
                         max_retries_per_trial=1,
                         )


    tuner.search_space_summary()
    #tuner.search(train, epochs=args.num_epochs, validation_data=val, callbacks=callbacks, verbose=2)
    tuner.reload()
    tuner.results_summary(3)
    best_hps = tuner.get_best_hyperparameters(3)
    print(f"best hyperparameters for {directory}/{name}: {best_hps}")

    
def train_model_on_dataset(name, model, data_loader):
    yaml_filepath = "experiment/dataset_analysis.yaml"
    args = parse_arguments(yaml_filepath=yaml_filepath)
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
    
    model.evaluate(test)
    model.summary()
    
if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    print("Is GPU available:", tf.test.is_gpu_available())
    print("Built with CUDA:", tf.test.is_built_with_cuda())

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    print("GPUs:", gpus)

    #train_model_on_dataset("TiDE", TiDE, TiDEData)
    #train_model_on_dataset("iTransformer", ITransformer, ITransformerData)
    #tune_model_on_dataset("TiDE", build_tide, TiDEData)
    tune_model_on_dataset("iTransformer", build_itransformer, ITransformerData)
    
    
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