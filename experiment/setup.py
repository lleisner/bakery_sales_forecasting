from models.iTransformer.data_loader import ITransformerData
from models.iTransformer.i_transformer import Model
from models.training import CustomModel
from utils.analyze_data import analyze_all_datasets
from utils.plot_attention import plot_attention_weights
from utils.plot_preds_and_actuals import plot_time_series
import yaml
import argparse
import tensorflow as tf

def load_config_from_yaml(filepath):
    with open(filepath, 'r') as file:
        config = yaml.safe_load(file)
        return config

def create_model_config(args):
    def calculate_data_ranges(train_size, val_size, test_size):
        return ((0, train_size), 
                (train_size, train_size + val_size), 
                (train_size + val_size, train_size + val_size + test_size))

    # Load configuration from YAML
    with open(args.config_file, 'r') as file:
        data_config = yaml.safe_load(file)

    
    dataset_name = args.dataset
    data_config = data_config[dataset_name]
    
    train_range, val_range, test_range = calculate_data_ranges(data_config['train_size'], data_config['val_size'], data_config['test_size'])
    
    return {
        "data_path": f"data/sales_forecasting/{data_config['file_name']}",
        "datetime_col": 'date',
        "numerical_cov_cols": data_config['cov_cols'],
        "categorical_cov_cols": None,
        "cyclic_cov_cols": None,
        "timeseries_cols": data_config['ts_cols'], 
        "train_range": train_range,
        "val_range": val_range,
        "test_range": test_range,
        "hist_len": data_config['suggested_window'],
        "pred_len": data_config['suggested_forecast'],
        "stride": 1,
        "sample_rate": 1,
        "batch_size": args.batch_size,
        "epoch_len": None,
        "val_len": None,
        "normalize": False,
    }, {"seq_len": data_config['suggested_window'],
        "pred_len": data_config['suggested_forecast'],
        "num_targets": data_config['ts_dim'],
        "d_model": 32,
        "n_heads": 8,
        "d_ff": 128,
        "e_layers": 2,
        "dropout": 0.2,
        "output_attention": True,
     }

def parse_arguments(yaml_filepath):
    parser = argparse.ArgumentParser(description='Configure model parameters.')

    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training the model.')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate for the optimizer.')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs for training.')
    parser.add_argument('--config_file', type=str, default=yaml_filepath, help='Path to the YAML configuration file.')
    parser.add_argument('--data_directory', type=str, default="data/sales_forecasting", help='Path to data directory')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to be used for experiment')

    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    yaml_filepath = "experiment/dataset_analysis.yaml"
    args = parse_arguments(yaml_filepath)
    
    analyze_all_datasets(args.data_directory, infer_ts_cols=True, yaml_output_path=yaml_filepath)
    
    loader_config, model_config = create_model_config(args)
    transformer_data = ITransformerData(**loader_config)
    df = transformer_data.get_data()
    print(df)
    print(df.columns)
    
    train, val, test = transformer_data.get_train_test_splits()
    print(transformer_data.get_feature_names_out())
    
    baseline = CustomModel(seq_len=loader_config['hist_len'], 
                           pred_len=loader_config['pred_len'])
    
    itransformer = Model(**model_config)
    
    loss = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    
    baseline.compile(optimizer=optimizer, loss=loss, weighted_metrics=[])
    
    itransformer.compile(optimizer=optimizer, loss=loss, metrics=['mae'],weighted_metrics=[])
    
    baseline.fit(train, epochs=1, validation_data=val)
    
    hist = itransformer.fit(train, epochs=args.num_epochs,validation_data=val)
    
    print(f"baseline evaluation on test data: {baseline.evaluate(test)}")
    print(f"itransformer evaluation on test data: {itransformer.evaluate(test)}")
    
    itransformer.summary()
    
    sample = test.take(1)
    itransformer_preds, attns = itransformer.predict(sample)
    baseline_preds, actuals = baseline.predict(sample)
    
    labels = transformer_data.get_feature_names_out()[:-model_config['num_targets']]
    
    t_preds = itransformer_preds[0]
    b_preds = baseline_preds[0]
    actuals = actuals[0]
    attn_heads = attns[0][0]
    
    
    plot_attention_weights(labels, attn_heads)
    plot_time_series(actuals, b_preds, t_preds)