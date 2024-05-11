from models.iTransformer.data_loader import ITransformerData
import yaml
import argparse

def load_config_from_yaml(filepath):
    with open(filepath, 'r') as file:
        config = yaml.safe_load(file)
        return config

def create_model_config(args):
    # Load configuration from YAML
    with open(args.config_file, 'r') as file:
        data_config = yaml.safe_load(file)
    
    # Merge YAML configurations with command-line arguments
    config = {
        'hist_len': args.hist_len,
        'pred_len': args.pred_len,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_epochs': args.num_epochs,
        'dataset': args.dataset,
    }

    
    dataset_name = config['dataset']
    data_config = data_config[dataset_name]
    
    def calculate_data_ranges(train_size, val_size, test_size):
        return ((0, train_size), 
                (train_size, train_size + val_size), 
                (train_size + val_size, train_size + val_size + test_size))

    train_range, val_range, test_range = calculate_data_ranges(data_config['train_size'], data_config['val_size'], data_config['test_size'])
    
    return {
        "data_path": f"ts_datasets/{data_config['file_name']}",
        "datetime_col": 'date',
        "numerical_cov_cols": None,
        "categorical_cov_cols": None,
        "cyclic_cov_cols": None,
        "timeseries_cols": None, 
        "train_range": train_range,
        "val_range": val_range,
        "test_range": test_range,
        "hist_len": config['hist_len'],
        "pred_len": config['pred_len'],
        "stride": 1,
        "sample_rate": 1,
        "batch_size": config['batch_size'],
        "epoch_len": None,
        "val_len": None,
        "normalize": True,
    }

def parse_arguments():
    parser = argparse.ArgumentParser(description='Configure model parameters.')
    parser.add_argument('--hist_len', type=int, default=96, help='Historical length for the model.')
    parser.add_argument('--pred_len', type=int, default=96, help='Prediction length for the model.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training the model.')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate for the optimizer.')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs for training.')
    parser.add_argument('--config_file', type=str, default='experiment/data_config.yaml', help='Path to the YAML configuration file.')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to be used for experiment')

    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    args = parse_arguments()
    model_config = create_model_config(args)
    transformer_data = ITransformerData(**model_config)
    df = transformer_data.get_data()
    print(df)
    print(df.columns)