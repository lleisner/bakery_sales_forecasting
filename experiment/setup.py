from models.iTransformer.data_loader import ITransformerData

def initialize_model(hist_len, pred_len, num_targets, learning_rate):
    pass

def create_transformer_data(config):
    def calculate_data_ranges(train_size, val_size, test_size):
        return ((0, train_size), 
                (train_size, train_size + val_size), 
                (train_size + val_size, train_size + val_size + test_size))

    train_range, val_range, test_range = calculate_data_ranges(config['train_size'], config['val_size'], config['test_size'])
    
    return ITransformerData(
        data_path = f'ts_datasets/{config['file_name']}',
        datetime_col = 'date',
        numerical_cov_cols=None,
        categorical_cov_cols=None,
        cyclic_cov_cols=None,
        timeseries_cols=None, 
        train_range=train_range,
        val_range=val_range,
        test_range=test_range,
        hist_len=config['hist_len'],
        pred_len=config['pred_len'],
        stride=1,
        sample_rate=1,
        batch_size=config['batch_size'],
        epoch_len=None,
        val_len=None,
        normalize=True,
    )