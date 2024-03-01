from models.tide_google.data_loader import TimeSeriesdata
from models.tide_google.tide_model import *
from models.tide_google.train import training
import pandas as pd

data = pd.read_csv('data/tide_data.csv', index_col=0)
print(data)

pred_len = 64
hist_len = 512
batch_size = 4
num_ts = 4

hidden_size = 256
decoder_output_dim = 4
final_decoder_hidden = 64
num_layers = 2

time_series = TimeSeriesdata(
    data_path='data/tide_data.csv',
    datetime_col='datetime',
    num_cov_cols=['gaestezahlen', 'wind_direction', 'temperature', 'precipitation', 'cloud_cover', 'wind_speed'],
    cat_cov_cols=['is_open', 'holidays', 'arrival', 'departure'],
    ts_cols=['10', '11', '83', '84'],
    train_range=(0, 20087),
    val_range=(20088, 23435),
    test_range=(23436, 26783),
    hist_len=hist_len,
    pred_len=pred_len,
    batch_size=batch_size,
    freq='h'
    )

model_config = {
    'model_type': 'dnn',
    'hidden_dims': [hidden_size] * num_layers,
    'time_encoder_dims': [64, 4],
    'decoder_output_dim': decoder_output_dim,
    'final_decoder_hidden': final_decoder_hidden,
    'batch_size': time_series.batch_size
}

tide_model = TideModel(
    model_config=model_config,
    pred_len=pred_len,
    num_ts=num_ts,
    cat_sizes=time_series.cat_sizes,
    transform=False,
    layer_norm=False,
    dropout_rate=0.0
)

training(dtl=time_series, model=tide_model, lr=1e-4, train_epochs=100, sum_dir='logs/experimental')
