from models.tide_google.data_loader import TimeSeriesdata
from models.tide_google.tide_model import *
from models.tide_google.train import training
from utils.plot_preds_and_actuals import plot_preds_actuals

import pandas as pd
from models.tide_google.data_loader import TimeSeriesdata as tsd
import tensorflow as tf

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)



fix_gpu()

data = pd.read_csv('data/tide_data.csv', index_col=0)
print(data)

pred_len = 64
hist_len = 512
num_ts = 4
batch_size = 4
batch_size = min(num_ts, batch_size)

hidden_size = 256
decoder_output_dim = 16
final_decoder_hidden = 64
num_layers = 1

num_epochs = 5
patience = 20
lr = 1e-4

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
    freq='h',
    normalize=False, 
    epoch_len=19511, 
    val_samples=3284
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
    transform=True,
    layer_norm=True,
    dropout_rate=0.3
)

lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=lr,
    decay_steps=30 * time_series.train_range[1]
)
#optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipvalue=1e3)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)


tide_model.compile(optimizer=optimizer, 
                   loss=tf.keras.losses.MeanSquaredError(),
                   metrics=["mae"])

callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)]

train_ds = time_series.tf_dataset(mode='train').repeat()
val_ds = time_series.tf_dataset(mode='val').repeat()
test_ds = time_series.tf_dataset(mode='test').repeat()

train_samples = time_series.epoch_len
print("total train samples:", train_samples)
test_val_samples = time_series.val_samples
print("total validation samples: ", test_val_samples)

(sample,) = test_ds.take(1)
inputs, y_true = tsd.prepare_batch(*sample)



(sample2,) = train_ds.take(1)
inputs2, y_true2 = tsd.prepare_batch(*sample2)

plot_preds_actuals(y_true, y_true)

tide_model.fit(train_ds, validation_data=val_ds, epochs=num_epochs, steps_per_epoch=train_samples, validation_steps=test_val_samples, callbacks=callbacks, batch_size=batch_size)
tide_model.evaluate(test_ds, steps=test_val_samples)

predictions = tide_model(inputs)

plot_preds_actuals(predictions, y_true)







#training(dtl=time_series, model=tide_model, lr=1e-4, train_epochs=1, sum_dir='logs/experimental')
