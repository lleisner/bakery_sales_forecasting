from models.tide_google.data_loader import TimeSeriesdata
from models.tide_google.tide_model import TiDE
from utils.plot_preds_and_actuals import plot_preds_actuals, plot_df_per_column

import pandas as pd
from models.tide_google.data_loader import TimeSeriesdata as tsd
from models.tide_google.new_data_loader import TiDEData
import tensorflow as tf

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)



fix_gpu()

filename = 'data/sales_forecasting/sales_forecasting_8h.csv'

data = pd.read_csv('data/sales_forecasting/sales_forecasting_8h.csv', index_col=0, parse_dates=True)

#plot_df_per_column(data)

data = data.reset_index()

total_timestamps = data.shape[0]
train_end = int(total_timestamps * 0.7)
val_end = int((total_timestamps - train_end) * 0.5) + train_end



ts_cols = ['10', '20', '16', '24']
numerical_covariates = ['gaestezahlen', 'wind_direction', 'temperature', 'precipitation', 'cloud_cover', 'wind_speed', 'is_open', 'holidays', 'arrival', 'departure']
#categorical_covariates = ['is_open', 'holidays', 'arrival', 'departure']
categorical_covariates = None

pred_len = 224
hist_len = 56
num_ts = len(ts_cols)
batch_size = num_ts
batch_size = min(num_ts, batch_size)

hidden_size = 128
decoder_output_dim = 4
final_decoder_hidden = 64
num_layers = 2

num_epochs = 10
patience = 100
lr = 0.001

epoch_len=train_end-pred_len-hist_len
val_samples=val_end-train_end-pred_len


print("epoch and samples: ", epoch_len, val_samples)
print("what the fuckkerydoo is going on here")

time_series = TiDEData(
    data_path='data/sales_forecasting/sales_forecasting_8h.csv',
    datetime_col='date',
    numerical_cov_cols=numerical_covariates,
    categorical_cov_cols=categorical_covariates,
    cyclic_cov_cols=None,
    timeseries_cols=ts_cols,
    train_range=(0, 6496),
    val_range=(6498, 8516),
    test_range=(8516, 10534),
    hist_len=hist_len,
    pred_len=pred_len,
    batch_size=batch_size,
    freq='h',
    normalize=False, 
    steps_per_epoch=None, 
    validation_steps=None,
    permute=False,
    )

tide_model = TiDE(
    seq_len=None,
    pred_len=pred_len,
    cat_sizes=[],
    num_ts=num_ts,
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

#train_ds = time_series.tf_dataset(mode='train').repeat()
#val_ds = time_series.tf_dataset(mode='val').repeat()
#test_ds = time_series.tf_dataset(mode='test').repeat()

train_ds, val_ds, test_ds = time_series.get_train_test_splits()

train_samples = time_series.steps_per_epoch
print("total train samples:", train_samples)
test_val_samples = time_series.validation_steps
print("total validation samples: ", test_val_samples)

(sample,) = val_ds.take(1)
inputs, y_true = TiDEData.prepare_batch(*sample)



(sample2,) = train_ds.take(1)
inputs2, y_true2 = TiDEData.prepare_batch(*sample2)

#plot_preds_actuals(y_true, y_true)

tide_model.fit(train_ds, 
    validation_data=val_ds, 
    epochs=num_epochs, 
   # steps_per_epoch=train_samples, 
    #validation_steps=test_val_samples, 
    callbacks=callbacks, 
    batch_size=batch_size, 
    use_multiprocessing=True, 
    )

tide_model.evaluate(test_ds, steps=test_val_samples)

predictions = tide_model(inputs)

plot_preds_actuals(predictions, y_true)

print(predictions)
print(y_true)






#training(dtl=time_series, model=tide_model, lr=1e-4, train_epochs=1, sum_dir='logs/experimental')
