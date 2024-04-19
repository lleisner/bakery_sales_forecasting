from utils.configs import *
from utils.predict_sample import predict_sample
from utils.plot_hist import plot_training_history
from utils.plot_attention import plot_attention_weights
from utils.loss import SMAPELoss
from utils.plot_preds_and_actuals import plot_time_series

from models.iTransformer.i_transformer import Model
from models.training import CustomModel
from models.iTransformer.data_loader import ITransformerData

from data_provider.data_provider import DataProvider
from data_provider.data_encoder import DataEncoder
from data_provider.data_pipeline import DataPipeline

import warnings
import math
import numpy as np
import matplotlib.pyplot as plt




if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.utils.validation")

    hist_len = 96
    pred_len = 96
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 10
    



    """   

    
    data_loader = ITransformerData(
                data_path = 'ts_datasets/electricity.csv',
                datetime_col='date',
                numerical_cov_cols=None,
                categorical_cov_cols=None,
                cyclic_cov_cols=None,
                timeseries_cols=None,
                train_range=(0, 18317),
                val_range=(18318, 20951),
                test_range=(20952, 26207),
                hist_len=hist_len,
                pred_len=pred_len,
                stride=1,
                sample_rate=1,
                batch_size=batch_size,
                epoch_len=None,
                val_len=None,
                normalize=True,
            )
    data_loader = ITransformerData(
                data_path = 'ts_datasets/traffic.csv',
                datetime_col='date',
                numerical_cov_cols=None,
                categorical_cov_cols=None,
                cyclic_cov_cols=None,
                timeseries_cols=None,
                train_range=(0, 12185),
                val_range=(12186, 13939),
                test_range=(13940, 17447),
                hist_len=hist_len,
                pred_len=pred_len,
                stride=1,
                sample_rate=1,
                batch_size=batch_size,
                epoch_len=None,
                val_len=None,
                normalize=True,
            )
    """
    
    
    data_loader = ITransformerData(
                    data_path = 'ts_datasets/sales_forecasting_8h.csv',
                    datetime_col='date',
                    numerical_cov_cols=None, #['gaestezahlen', 'holidays', 'temperature', 'precipitation', 'cloud_cover', 'wind_speed', 'is_open'],
                    categorical_cov_cols=None,
                    cyclic_cov_cols=None,#['wind_direction'],
                    timeseries_cols=None,#['10', '11', '12', '13', '16', '20', '21', '22', '23', '24', '80', '82', '83', '84', '85', '86'],
                    train_range=(0, 9015),
                    val_range=(9016, 10947),
                    test_range=(10948, 12000),
                    hist_len=hist_len,
                    pred_len=pred_len,
                    stride=1,
                    sample_rate=1,
                    batch_size=batch_size,
                    epoch_len=None,
                    val_len=None,
                    normalize=True,
                    drop_remainder=True,
                )
    """
    
    data_loader = ITransformerData(
                    data_path = 'ts_datasets/ETTh2.csv',
                    datetime_col='date',
                    numerical_cov_cols=None,
                    categorical_cov_cols=None,
                    cyclic_cov_cols=None,
                    timeseries_cols=None, #['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'],
                    train_range=(0, 8544),
                    val_range=(8545, 11425),
                    test_range=(11426, 14307),
                    hist_len=hist_len,
                    pred_len=pred_len,
                    stride=1,
                    sample_rate=1,
                    batch_size=batch_size,
                    epoch_len=None,
                    val_len=None,
                    normalize=True,
                )
    
        data_loader = ITransformerData(
                    data_path = 'data/tide_data_daily.csv',
                    datetime_col='datetime',
                    numerical_cov_cols=None, #['gaestezahlen', 'holidays', 'temperature', 'precipitation', 'cloud_cover', 'wind_speed'],
                    categorical_cov_cols=None,
                    cyclic_cov_cols=None, #['wind_direction'],
                    timeseries_cols=None,
                    train_range=(0, 1103),
                    val_range=(1104, 1339),
                    test_range=(1340, 1577),
                    hist_len=hist_len,
                    pred_len=pred_len,
                    stride=1,
                    batch_size=batch_size,
                    epoch_len=None,
                    val_len=None,
                    normalize=True,
                    drop_remainder=True,
                )
    """
    
    database = data_loader.get_data()
    data_columns = data_loader.get_feature_names_out()
    train, val, test = data_loader.get_train_test_splits()
    print(database)
    print(train)
    print("data_columns: ", data_columns)
    
    num_targets = train.element_spec[1].shape[-1]
    print("num targets: ", num_targets)
    

    dmodel = get_d_model(num_targets)
    
    baseline = CustomModel(seq_len=hist_len,
                           pred_len=pred_len,
                           )
    
    itransformer = Model(seq_len=hist_len,
                         pred_len=pred_len,
                         num_targets=num_targets,
                         d_model=32,
                         n_heads=8,
                         d_ff=128,
                         e_layers=2,
                         dropout=0.2,
                         output_attention=True,
                         
                         )
    
    
    def asymmetric_loss(y_true, y_pred):
        residual = (y_true - y_pred)
        alpha = 1  # Can be tweaked
        l1 = alpha * tf.square(residual) * tf.cast(tf.greater(y_true, y_pred), tf.float32)
        l2 = (1 - alpha) * tf.square(residual) * tf.cast(tf.less_equal(y_true, y_pred), tf.float32)
        return tf.reduce_mean(l1 + l2)



    def advanced_custom_loss(y_true, y_pred, alpha=0.5, beta=0.5, gamma=0.5):
        """
        Custom loss function that incorporates hourly and daily aggregation 
        as well as asymmetric weighting for overestimations and underestimations.

        Parameters:
        - y_true: actual values
        - y_pred: predicted values
        - alpha: weight between 0 and 1 to balance the hourly and daily components (daily aggregation weight)
        - beta: weight between 0 and 1 to balance the overestimation and underestimation (underestimation penalty weight)
        - gamma: overall balance between using MSE or MAE-like behavior to reduce the influence of large errors (optional)

        Returns:
        - combined_loss: the calculated loss
        """
        # Reshape data if it contains features other than one
        if y_true.shape[-1] != 1:
            y_true = tf.reshape(y_true, [y_true.shape[0], -1])
            y_pred = tf.reshape(y_pred, [y_true.shape[0], -1])

        # Define underestimation and overestimation masks
        underestimation = tf.cast(tf.greater(y_true, y_pred), tf.float32)
        overestimation = 1 - underestimation

        # Hourly loss calculations
        hourly_loss = beta * underestimation * tf.square(y_true - y_pred) + (1 - beta) * overestimation * tf.square(y_true - y_pred)
        hourly_mse = tf.reduce_mean(hourly_loss)

        # Daily aggregation
        daily_true = tf.reduce_sum(tf.reshape(y_true, [-1, 24]), axis=1)
        daily_pred = tf.reduce_sum(tf.reshape(y_pred, [-1, 24]), axis=1)
        
        daily_loss = beta * underestimation * tf.square(daily_true - daily_pred) + (1 - beta) * overestimation * tf.square(daily_true - daily_pred)
        daily_mse = tf.reduce_mean(daily_loss)

        # Combined loss with daily and hourly balance
        combined_loss = alpha * daily_mse + (1 - alpha) * hourly_mse
        return combined_loss




    loss = tf.keras.losses.MeanSquaredError()
    #loss = tf.keras.losses.MeanAbsoluteError()
    #loss = tf.keras.losses.Huber(delta=1.0)
    #loss = SMAPELoss()
    loss = asymmetric_loss
    loss = lambda y_true, y_pred: advanced_custom_loss(y_true, y_pred, alpha=0.5, beta=0.7, gamma=0.5)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    
    baseline.compile(optimizer=optimizer, 
                     loss=loss, 
                     weighted_metrics=[])
    
    itransformer.compile(optimizer=optimizer, 
                         loss=loss, 
                         metrics=['mae'],
                         weighted_metrics=[],
                         )
    
    sample = train.take(1)
    out, attns = itransformer.predict(sample)
    
    attention_heads = attns[0][0]
    variate_labels = [item.split("__")[-1] for item in data_columns[:-num_targets]]
    
    #plot_attention_weights(variate_labels, attention_heads)
    
    
    baseline.fit(train, 
                 epochs=1, 
                 validation_data=val, 
                 use_multiprocessing=True)
    baseline.evaluate(test)
    
    hist = itransformer.fit(train, 
                            epochs=num_epochs, 
                            validation_data=val,
                            use_multiprocessing=True)

    itransformer.summary()
    itransformer.evaluate(test)
    
    # take one batch from test data
    sample = test.take(1)
    print("sample:",sample)
    itransformer_preds, attns = itransformer.predict(sample)
    baseline_preds, actuals = baseline.predict(sample)
    
    itransformer_preds = itransformer_preds[0]
    baseline_preds = baseline_preds[0]
    actuals = actuals[0]
    
    print("transformer preds shape:", itransformer_preds.shape)
    
    attention_heads = attns[0][0]

    #plot_attention_weights(variate_labels, attention_heads)
    
    
    #plot_training_history(hist)
    
    plot_time_series(actuals, baseline_preds, itransformer_preds)