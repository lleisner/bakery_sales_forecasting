from utils.configs import *
from utils.predict_sample import predict_sample
from utils.plot_hist import plot_training_history
from utils.plot_attention import plot_attention_weights

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

    hist_len = 112
    pred_len = 56
    batch_size = 32
    learning_rate = 0.0001
    num_epochs = 15
    

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
                    data_path = 'data/tide_data.csv',
                    datetime_col='datetime',
                    numerical_cov_cols=['gaestezahlen', 'holidays', 'temperature', 'precipitation', 'cloud_cover', 'wind_speed'],
                    categorical_cov_cols=None,
                    cyclic_cov_cols=None, #['wind_direction'],
                    timeseries_cols=None,
                    train_range=(0, 9015),
                    val_range=(9016, 10947),
                    test_range=(10948, 12623),
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
    
    # Calculate optimal model size as described in TimesNet  
    C = num_targets
    dmin = 64
    dmax = 512
    opt = 2 ** math.ceil(math.log2(C))
    print("optimal d_model: ", opt)

    dmodel = min(max(opt, dmin), dmax)
    
    baseline = CustomModel(seq_len=hist_len,
                           pred_len=pred_len,
                           )
    
    itransformer = Model(seq_len=hist_len,
                         pred_len=pred_len,
                         d_model=dmodel,
                         n_heads=8,
                         d_ff=4*dmodel,
                         e_layers=2,
                         dropout=0.2,
                         output_attention=True,
                         
                         )
    
    loss = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    
    baseline.compile(optimizer=optimizer, 
                     loss=loss, 
                     weighted_metrics=[])
    
    itransformer.compile(optimizer=optimizer, 
                         loss=loss, 
                         metrics=['mae'],
                         weighted_metrics=[])
    
    sample = train.take(1)
    out, attns = itransformer.predict(sample)
    
    attention_heads = attns[0][0]
    variate_labels = [item.split("__")[-1] for item in data_columns[:-num_targets]]

   # plot_attention_weights(variate_labels, attention_heads)
    
    
    baseline.fit(train, 
                 epochs=1, 
                 validation_data=val, 
                 use_multiprocessing=True)
    
    hist = itransformer.fit(train, 
                            epochs=num_epochs, 
                            validation_data=val,
                            use_multiprocessing=True)

    itransformer.summary()
    itransformer.evaluate(test)
    
    sample = test.take(1)
    out, attns = itransformer.predict(sample)
    
    attention_heads = attns[0][0]

    plot_attention_weights(variate_labels, attention_heads)
    
    
    plot_training_history(hist)