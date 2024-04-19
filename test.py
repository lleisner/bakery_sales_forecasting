from utils.configs import get_d_model
from utils.predict_sample import predict_sample
from utils.plot_hist import plot_training_history
from utils.plot_attention import plot_attention_weights
from utils.loss import SMAPELoss

from models.iTransformer.i_transformer import Model
from models.training import CustomModel
from models.iTransformer.data_loader import ITransformerData

from data_provider.data_provider import DataProvider
from data_provider.data_encoder import DataEncoder
from data_provider.data_pipeline import DataPipeline

import warnings
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt





if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.utils.validation")

    hist_len = 96
    pred_len = 96
    batch_size = 32
    learning_rate = 0.0005
    num_epochs = 10
    
    etth1 = ITransformerData(
            data_path = 'ts_datasets/ETTh1.csv',
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

    etth2 = ITransformerData(
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
    
    electricity = ITransformerData(
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
    traffic = ITransformerData(
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
    
    for data_loader in [etth1, etth2, electricity, traffic]:
        
        database = data_loader.get_data()
        data_columns = data_loader.get_feature_names_out()
        train, val, test = data_loader.get_train_test_splits()
        
        num_targets = train.element_spec[1].shape[-1]
        dmodel = get_d_model(num_targets)
        
        itransformer = Model(seq_len=hist_len,
                            pred_len=pred_len,
                            num_targets=num_targets,
                            d_model=dmodel,
                            n_heads=8,
                            d_ff=4*dmodel,
                            e_layers=2,
                            dropout=0.2,
                            output_attention=True,
                            )
        
        loss = tf.keras.losses.MeanSquaredError()
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        itransformer.compile(optimizer=optimizer, 
                            loss=loss, 
                            metrics=['mae'],
                            weighted_metrics=[],
                            )
        
        hist = itransformer.fit(train, 
                                epochs=num_epochs, 
                                validation_data=val,
                                use_multiprocessing=True)

        itransformer.summary()
        itransformer.evaluate(test)
        
        sample = test.take(1)
        out, attns = itransformer.predict(sample)
        
        # Get the attention heads for a sample
        attention_heads = attns[0][0]
        
        # Get the variate names for the attention plots
        variate_labels = [item.split("__")[-1] for item in data_columns[:-num_targets]]


        plot_attention_weights(variate_labels, attention_heads)
        plot_training_history(hist)