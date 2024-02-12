import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import keras_tuner
import os
from tensorboard.plugins.hparams import api as hp

from utils.loss import CustomLoss, CombinedLossWithDynamicWeights
from utils.plot_hist import plot_training_history
from utils.visual_season import visualize_seasonality
from utils.combined_hist import CombinedHistory
from utils.configs import Settings, ProviderConfigs, ProcessorConfigs, PipelineConfigs, TransformerConfigs, build_model
from models.lstm_model.lstm import CustomLSTM
#from utils.plot_attention import plot_attention_heatmap

from models.iTransformer.i_transformer import Model
from models.iTransformer.fine_tuning import FineTuner
from models.conv_dense_hybrid.hybrid_model import HybridModel
from models.training import CustomModel

from data_provider.data_provider import DataProvider
from data_provider.data_encoder import DataEncoder
from data_provider.data_pipeline import DataPipeline
from data_provider.new_data_pipeline import NewDataPipeline
from data_provider.time_configs import TimeConfigs
from tensorflow.keras.callbacks import Callback, EarlyStopping, TensorBoard, ModelCheckpoint

import warnings


if __name__ == "__main__":
    
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.utils.validation")

    settings = Settings()
    provider_configs = ProviderConfigs()
    processor_configs = ProcessorConfigs(settings)

    provider = DataProvider(provider_configs)
    encoder = DataEncoder(configs = processor_configs)
    
    #provider.create_new_database(provider_list=['sales'])
        
    df = provider.load_database()
        
    #processor = DataProcessor(data=df, configs=processor_configs)
    encoder = DataEncoder(configs = processor_configs)
    
    #encoding = processor.fit_and_encode()
    encoding = encoder.process_data(df)
        

        
    #encoding = processor.get_uncoded_data()
    
    print("dataset features: ", encoding.columns)
    #num_features, num_targets = processor.get_shape()
    num_features, num_targets = encoder.get_feature_target_nums(encoding)
    
    pipeline_configs = PipelineConfigs(settings)#, num_features, num_targets)
    t_configs = TransformerConfigs(settings, num_features, num_targets)

    to_predict = encoding.tail(settings.seq_length)
    # Set train cutoff to two months ago
    dataset = encoding[:-512]

    steps_per_epoch, validation_steps, test_steps = settings.calculate_steps(dataset.shape[0])
    

    pipeline = DataPipeline(pipeline_configs)
    new_pipeline = NewDataPipeline(pipeline_configs)
    
    train, val, test = pipeline.generate_train_test_splits(dataset)
    
    

    
    log_dir = "logs/tmp/tb_logs/"
    checkpoint_path = 'saved_models/i_transformer_weights/checkpoint.ckpt'

    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',  # Monitor loss
        patience=settings.early_stopping_patience,         # Number of epochs with no improvement to wait
        restore_best_weights=True  # Restore the best weights when stopped
    )
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    search_best_model = False
    if search_best_model:
        tuner = keras_tuner.RandomSearch(
        hypermodel=lambda hp: build_model(hp, model=Model, configs=t_configs),
        objective="val_loss",
        max_trials=25,
        executions_per_trial=1,
        overwrite=True,
        directory="logs/tuner",
        project_name="iTransformer",
        )
        tuner.search_space_summary()
        tuner.search(train, epochs=settings.num_epochs, steps_per_epoch=steps_per_epoch, validation_data=val, validation_steps=validation_steps, callbacks=[early_stopping, tensorboard_callback, checkpoint], use_multiprocessing=True)

        tuner.results_summary()
        best_hps = tuner.get_best_hyperparameters(5)
        print("best hyperparameters: ", best_hps)
    print(
    f"model specifications:\n"
    f"sequence_length: {settings.seq_length}\n"
    f"future_steps: {settings.future_steps}\n"
    f"window_strides: {settings.strides}\n"
    f"timeframe: {provider_configs.start_date} to {provider_configs.end_date}\n"
    f"item_selection: {provider_configs.item_selection}\n"
    f"covariate_selection: {processor_configs.covariate_selection}\n"
    f"loss_function: {t_configs.loss}"
    )
    


   # configs = Configurator()
    #model = HybridModel(t_configs)
    model = Model(t_configs)
    #model = CustomLSTM(t_configs)
    baseline = CustomModel(t_configs)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=t_configs.learning_rate), loss=t_configs.loss, weighted_metrics=[])
    baseline.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=t_configs.learning_rate), loss=t_configs.loss, weighted_metrics=[])



    log_dir = "logs/fit/"
    checkpoint_path = 'saved_models/i_transformer_weights/checkpoint.ckpt'

    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',  # Monitor loss
        patience=settings.early_stopping_patience,         # Number of epochs with no improvement to wait
        restore_best_weights=True  # Restore the best weights when stopped
    )
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    #model.load_weights(checkpoint_path)
    print("baseline:\n")
    baseline.fit(train, epochs=1, steps_per_epoch=steps_per_epoch, validation_data=val, validation_steps=validation_steps)
    print("model training:\n")
    hist = model.fit(train, epochs=settings.num_epochs, steps_per_epoch=steps_per_epoch, validation_data=val, validation_steps=validation_steps, verbose=1, callbacks=[early_stopping, tensorboard_callback, checkpoint], use_multiprocessing=True)

    combined_history = CombinedHistory()
    train_on_cross_validation_data = False

    if train_on_cross_validation_data:
        
        train_val_splits = new_pipeline.generate_train_val_splits(dataset)
        test_split = new_pipeline.get_test_set(dataset)
        
        for train_size, train_ds, val_size, val_ds in train_val_splits:
            steps_per_epoch, val_steps = train_size // settings.batch_size, val_size // settings.batch_size
            history = model.fit(train_ds, epochs=settings.num_epochs, steps_per_epoch=steps_per_epoch, validation_data=val_ds, validation_steps=val_steps, callbacks=[early_stopping, tensorboard_callback, checkpoint], use_multiprocessing=True)
                # Update the custom history object with the metrics from the current training step
            combined_history.history['loss'].extend(history.history['loss'])
            combined_history.history['val_loss'].extend(history.history['val_loss'])
        plot_training_history(combined_history)
            
    model.summary()

    model.evaluate(test, steps=test_steps)
    baseline.evaluate(test, steps=test_steps)

    
    for layer in model.layers:
        #print(f"found layer {layer}")
        layer.trainable=False

    fine_tuning_model = FineTuner(t_configs, model)
    fine_tuning_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=t_configs.learning_rate), loss=t_configs.loss, weighted_metrics=[])

    
    fine_tuning_hist = fine_tuning_model.fit(train, epochs=settings.num_epochs, steps_per_epoch=steps_per_epoch, validation_data=val, validation_steps=validation_steps, callbacks=[early_stopping, tensorboard_callback, checkpoint], use_multiprocessing=True)
    
    fine_tuning_model.summary()
    
    fine_tuning_model.evaluate(test, steps=test_steps)
    
    plot_training_history(hist)
    plot_training_history(fine_tuning_hist)

    index = to_predict.index[-settings.future_steps:]
    
    X = to_predict.iloc[:, :num_features]
    y = to_predict.iloc[-settings.future_steps:, -num_targets:]

    x_s , x_mark = X.iloc[:, :num_targets].values, X.iloc[:, num_targets:].values
    x_s, x_mark = np.expand_dims(x_s, axis=0), np.expand_dims(x_mark, axis=0)
    x_s, x_mark = tf.convert_to_tensor(x_s), tf.convert_to_tensor(x_mark)

    prediction = model.predict((x_s, x_mark), batch_size=1)
    #fine_prediction = fine_tuning_model.predict((x_s, x_mark), batch_size=1)
    
    prediction = pd.DataFrame(np.squeeze(prediction), index=index)
    print(y.columns)
    prediction = encoder.decode_data(prediction, col_names=y.columns)
    true_y = encoder.decode_data(y, col_names=y.columns)
    print(prediction)
    
    pred_sum = prediction.resample('D').sum()
    true_sum = true_y.resample('D').sum()
    print(pred_sum)
    print(true_sum)
    new_df = true_y.apply(lambda x: x.astype(str) + ' | ' + prediction[x.name].astype(str))
    sum_df = true_sum.apply(lambda x: x.astype(str) + ' | ' + pred_sum[x.name].astype(str))
    print(new_df)   
    print(sum_df) 
    
    prediction.to_csv('data/prediction.csv')
    
    


    
