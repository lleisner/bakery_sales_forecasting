from utils.configs import *
from utils.plot_hist import plot_training_history
from models.iTransformer.i_transformer import Model
from models.training import CustomModel

from data_provider.data_provider import DataProvider
from data_provider.data_encoder import DataEncoder
from data_provider.data_pipeline import DataPipeline

import warnings

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.utils.validation")
    
    settings = Settings()
    provider_configs = ProviderConfigs()
    encoder_configs = ProcessorConfigs(settings=settings)
    pipeline_configs = PipelineConfigs(settings=settings)
    
    provider = DataProvider(configs=provider_configs)
    encoder = DataEncoder(configs=encoder_configs)
    pipeline = DataPipeline(configs=pipeline_configs)
    
    database = provider.load_database()
    encoding = encoder.process_data(database)
    train, val, test = pipeline.generate_train_test_splits(encoding)
    
    print(database)
    print(encoding)
    print(train)
    
    num_targets = train.element_spec[1].shape[-1]
    model_configs = TransformerConfigs(settings=settings, num_features=0, num_targets=num_targets)
    
    steps_per_epoch, validation_steps, test_steps = settings.calculate_steps(encoding.shape[0])

    baseline = CustomModel(model_configs)
    itransformer = Model(model_configs)
    
    baseline.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=model_configs.learning_rate), loss=model_configs.loss, weighted_metrics=[])
    itransformer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=model_configs.learning_rate), loss=model_configs.loss, weighted_metrics=[])

    baseline.fit(train, epochs=1, steps_per_epoch=steps_per_epoch, validation_data=val, validation_steps=validation_steps)
    hist = itransformer.fit(train, epochs=settings.num_epochs, steps_per_epoch=steps_per_epoch, validation_data=val, validation_steps=validation_steps, verbose=1, use_multiprocessing=True)

    itransformer.summary()
    itransformer.evaluate(test, steps=test_steps)
    
    plot_training_history(hist)