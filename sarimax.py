from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

from utils.configs import *
from data_provider.data_provider import DataProvider
from data_provider.data_encoder import DataEncoder
from data_provider.data_pipeline import DataPipeline

import warnings
import numpy as np
import matplotlib.pyplot as plt

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
    
    y = encoding['10']
    X = encoding.drop(columns=['10'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = SARIMAX(y_train, 
                    exog=X_train,
                    order=(1,1,1),
                    seasonal_order=(1,1,1,112),
                    enforce_stationarity=False,
                    enfore_invertibility=False)

    model_fit = model.fit(disp=False)
    predictions = model_fit.forecast(steps=len(y_test), exog=X_test)
    
    mae = mean_absolute_error(y_test, predictions)
    mse=mean_squared_error(y_test, predictions)
    print(f'mae: {mae}, mse: {mse}')