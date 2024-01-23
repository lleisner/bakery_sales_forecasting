import pandas as pd
import joblib
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from collections import OrderedDict
from models.autoencoder.training import build_autoencoder, train_model
from data_provider.data_provider import DataProvider
from utils.configs import ProviderConfigs, Settings, ProcessorConfigs

class BaseEncoder:
    def __init__(self, encoder_model="standard", encoder_filename: str=None):
        self.encoder_filename = encoder_filename
        self.encoder = MinMaxScaler() if encoder_model=="min_max" else StandardScaler()
        
    def fit_encoder(self, data: pd.DataFrame):
        if self.encoder_filename:
            self.encoder.fit(data)
            joblib.dump(self.encoder, self.encoder_filename)
 
    def load_encoder(self):
        if self.encoder_filename:
            try:
                self.encoder = joblib.load(self.encoder_filename)
            except FileNotFoundError as e:
                raise Exception("Error while loading encoder. Try running BaseEncoder.fit_encoder() or BaseEncoder.fit_and_encode() first to create a new encoder") from e

    def encode(self, data: pd.DataFrame) -> pd.DataFrame:
        self.load_encoder()
        return self._encode_data(data)

    def fit_and_encode(self, data: pd.DataFrame) -> pd.DataFrame:
        self.fit_encoder(data)
        return self._encode_data(data)
    
    def _encode_data(self, data):
        return data
    
    def get_uncoded_data(self, data):
        return data
    
class TimeFeatureEncoder:
    @staticmethod
    def create_encoding(data: pd.Series, name: str, parameter_range: int) -> pd.DataFrame:
        feature = data / parameter_range - 0.5
        encoding_df = pd.DataFrame({name: feature})
        return encoding_df

class SineCosineEncoder:
    @staticmethod
    def create_encoding(data: pd.Series, name: str, parameter_range: int) -> pd.DataFrame:
        """
        Create sine and cosine encodings for a column of cyclic data within the range of -0.5 to 0.5.

        Args:
            data (pd.Series): The DataFrame column containing the cyclic data.
            name (str): Name for the encoding columns.
            parameter_range (int): The range of the cyclic data (e.g. 12 for months).

        Returns:
            pd.DataFrame: A DataFrame containing two columns: 'sine_encoding' and 'cosine_encoding' scaled to the range [-0.5, 0,5].
        """
        # Convert the column to radians
        radians = (data / parameter_range) * 2 * np.pi

        # Calculate sine and cosine encodings
        sine_encoding = 0.5 * (np.sin(radians) + 1) - 0.5
        cosine_encoding = 0.5 * (np.cos(radians) + 1) - 0.5

        # Create a DataFrame with the encodings
        encoding_df = pd.DataFrame({f'{name}_sine_encoding': sine_encoding, f'{name}_cosine_encoding': cosine_encoding})

        return encoding_df
        

class TemporalEncoder(BaseEncoder):
    def __init__(self, encoder_model="standard"):
        self.encoder = SineCosineEncoder() if encoder_model=="sine_cosine" else TimeFeatureEncoder()
        self.encoder_filename = None
    
    def _encode_data(self, data: pd.DatetimeIndex):
        dataframes = []
      #  dataframes.append(pd.get_dummies(data.year, prefix='year', dtype=float))
        dataframes.append(self.encoder.create_encoding(data.hour, 'hour', 24))
        dataframes.append(self.encoder.create_encoding(data.dayofweek, 'dayofweek', 7))
        dataframes.append(self.encoder.create_encoding((data.day - 1), 'dayofmonth', 30))
        dataframes.append(self.encoder.create_encoding((data.dayofyear - 1), 'dayofyear', 364))
      #  dataframes.append(self.encoder.create_encoding((data.isocalendar().week -1), 'week', 51))
        dataframes.append(self.encoder.create_encoding((data.month - 1), 'month', 11))
        dataframes.append(self.encoder.create_encoding((data.quarter - 1), 'quarter', 3))

        df = pd.concat(dataframes, axis=1).set_index(data)
        return df
    

class WeatherEncoder(BaseEncoder):
    def __init__(self, encoder_model, encoder_filename: str='saved_models/weather_encoding.save'):
        super().__init__(encoder_model=encoder_model, encoder_filename=encoder_filename)
    
    def _encode_data(self, data: pd.DataFrame) -> pd.DataFrame:
        sin_cos_coding = SineCosineEncoder.create_encoding(data['wind_direction'], 'wind_direction', 360).set_index(data.index)
        encoding = pd.DataFrame(self.encoder.transform(data), index=data.index, columns=data.columns)
        encoding = pd.concat([sin_cos_coding, encoding], axis=1).drop(['wind_direction'], axis=1)
        return encoding

class SalesEncoder(BaseEncoder):
    def __init__(self, encoder_model, encoder_filename: str='saved_models/sales_encoding.save'):
        super().__init__(encoder_model=encoder_model, encoder_filename=encoder_filename)

    def _encode_data(self, data: pd.DataFrame) -> pd.DataFrame:
        encoding = self.encoder.transform(data)
        encoding = pd.DataFrame(encoding, index=data.index, columns=data.columns)
        return encoding
    

class LabelEncoder(SalesEncoder):
    def __init__(self, encoder_model, encoder_filename: str='saved_models/label_encoding.save'):
        super().__init__(encoder_model=encoder_model, encoder_filename=encoder_filename)

    def decode_data(self, data: pd.DataFrame):
        self.load_encoder()
        return self.encoder.inverse_transform(data)


class AutoEncoder(BaseEncoder):
    def __init__(self, encoder_filename: str='saved_models/autoencoder.h5'):
        super().__init__(encoder_model=None, encoder_filename=encoder_filename)
        

    def fit_encoder(self, data: pd.DataFrame) -> pd.DataFrame:
        self.encoder = build_autoencoder(data.shape[1])
        train_model(data, self.encoder, self.encoder_filename)
        return self.encoder.get_layer('encoder')

    def load_encoder(self) -> tf.keras.models.Model:
        try:
            if self.encoder_filename:
                return tf.keras.models.load_model(self.encoder_filename).get_layer('encoder')
        except FileNotFoundError as e:
            raise Exception("Error while loading encoder. Try running BaseEncoder.fit_encoder() or BaseEncoder.fit_and_encode() first to create a new encoder") from e
    
    def _encode_data(self, data: pd.DataFrame) -> pd.DataFrame:
        encoding = self.encoder.predict(data)
        columns = [f"{self.encoder_filename.split('/')[-1].split('.h5')[0]}_{i}" for i in range(encoding.shape[1])]
        encoding =  pd.DataFrame(encoding, index=data.index, columns=columns)
        return encoding
    

class FerienEncoder(AutoEncoder):
    def __init__(self, encoder_filename: str='saved_models/ferien_encoding.h5'):
        super().__init__(encoder_filename=encoder_filename)


class FahrtenEncoder(AutoEncoder):
    def __init__(self, encoder_filename: str='saved_models/fahrten_encoding.h5'):
        super().__init__(encoder_filename=encoder_filename)


class DataProcessor:
    def __init__(self, data: pd.DataFrame, configs):
        self.configs = configs
        self.data = {
            'datetime': data.index,
            'weather': data[['temperature', 'precipitation', 'cloud_cover', 'wind_speed', 'wind_direction']],
            'ferien': data[['BW', 'BY', 'BE', 'BB', 'HB', 'HH', 'HE', 'MV', 'NI', 'NW', 'RP', 'SL', 'SN', 'ST', 'SH', 'TH']],
            'fahrten': data[['SP1_an', 'SP2_an', 'SP4_an', 'SP1_ab', 'SP2_ab', 'SP4_ab']],
            'is_open': data[['is_open']],
            'labels': data[[col for col in data.columns if str(col).isnumeric()]]
        }
        self._create_sales_features()

        self.encoders = {
            'datetime': TemporalEncoder(encoder_model=self.configs.temp_encoder),
            'weather': WeatherEncoder(encoder_model=self.configs.def_encoder),
            'ferien': FerienEncoder() if self.configs.reduce_one_hots else BaseEncoder(encoder_model=self.configs.def_encoder),
            'fahrten': FahrtenEncoder() if self.configs.reduce_one_hots else BaseEncoder(encoder_model=self.configs.def_encoder),
            'is_open': BaseEncoder(encoder_model=self.configs.def_encoder),
            'labels': LabelEncoder(encoder_model=self.configs.def_encoder),
            'sales': SalesEncoder(encoder_model=self.configs.def_encoder)
        }
        selected_keys = self.configs.covariate_selection
        selected_keys.insert(0, "sales")
        selected_keys.append("labels")
        self.data = OrderedDict((key, self.data[key]) for key in selected_keys)
        self.shape = None
        
    def _create_sales_features(self) -> None:
        sales_feature = self.data['labels'].copy().add_prefix(f'(t-{self.configs.future_days})')
        sales_feature.index = sales_feature.index + pd.Timedelta(days=self.configs.future_days)
        self.data['sales'] = sales_feature

    def fit_and_encode(self) -> pd.DataFrame:
        return self._encode_data(fit_encoder=True)
    
    def encode(self) -> pd.DataFrame:
        return self._encode_data(fit_encoder=False)
    
    def _encode_data(self, fit_encoder: bool) -> pd.DataFrame:
        encoded_data = {}
        for key, data in self.data.items():
            encoder = self.encoders[key]
            encoded_data[key] = encoder.fit_and_encode(data) if fit_encoder else encoder.encode(data)
   
        result = pd.concat(encoded_data.values(), axis=1).dropna()
        num_targets = len(self.data['labels'].columns)
        num_features = result.shape[1] - num_targets
        self.shape = num_features, num_targets
        return result

    def get_uncoded_data(self):
        uncoded_data = {}
        for key, data in self.data.items():
            encoder = self.encoders[key]
            if key == "datetime":
                uncoded_data[key] = encoder.encode(data)
            else:
                uncoded_data[key] = encoder.get_uncoded_data(data)
        result = pd.concat(uncoded_data.values(), axis=1).dropna()
        num_targets = len(self.data['labels'].columns)
        num_features = result.shape[1] - num_targets
        self.shape = num_features, num_targets
        return result
    
    
    def decode_data(self, data: pd.DataFrame):
        decoder = self.encoders['labels']
        return pd.DataFrame(decoder.decode_data(data), columns=self.data['labels'].columns, index=data.index).astype(int)
    
    def get_shape(self):
        # shape = num_features, num_targets
        if self.shape is None:
            raise ValueError('shape of encoding is None, run .encode() first to determine the shape of the data')
        return self.shape
        

if __name__ == "__main__":
    settings = Settings()
    configs = ProviderConfigs()
    provider = DataProvider(configs)
    provider.create_new_database(provider_list=['sales'])
    df = provider.load_database()
    p_configs = ProcessorConfigs(settings)
    processor = DataProcessor(data=df, configs=p_configs)
    uncoded = processor.get_uncoded_data()
    print(uncoded)
    
    encoding = processor.fit_and_encode()
    print(encoding)

