import pandas as pd
import joblib
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from collections import OrderedDict
from models.autoencoder.training import build_autoencoder, train_model
from data_provider.data_provider import DataProvider
from utils.configs import ProviderConfigs, Settings

class BaseEncoder:
    def __init__(self, encoder_filename: str=None):
        self.encoder_filename = encoder_filename

    def fit_encoder(self, data: pd.DataFrame):
        if self.encoder_filename:
            encoder = StandardScaler()
            #encoder = MinMaxScaler()
            encoder.fit(data)
            joblib.dump(encoder, self.encoder_filename)
            return encoder
        else:
            return None
    
    def load_encoder(self):
        if self.encoder_filename:
            try:
                return joblib.load(self.encoder_filename)
            except FileNotFoundError as e:
                raise Exception("Error while loading encoder. Try running BaseEncoder.fit_encoder() or BaseEncoder.fit_and_encode() first to create a new encoder") from e
        else:
            return None
        
    def encode(self, data: pd.DataFrame) -> pd.DataFrame:
        encoder = self.load_encoder()
        return self._encode_data(data, encoder)

    def fit_and_encode(self, data: pd.DataFrame) -> pd.DataFrame:
        encoder = self.fit_encoder(data)
        return self._encode_data(data, encoder)
    
    def _encode_data(self, data, encoder):
        return data
        

class TemporalEncoder(BaseEncoder):
    def _encode_data(self, data: pd.DatetimeIndex, encoder):
        dataframes = []
        dataframes.append(pd.get_dummies(data.year, prefix='year', dtype=float))
        dataframes.append(SineCosineEncoder.create_encoding(data.month, 'month', 12))
        dataframes.append(SineCosineEncoder.create_encoding(data.day, 'day', 31))
        dataframes.append(SineCosineEncoder.create_encoding(data.dayofweek, 'weekday', 7))
        dataframes.append(SineCosineEncoder.create_encoding(data.hour, 'hour', 24))
        df = pd.concat(dataframes, axis=1).set_index(data)
        return df
    

class WeatherEncoder(BaseEncoder):
    def __init__(self, encoder_filename: str='saved_models/weather_encoding.save'):
        super().__init__(encoder_filename)
    
    def _encode_data(self, data: pd.DataFrame, encoder) -> pd.DataFrame:
        sin_cos_coding = SineCosineEncoder.create_encoding(data['wind_direction'], 'wind_direction', 360).set_index(data.index)
        encoding = pd.DataFrame(encoder.transform(data), index=data.index, columns=data.columns)
        encoding = pd.concat([sin_cos_coding, encoding], axis=1).drop(['wind_direction'], axis=1)
        return encoding

class SalesEncoder(BaseEncoder):
    def __init__(self, encoder_filename: str='saved_models/sales_encoding.save'):
        super().__init__(encoder_filename)

    def _encode_data(self, data: pd.DataFrame, encoder) -> pd.DataFrame:
        encoding = encoder.transform(data)
        encoding = pd.DataFrame(encoding, index=data.index, columns=data.columns)
        return encoding
    

class LabelEncoder(SalesEncoder):
    def __init__(self, encoder_filename: str='saved_models/label_encoding.save'):
        super().__init__(encoder_filename)

    def decode_data(self, data: pd.DataFrame):
        coder = self.load_encoder()
        coder.inverse_transform(data)
        return coder.inverse_transform(data)


class AutoEncoder(BaseEncoder):
    def __init__(self, encoder_filename: str='saved_models/autoencoder.h5'):
        super().__init__(encoder_filename)

    def fit_encoder(self, data: pd.DataFrame) -> pd.DataFrame:
        encoder = build_autoencoder(data.shape[1])
        train_model(data, encoder, self.encoder_filename)
        return encoder.get_layer('encoder')

    def load_encoder(self) -> tf.keras.models.Model:
        try:
            if self.encoder_filename:
                return tf.keras.models.load_model(self.encoder_filename).get_layer('encoder')
        except FileNotFoundError as e:
            raise Exception("Error while loading encoder. Try running BaseEncoder.fit_encoder() or BaseEncoder.fit_and_encode() first to create a new encoder") from e
    
    def _encode_data(self, data: pd.DataFrame, encoder: tf.keras.models.Model) -> pd.DataFrame:
        encoding = encoder.predict(data)
        columns = [f"{self.encoder_filename.split('/')[-1].split('.h5')[0]}_{i}" for i in range(encoding.shape[1])]
        encoding =  pd.DataFrame(encoding, index=data.index, columns=columns)
        return encoding
    

class FerienEncoder(AutoEncoder):
    def __init__(self, encoder_filename: str='saved_models/ferien_encoding.h5'):
        super().__init__(encoder_filename)


class FahrtenEncoder(AutoEncoder):
    def __init__(self, encoder_filename: str='saved_models/fahrten_encoding.h5'):
        super().__init__(encoder_filename)


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


class DataProcessor:
    def __init__(self, data: pd.DataFrame, future_days: int, reduce_one_hots: bool=False):
        self.future_days = future_days
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
            'datetime': TemporalEncoder(),
            'weather': WeatherEncoder(),
            'ferien': FerienEncoder() if reduce_one_hots else BaseEncoder(),
            'fahrten': FahrtenEncoder() if reduce_one_hots else BaseEncoder(),
            'is_open': BaseEncoder(),
            'labels': LabelEncoder(),
            'sales': SalesEncoder()
        }
        
        
    def _create_sales_features(self) -> None:
        sales_feature = self.data['labels'].copy().add_prefix(f'(t-{self.future_days})')
        sales_feature.index = sales_feature.index + pd.Timedelta(days=self.future_days)
        self.data['sales'] = sales_feature
        

    def get_uncoded_data(self):
        datetime_df = pd.DataFrame({'datetime': self.data['datetime']})

        # Extracting DataFrames for concatenation
        data_frames = [datetime_df] + [self.data[key] for key in ['weather', 'ferien', 'fahrten', 'is_open', 'labels', 'sales']]
      #  data_frames = [datetime_df] + [self.data[key] for key in self.data.keys() if key != 'datetime']

        # Merging the DataFrames using reduce
        from functools import reduce
        result_df = reduce(lambda left, right: pd.merge(left, right, on='datetime'), data_frames)
        result_df.set_index('datetime', inplace=True)
        result_df.to_csv("data/uncoded_data.csv")
        return result_df


    def fit_and_encode(self) -> pd.DataFrame:
        return self._encode_data(fit_encoder=True)
    
    def encode(self) -> pd.DataFrame:
        return self._encode_data(fit_encoder=False)
    
    def _encode_data(self, fit_encoder: bool) -> pd.DataFrame:
        encoded_data = {}
        for key, data in self.data.items():
            encoder = self.encoders[key]
            encoded_data[key] = encoder.fit_and_encode(data) if fit_encoder else encoder.encode(data)
            print(encoded_data[key])
            print(key, "encoder done")
            
        new_order = ['sales','datetime','is_open','weather','ferien','fahrten','labels']
        encoded_data = OrderedDict((key, encoded_data[key]) for key in new_order)
        return pd.concat(encoded_data.values(), axis=1).dropna()

    def decode_data(self, data: pd.DataFrame):
        decoder = self.encoders['labels']
        return pd.DataFrame(decoder.decode_data(data), columns=self.data['labels'].columns, index=data.index).astype(int)
    
    def get_shape(self):
        num_features = sum(len(value.columns) for key, value in self.data.items() if key not in ['datetime', 'labels'])
        num_targets = len(self.data['labels'].columns)
        return num_features, num_targets
        

if __name__ == "__main__":
    settings = Settings()
    configs = ProviderConfigs()
    provider = DataProvider(configs)
    provider.create_new_database(provider_list=['sales'])
    df = provider.load_database()
    processor = DataProcessor(data=df, future_days=settings.future_days)
    uncoded = processor.get_uncoded_data()
    
    try:
        encoding = processor.encode()
    except:
        encoding = processor.fit_and_encode()
    print(encoding)

