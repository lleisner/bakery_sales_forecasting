import pandas as pd
import joblib
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from collections import OrderedDict
from models.autoencoder.training import build_autoencoder, train_model
from data_provider.data_provider import DataProvider
from utils.configs import ProviderConfigs, Settings, ProcessorConfigs
import warnings

class BaseEncoder:
    def __init__(self, encoder_model="standard", encoder_filename: str=None):
        self.encoder_filename = encoder_filename
        self.encoder = MinMaxScaler() if encoder_model=="min_max" else StandardScaler()
        
    def fit_encoder(self, data: pd.DataFrame):

        try:
            self.encoder.fit(data)
        except:
            pass
        if self.encoder_filename:
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
    
    def _encode_data(self, data: pd.DataFrame) -> pd.DataFrame:
        encoding = self.encoder.transform(data)
        encoding = pd.DataFrame(encoding, index=data.index, columns=data.columns)
        return encoding
    
    def get_uncoded_data(self, data):
        return data
    
class SineCosineEncoder:
    @staticmethod
    def create_encoding(data: pd.Series, name: str) -> pd.DataFrame:
        """
        Create sine and cosine encodings for a column of cyclic data within the range of -0.5 to 0.5.

        Args:
            data (pd.Series): The DataFrame column containing the cyclic data.
            name (str): Name for the encoding columns.
            parameter_range (int): The range of the cyclic data (e.g. 12 for months).

        Returns:
            pd.DataFrame: A DataFrame containing two columns: 'sine_encoding' and 'cosine_encoding' scaled to the range [-0.5, 0,5].
        """
        data -= min(data)
        # Convert the column to radians
        radians = (data / max(data)) * 2 * np.pi

        # Calculate sine and cosine encodings
        sine_encoding = 0.5 * (np.sin(radians) + 1) - 0.5
        cosine_encoding = 0.5 * (np.cos(radians) + 1) - 0.5

        # Create a DataFrame with the encodings
        encoding_df = pd.DataFrame({f'{name}_sine_encoding': sine_encoding, f'{name}_cosine_encoding': cosine_encoding})

        return encoding_df
    
class TimeFeatureEncoder:
    @staticmethod
    def create_encoding(data: pd.Series, name: str) -> pd.DataFrame:
        data -= min(data)
        feature = data / max(data) - 0.5
        encoding_df = pd.DataFrame({name: feature})
        return encoding_df
    
class OneHotEncoder:
    @staticmethod
    def create_encoding(data: pd.Series, name: str) -> pd.DataFrame:
        return pd.get_dummies(data, prefix=name, dtype=float)
        

class TemporalEncoder(BaseEncoder):
    def __init__(self, encoder_model="standard"):
        self.encoder = SineCosineEncoder() if encoder_model=="sine_cosine" else TimeFeatureEncoder()
        self.one_hot = OneHotEncoder()
        self.encoder_filename = None
        
    def process_data(self, data: pd.DatetimeIndex, encode=True):
        encoding_specs = [
            ('hour', lambda x: x.hour),
            ('dayofweek', lambda x: x.dayofweek),
            ('dayofmonth', lambda x: x.day),
            ('dayofyear', lambda x: x.dayofyear), 
            ('month', lambda x: x.month), 
            ('quarter', lambda x: x.quarter)
            ]
        dataframes = []
        for name, value_func in encoding_specs:
            if encode:
                df = self.encoder.create_encoding(data=value_func(data), name=name)
            else:
                df = pd.DataFrame({name: value_func(data)})
                
            dataframes.append(df)
            
        return pd.concat(dataframes, axis=1).set_index(data)
    
    def _encode_data(self, data: pd.DatetimeIndex):
        return self.process_data(data, encode=True)
    
    def get_uncoded_data(self, data):
        return self.process_data(data, encode=False)

class WeatherEncoder(BaseEncoder):
    def __init__(self, encoder_model, encoder_filename: str='saved_models/weather_encoding.save'):
        super().__init__(encoder_model=encoder_model, encoder_filename=encoder_filename)
    
    def _encode_data(self, data: pd.DataFrame) -> pd.DataFrame:
        sin_cos_coding = SineCosineEncoder.create_encoding(data['wind_direction'], 'wind_direction').set_index(data.index)
        encoding = pd.DataFrame(self.encoder.transform(data), index=data.index, columns=data.columns)
        encoding = pd.concat([sin_cos_coding, encoding], axis=1).drop(['wind_direction'], axis=1)
        return encoding

class SalesEncoder(BaseEncoder):
    def __init__(self, encoder_model, encoder_filename: str='saved_models/sales_encoding.save'):
        super().__init__(encoder_model=encoder_model, encoder_filename=encoder_filename)

class LabelEncoder(SalesEncoder):
    def __init__(self, encoder_model, encoder_filename: str='saved_models/sales_encoding.save'):
        super().__init__(encoder_model=encoder_model, encoder_filename=encoder_filename)

    def decode_data(self, data: pd.DataFrame):
        self.load_encoder()
        return self.encoder.inverse_transform(data)


class DataEncoder:
    def __init__(self, configs):
        self.configs = configs
        self.encoders = {
            'datetime': TemporalEncoder(encoder_model=self.configs.temp_encoder),
            'weather': WeatherEncoder(encoder_model=self.configs.def_encoder),
            'ferien': FerienEncoder() if self.configs.reduce_one_hots else BaseEncoder(encoder_model=self.configs.def_encoder),
            'fahrten': FahrtenEncoder() if self.configs.reduce_one_hots else BaseEncoder(encoder_model=self.configs.def_encoder),
            'is_open': BaseEncoder(encoder_model=self.configs.def_encoder),
            'gaeste': BaseEncoder(encoder_model=self.configs.def_encoder),
            'labels': LabelEncoder(encoder_model=self.configs.def_encoder),
            'sales': SalesEncoder(encoder_model=self.configs.def_encoder)
        }
        self.shape = None
    
    def process_data(self, data: pd.DataFrame, encode: bool=True):
        data = self._categorize_data(data)
        num_targets = len(data['labels'].columns)
        
        data = self._encode_or_return_raw(data, encode)
        num_features = len(data.columns) - num_targets
        
        self.shape = (num_features, num_targets)
        return data
    
    def decode_data(self, data: pd.DataFrame, col_names=None):
        decoder = self.encoders['labels']
        try:
            return pd.DataFrame(decoder.decode_data(data), columns=col_names, index=data.index).astype(int)
        except:
            return decoder.decode_data(data).astype(int)
    
    def get_feature_target_nums(self, df):
        return (pd.to_numeric(df.columns, errors='coerce').isnull().sum(), pd.to_numeric(df.columns, errors='coerce').notnull().sum())

        
    def _categorize_data(self, data: pd.DataFrame):
        selected_keys = self.configs.covariate_selection
        grouped_data = {
            'datetime': data.index,
            'weather': data[['temperature', 'precipitation', 'cloud_cover', 'wind_speed', 'wind_direction']],
            'ferien': data[['BW', 'BY', 'BE', 'BB', 'HB', 'HH', 'HE', 'MV', 'NI', 'NW', 'RP', 'SL', 'SN', 'ST', 'SH', 'TH']] if not self.configs.aggregate else data[['holidays']],
            'fahrten': data[['SP1_an', 'SP2_an', 'SP4_an', 'SP1_ab', 'SP2_ab', 'SP4_ab']] if not self.configs.aggregate else data[['arrival', 'departure']],
            'gaeste': data[['gaestezahlen']],
            'is_open': data[['is_open']],
            'labels': data[[col for col in data.columns if str(col).isnumeric()]]
        }
        
        if self.configs.create_sales_features:
            grouped_data = self._create_sales_features(grouped_data)
            selected_keys.insert(0, "sales")
        
        selected_keys.append("labels")
        grouped_data = OrderedDict((key, grouped_data[key]) for key in selected_keys)
        return grouped_data
        
    
    def _create_sales_features(self, data):
        sales_feature = data['labels'].copy().add_prefix(f'(t-{self.configs.future_days})')
        sales_feature.index = sales_feature.index + pd.Timedelta(days=self.configs.future_days)
        data['sales'] = sales_feature
        return data
    
    def _encode_or_return_raw(self, data, encode=True):
        encoded_data = {}
        for key, df in data.items():
            encoder = self.encoders[key]
            encoded_data[key] = encoder.fit_and_encode(df) if encode else encoder.get_uncoded_data(df)
        return pd.concat(encoded_data.values(), axis=1).dropna()



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
        
        
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.utils.validation")

    settings = Settings()
    configs = ProviderConfigs()
    provider = DataProvider(configs)
    #provider.create_new_database()
    df = provider.load_database()
    p_configs = ProcessorConfigs(settings)
    encoder = DataEncoder(configs=p_configs)
    encoder_def = encoder.process_data(df, encode=False)
    print(encoder_def)
    print(encoder.get_feature_target_nums(encoder_def))
    encoder_enc = encoder.process_data(df, encode=True)
    print(encoder_enc)
    print(encoder_enc.columns)
    print(encoder.get_feature_target_nums(encoder_enc))
    
    

